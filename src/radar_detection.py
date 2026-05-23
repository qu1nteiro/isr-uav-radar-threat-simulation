"""
radar_detection.py — Stage 1: stochastic radar detection model

Implements the detection probability function and Bernoulli trial model
as defined in the mathematical formulation (Section 3, equations 4–10).

    P_i(t) = clip( exp(-d_i / lambda) + eps_i , 0, 1 )   if d_i <= R
           = 0                                             otherwise

    D_i(t) ~ Bernoulli( P_i^eff(t) )

    D(t)   = 1  if any D_i(t) = 1

Usage
-----
    from radar_detection import detection_probability, global_detection
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import config


# ─── RNG ──────────────────────────────────────────────────────────────────────

def _make_rng(seed=None):
    """Return a numpy default_rng, using config.RANDOM_SEED if seed is None."""
    if seed is None:
        seed = config.RANDOM_SEED
    return np.random.default_rng(seed)


# ─── Core detection functions ─────────────────────────────────────────────────

def detection_probability(d, lam=None, R=None, noise_std=None, rng=None):
    """
    Compute the stochastic detection probability for a single radar at distance d.

    Implements equation (4) of the formulation:
        P(d) = clip( exp(-d / lambda) + epsilon , 0, 1 )   if d <= R
             = 0                                            otherwise

    Parameters
    ----------
    d : float or np.ndarray
        Distance(s) between the UAV and the radar node [spatial units].
    lam : float, optional
        Decay length lambda [units]. Defaults to config.DETECTION_LAMBDA.
    R : float, optional
        Maximum radar detection radius [units]. Defaults to config.RADAR_RANGE.
    noise_std : float, optional
        Std of additive Gaussian noise epsilon. Defaults to config.DETECTION_NOISE_STD.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    float or np.ndarray
        Detection probability in [0, 1], same shape as d.
    """
    lam       = lam       if lam       is not None else config.DETECTION_LAMBDA
    R         = R         if R         is not None else config.RADAR_RANGE
    noise_std = noise_std if noise_std is not None else config.DETECTION_NOISE_STD

    if rng is None:
        rng = _make_rng()

    d = np.asarray(d, dtype=float)

    # Base exponential decay
    p_base = np.exp(-d / lam)

    # Additive sensor noise
    noise = rng.normal(0.0, noise_std, size=d.shape)

    # Clip to valid probability range
    p = np.clip(p_base + noise, 0.0, 1.0)

    # Zero probability outside detection radius
    p = np.where(d <= R, p, 0.0)

    return p


def detection_event(p, rng=None):
    """
    Perform a Bernoulli trial for a single radar detection event.

    Implements equation (9): D_i(t) ~ Bernoulli( P_i^eff(t) )

    Parameters
    ----------
    p : float or np.ndarray
        Detection probability in [0, 1].
    rng : np.random.Generator, optional

    Returns
    -------
    bool or np.ndarray of bool
        True if the radar detects the UAV at this step.
    """
    if rng is None:
        rng = _make_rng()

    p = np.asarray(p, dtype=float)
    return rng.random(size=p.shape) < p


def global_detection(uav_pos, radar_positions, alert_states=None,
                     lam=None, R=None, noise_std=None,
                     gamma=None, rng=None):
    """
    Check whether the UAV is detected by any radar at the current step.

    Implements equation (10):
        D(t) = 1  if exists i: D_i(t) = 1

    Each radar within range performs an independent Bernoulli trial.
    If alert_states is provided, the effective probability is amplified
    per equation (8): P_eff = clip( P_base * (1 + gamma * A_i), 0, 1 )

    Parameters
    ----------
    uav_pos : array-like, shape (2,)
        Current UAV position (x1, x2) [spatial units].
    radar_positions : np.ndarray, shape (N, 2)
        Positions of all N radar nodes [spatial units].
    alert_states : np.ndarray of int, shape (N,), optional
        Alert countdown c_i(t) for each radar. A_i = 1 if c_i > 0.
        If None, all radars are assumed to be in the base (non-alert) state.
    lam : float, optional
    R : float, optional
    noise_std : float, optional
    gamma : float, optional
        Alert amplification factor. Defaults to config.ALERT_GAMMA.
    rng : np.random.Generator, optional

    Returns
    -------
    detected : bool
        True if any radar detects the UAV.
    detecting_ids : np.ndarray of int
        Indices of radar nodes that fired a detection event this step.
    probabilities : np.ndarray, shape (N,)
        Effective detection probability for each radar at this step.
    """
    lam       = lam       if lam       is not None else config.DETECTION_LAMBDA
    R         = R         if R         is not None else config.RADAR_RANGE
    noise_std = noise_std if noise_std is not None else config.DETECTION_NOISE_STD
    gamma     = gamma     if gamma     is not None else config.ALERT_GAMMA

    if rng is None:
        rng = _make_rng()

    uav_pos        = np.asarray(uav_pos, dtype=float)
    radar_positions = np.asarray(radar_positions, dtype=float)
    N              = len(radar_positions)

    # Euclidean distances to all radars
    distances = np.linalg.norm(radar_positions - uav_pos, axis=1)   # shape (N,)

    # Base detection probabilities
    p_base = detection_probability(distances, lam=lam, R=R,
                                   noise_std=noise_std, rng=rng)

    # Alert amplification: P_eff = clip(P_base * (1 + gamma * A_i), 0, 1)
    if alert_states is not None:
        alert_states = np.asarray(alert_states, dtype=int)
        A = (alert_states > 0).astype(float)                        # binary alert flag
        p_eff = np.clip(p_base * (1.0 + gamma * A), 0.0, 1.0)
    else:
        p_eff = p_base

    # Independent Bernoulli trials
    fired = detection_event(p_eff, rng=rng)

    detecting_ids = np.where(fired)[0]
    detected      = len(detecting_ids) > 0

    return detected, detecting_ids, p_eff


# ─── Characterisation ─────────────────────────────────────────────────────────

def p_curve(d_max=None, n_points=300, n_samples=500,
            lam=None, R=None, noise_std=None, seed=None):
    """
    Compute the mean and std of P(d) over many noise realisations.

    Parameters
    ----------
    d_max : float
        Maximum distance to evaluate [units]. Defaults to 1.5 * RADAR_RANGE.
    n_points : int
        Number of distance values.
    n_samples : int
        Number of noise realisations per distance (for mean/std band).
    lam, R, noise_std : float, optional
    seed : int, optional

    Returns
    -------
    distances : np.ndarray, shape (n_points,)
    p_mean    : np.ndarray, shape (n_points,)
    p_std     : np.ndarray, shape (n_points,)
    p_noiseless : np.ndarray, shape (n_points,)
        Pure exponential without noise, for reference.
    """
    R     = R     if R     is not None else config.RADAR_RANGE
    lam   = lam   if lam   is not None else config.DETECTION_LAMBDA
    d_max = d_max if d_max is not None else 1.5 * R

    rng       = _make_rng(seed)
    distances = np.linspace(0.0, d_max, n_points)

    # n_samples realisations for each distance
    p_samples = np.stack([
        detection_probability(distances, lam=lam, R=R,
                              noise_std=noise_std, rng=rng)
        for _ in range(n_samples)
    ])                                                               # (n_samples, n_points)

    p_mean      = p_samples.mean(axis=0)
    p_std       = p_samples.std(axis=0)
    p_noiseless = np.where(distances <= R, np.exp(-distances / lam), 0.0)

    return distances, p_mean, p_std, p_noiseless


def first_detection_distribution(n_trials=5000, radar_pos=None,
                                  domain_size=None, seed=None):
    """
    Sample the distribution of first-detection distances.

    Physical model: the UAV approaches a single radar from a random angle,
    starting just outside the detection radius R. It advances one step at a
    time toward the radar. The distance at which the first Bernoulli trial
    succeeds is recorded. Trials that reach the radar without detection
    are counted as misses.

    This models the physically meaningful question: when entering radar
    coverage on a direct approach, at what distance is the UAV first caught?

    Parameters
    ----------
    n_trials : int
        Number of Monte Carlo trajectories.
    radar_pos : array-like, shape (2,), optional
        Radar position. Defaults to domain centre (L/2, L/2).
    domain_size : float, optional
        Defaults to config.GRID_SIZE.
    seed : int, optional

    Returns
    -------
    detected_distances : np.ndarray
        Distances at first detection, one entry per detected trial.
    miss_fraction : float
        Fraction of trials that reached d ≈ 0 without being detected.
    """
    L         = domain_size if domain_size is not None else config.GRID_SIZE
    R         = config.RADAR_RANGE
    step      = 0.1         # approach step size [units]
    rng       = _make_rng(seed)
    radar_pos = np.array(radar_pos if radar_pos is not None else [L / 2, L / 2])

    detected_distances = []
    miss_count         = 0

    for _ in range(n_trials):
        # UAV starts just outside radar range at a random angle
        theta = rng.uniform(0.0, 2 * np.pi)
        uav   = radar_pos + (R + 0.1) * np.array([np.cos(theta), np.sin(theta)])

        # Unit vector pointing from UAV toward radar
        direction = radar_pos - uav
        direction = direction / np.linalg.norm(direction)

        detected = False
        d        = np.linalg.norm(uav - radar_pos)

        while d > step:
            uav = uav + step * direction
            d   = np.linalg.norm(uav - radar_pos)
            p   = detection_probability(d, rng=rng)
            if detection_event(p, rng=rng):
                detected_distances.append(d)
                detected = True
                break

        if not detected:
            miss_count += 1

    miss_fraction = miss_count / n_trials
    return np.array(detected_distances), miss_fraction


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_detection_model(save=True, seed=None):
    """
    Produce the Stage 1 characterisation figure (2 panels):
        Left  — P(d) curve with noise band, noiseless reference, R boundary
        Right — histogram of first-detection distances
    """
    rng = _make_rng(seed)

    # ── data ──────────────────────────────────────────────────────────────────
    d_vals, p_mean, p_std, p_noise_free = p_curve(seed=seed)

    det_dists, misses = first_detection_distribution(
        n_trials=8000, seed=seed
    )
    hit_rate = (1.0 - misses) * 100   # misses is now a fraction

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # ── left panel: P(d) ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    ax1.fill_between(d_vals,
                     np.clip(p_mean - p_std, 0, 1),
                     np.clip(p_mean + p_std, 0, 1),
                     alpha=0.18, color="#2563eb", label=r"$\pm 1\sigma$ noise band")
    ax1.plot(d_vals, p_mean,       color="#2563eb", lw=1.8, label=r"$\bar{P}(d)$ (mean over noise)")
    ax1.plot(d_vals, p_noise_free, color="#64748b", lw=1.2,
             linestyle="--", label=r"$e^{-d/\lambda}$ (noiseless)")

    ax1.axvline(config.RADAR_RANGE, color="#dc2626", lw=1.0,
                linestyle=":", label=f"$R = {config.RADAR_RANGE}$ u")
    ax1.axhline(np.exp(-config.RADAR_RANGE / config.DETECTION_LAMBDA),
                color="#94a3b8", lw=0.7, linestyle=":")

    ax1.set_xlabel("Distance  $d$  [spatial units]", fontsize=10)
    ax1.set_ylabel("Detection probability  $P(d)$",  fontsize=10)
    ax1.set_title("Detection probability profile", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, d_vals[-1])
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    # boundary annotation
    ax1.annotate(f"$P(R) \\approx {np.exp(-config.RADAR_RANGE/config.DETECTION_LAMBDA):.2f}$",
                 xy=(config.RADAR_RANGE, np.exp(-config.RADAR_RANGE / config.DETECTION_LAMBDA)),
                 xytext=(config.RADAR_RANGE + 0.8, 0.28),
                 fontsize=8, color="#dc2626",
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.8))

    # ── right panel: first-detection histogram ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    if len(det_dists) > 0:
        ax2.hist(det_dists, bins=40, color="#2563eb", alpha=0.75,
                 edgecolor="white", linewidth=0.4, density=True)
        ax2.axvline(det_dists.mean(), color="#dc2626", lw=1.2,
                    linestyle="--",
                    label=f"mean = {det_dists.mean():.2f} u")
        ax2.axvline(np.median(det_dists), color="#f97316", lw=1.2,
                    linestyle=":",
                    label=f"median = {np.median(det_dists):.2f} u")

    ax2.set_xlabel("First-detection distance  $d$  [spatial units]", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Distribution of first-detection distances", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8.5, framealpha=0.9)
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    hit_label = f"Detection rate: {hit_rate:.1f}%  ({len(det_dists)}/{len(det_dists)+misses} trials)"
    ax2.text(0.97, 0.96, hit_label,
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=8, color="#475569",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e1", lw=0.7))

    # ── save ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Stage 1 — Detection model   "
        f"($\\lambda={config.DETECTION_LAMBDA}$, "
        f"$R={config.RADAR_RANGE}$, "
        f"$\\sigma={config.DETECTION_NOISE_STD}$)",
        fontsize=11
    )



    if save:
        path = os.path.join(config.FIGURES_DIR, "stage1_detection_model.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Stage 1] Figure saved → {path}")

    return fig


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_limits(verbose=True):
    """
    Sanity-check the known analytic limits of the detection model.

        P(0)      → exp(0) = 1.0   (drone on top of radar → certain detection)
        P(R)      → exp(-R/λ) ≈ 0.19   (boundary detection probability)
        P(R + ε)  → 0.0            (outside range → no detection)

    Returns True if all checks pass.
    """
    rng = _make_rng(0)
    n   = 10_000    # large sample to suppress noise

    # At d = 0: mean P should be close to 1
    p_zero = detection_probability(np.zeros(n), noise_std=0.0, rng=rng).mean()

    # At d = R: mean P should be exp(-R/λ)
    p_boundary = detection_probability(
        np.full(n, config.RADAR_RANGE), noise_std=0.0, rng=rng
    ).mean()
    expected_boundary = np.exp(-config.RADAR_RANGE / config.DETECTION_LAMBDA)

    # At d = R + small epsilon: P must be 0
    p_outside = detection_probability(
        np.full(n, config.RADAR_RANGE + 0.01), noise_std=0.0, rng=rng
    ).mean()

    checks = {
        "P(d=0) ≈ 1.0":              (abs(p_zero       - 1.0)              < 0.01),
        "P(d=R) ≈ exp(-R/λ)":        (abs(p_boundary   - expected_boundary) < 0.01),
        "P(d=R+ε) = 0.0":            (abs(p_outside)                        < 1e-9),
    }

    all_pass = True
    if verbose:
        print("─── Detection model validation ───────────────────────")
        for label, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status}  {label}")
        print(f"  P(d=0)  computed : {p_zero:.4f}")
        print(f"  P(d=R)  computed : {p_boundary:.4f}   expected : {expected_boundary:.4f}")
        print(f"  P(d>R)  computed : {p_outside:.6f}")
        print("──────────────────────────────────────────────────────")

    all_pass = all(checks.values())
    return all_pass


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Stage 1 — radar detection model\n")

    ok = validate_limits(verbose=True)
    if not ok:
        print("\n[WARNING] One or more validation checks failed.")
    else:
        print("\n[OK] All validation checks passed.")

    print("\nGenerating characterisation figure...")
    plot_detection_model(save=True, seed=config.RANDOM_SEED)
    plt.show()