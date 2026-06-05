"""
uav_trajectory.py — Stage 3: UAV trajectory simulation.

Implements the full mission simulation integrating:
  - Biased random walk / Lévy flight (equations 1-2, formulation)
  - Stochastic radar detection (Stage 1 — radar_detection.py)
  - Alert propagation through the radar network (Stage 2 — network_builder.py)
  - Alert state dynamics (equations 5-8, formulation)

Public API (imported by Stages 4 and 5)
-----------------------------------------
    run_mission(radar_positions, G, rng, walk='gaussian')  →  MissionResult
    run_ensemble(radar_positions, G, rng, n, walk)         →  EnsembleStats

Usage
-----
    python src/uav_trajectory.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass, field
from typing import Optional
import os

import config
from radar_detection import detection_probability, detection_event
from network_builder import build_network, place_radars, network_metrics, TOPO_COLOURS, TOPO_LABELS


# ─── Result containers ────────────────────────────────────────────────────────

@dataclass
class MissionResult:
    """
    Complete record of a single UAV mission simulation.

    Attributes
    ----------
    outcome : str
        'success' | 'detected' | 'timeout'
    n_steps : int
        Number of steps taken before mission ended.
    positions : np.ndarray, shape (n_steps+1, 2)
        UAV position at every step, including start.
    p_max_history : np.ndarray, shape (n_steps+1,)
        Maximum effective P across all radars at each step.
    detection_step : int or None
        Step at which detection occurred, or None.
    detection_pos : tuple or None
        (x, y) at detection step, or None.
    detecting_radar : int or None
        Index of the radar that triggered detection, or None.
    alert_history : np.ndarray, shape (n_steps+1, N_radars)
        Alert countdown c_i(t) for every radar at every step.
    walk_type : str
        'gaussian' or 'levy'
    topology : str
        Network topology used ('ER', 'BA', 'WS', or 'none').
    step_sizes : np.ndarray
        Actual step sizes taken (useful for Lévy distribution analysis).
    """
    outcome:          str
    n_steps:          int
    positions:        np.ndarray
    p_max_history:    np.ndarray
    detection_step:   Optional[int]
    detection_pos:    Optional[tuple]
    detecting_radar:  Optional[int]
    alert_history:    np.ndarray
    walk_type:        str
    topology:         str
    step_sizes:       np.ndarray


@dataclass
class EnsembleStats:
    """
    Aggregate statistics over many mission simulations.

    Attributes
    ----------
    n_missions : int
    success_rate : float          S = successes / n_missions
    detection_rate : float
    timeout_rate : float
    mean_survival_steps : float   Mean steps before detection (detected only)
    survival_curve : np.ndarray   Fraction surviving at each step t
    outcome_counts : dict
    walk_type : str
    topology : str
    all_step_sizes : np.ndarray   Pooled step sizes across all missions
    """
    n_missions:          int
    success_rate:        float
    detection_rate:      float
    timeout_rate:        float
    mean_survival_steps: float
    survival_curve:      np.ndarray
    outcome_counts:      dict
    walk_type:           str
    topology:            str
    all_step_sizes:      np.ndarray


# ─── Step samplers ────────────────────────────────────────────────────────────

def _gaussian_step(rng):
    """Fixed step size s (deterministic magnitude, equation 2)."""
    return config.UAV_STEP_SIZE


def _levy_step(rng):
    """
    Sample step size from a Lévy-stable distribution via the
    inverse CDF method for a power-law:

        P(s) ∝ s^{-(μ+1)},  s ∈ [s_min, s_max]

    Inverse CDF: s = s_min * (1 - U)^{-1/μ}, clipped to s_max.
    (Clauset et al., 2009)
    """
    u = rng.uniform(0.0, 1.0)
    s = config.LEVY_S_MIN * (1.0 - u) ** (-1.0 / config.LEVY_ALPHA)
    return float(np.clip(s, config.LEVY_S_MIN, config.LEVY_S_MAX))


_STEP_SAMPLERS = {
    "gaussian": _gaussian_step,
    "levy":     _levy_step,
}


# ─── Alert propagation ────────────────────────────────────────────────────────

def _propagate_alerts(G, detecting_ids, alert_counts):
    """
    Propagate detection alerts through the radar network.

    For every radar i that fired a detection event this step,
    set the alert countdown of all neighbours j ∈ N(i) to TAU.
    Also reset the detecting radar itself.

    Implements equations (5)-(7) of the formulation.

    Parameters
    ----------
    G : nx.Graph
        Radar communication network.
    detecting_ids : array-like of int
        Indices of radars that detected the UAV this step.
    alert_counts : np.ndarray, shape (N,)
        Alert countdown c_i(t), modified in-place.
    """
    for i in detecting_ids:
        alert_counts[i] = config.ALERT_TAU
        for j in G.neighbors(i):
            alert_counts[j] = config.ALERT_TAU


def _decay_alerts(alert_counts):
    """
    Decrement all non-zero alert countdowns by 1.
    Implements the 'otherwise' branch of equation (7).
    """
    alert_counts[alert_counts > 0] -= 1


# ─── Direction vector ─────────────────────────────────────────────────────────

def _direction_vector(uav_pos, rng, known_radars=None, beta=None):
    """
    Compute the normalised displacement direction vector.

    Implements equation (1) for the base walk, and equation (13)
    for the passive sensing extension (Stage 5).

    Parameters
    ----------
    uav_pos : np.ndarray, shape (2,)
    rng : np.random.Generator
    known_radars : np.ndarray, shape (K, 2), optional
        Positions of passively detected radars (Stage 5 only).
    beta : float, optional
        Passive evasion weight (Stage 5 only).

    Returns
    -------
    direction : np.ndarray, shape (2,)
        Unit vector.
    """
    tx, ty = config.MISSION_TARGET
    target = np.array([tx, ty])

    # Unit vector toward target
    diff = target - uav_pos
    dist = np.linalg.norm(diff)
    e_target = diff / dist if dist > 1e-9 else np.zeros(2)

    # Random unit vector
    theta  = rng.uniform(0.0, 2.0 * np.pi)
    e_rand = np.array([np.cos(theta), np.sin(theta)])

    v = config.DRIFT_WEIGHT * e_target + (1.0 - config.DRIFT_WEIGHT) * e_rand

    # Passive evasion term (Stage 5)
    if known_radars is not None and beta is not None and len(known_radars) > 0:
        for rp in known_radars:
            diff_r = rp - uav_pos
            dist_r = np.linalg.norm(diff_r)
            if dist_r > 1e-9:
                e_rep = diff_r / dist_r   # points toward radar
                v -= beta * e_rep         # subtract = repel

    # Normalise
    v_norm = np.linalg.norm(v)
    return v / v_norm if v_norm > 1e-9 else e_rand


# ─── Single mission ───────────────────────────────────────────────────────────

def run_mission(radar_positions, G, rng,
                walk="gaussian",
                known_radars=None, beta=None):
    """
    Simulate a single UAV mission from start to target or failure.

    Parameters
    ----------
    radar_positions : np.ndarray, shape (N, 2)
    G : nx.Graph
        Radar communication network (from Stage 2).
    rng : np.random.Generator
    walk : str
        'gaussian' or 'levy'
    known_radars : np.ndarray, optional
        For Stage 5 passive sensing extension.
    beta : float, optional
        Passive evasion weight (Stage 5).

    Returns
    -------
    MissionResult
    """
    if walk not in _STEP_SAMPLERS:
        raise ValueError(f"walk must be 'gaussian' or 'levy', got '{walk}'")

    N          = len(radar_positions)
    step_fn    = _STEP_SAMPLERS[walk]
    topo       = G.nodes[0].get("topo", "unknown") if G.number_of_nodes() > 0 else "none"

    # Initialise state
    pos         = np.array(config.UAV_START, dtype=float)
    alert_cnts  = np.zeros(N, dtype=int)    # c_i(t) for all radars

    positions     = [pos.copy()]
    p_max_hist    = [0.0]
    alert_hist    = [alert_cnts.copy()]
    step_sizes    = []

    outcome        = "timeout"
    detection_step = None
    detection_pos  = None
    detecting_rad  = None

    for t in range(config.MAX_STEPS):

        # ── Check mission success ──────────────────────────────────────────────
        dist_to_target = np.linalg.norm(pos - np.array(config.MISSION_TARGET))
        if dist_to_target <= config.TARGET_RADIUS:
            outcome = "success"
            break

        # ── Move ──────────────────────────────────────────────────────────────
        s         = step_fn(rng)
        direction = _direction_vector(pos, rng, known_radars, beta)
        pos       = np.clip(pos + s * direction, 0.0, config.GRID_SIZE)
        step_sizes.append(s)

        # ── Detection check (with alert amplification) ─────────────────────────
        detected, det_ids, p_eff = _detection_check(pos, radar_positions,
                                                     alert_cnts, rng)
        p_max = float(p_eff.max()) if len(p_eff) > 0 else 0.0

        # ── Alert propagation ──────────────────────────────────────────────────
        if len(det_ids) > 0:
            _propagate_alerts(G, det_ids, alert_cnts)

        # ── Alert decay ────────────────────────────────────────────────────────
        _decay_alerts(alert_cnts)

        # ── Record state ───────────────────────────────────────────────────────
        positions.append(pos.copy())
        p_max_hist.append(p_max)
        alert_hist.append(alert_cnts.copy())

        # ── Mission failure ────────────────────────────────────────────────────
        if detected:
            outcome        = "detected"
            detection_step = t + 1
            detection_pos  = tuple(pos)
            detecting_rad  = int(det_ids[0])
            break

    return MissionResult(
        outcome          = outcome,
        n_steps          = len(positions) - 1,
        positions        = np.array(positions),
        p_max_history    = np.array(p_max_hist),
        detection_step   = detection_step,
        detection_pos    = detection_pos,
        detecting_radar  = detecting_rad,
        alert_history    = np.array(alert_hist),
        walk_type        = walk,
        topology         = topo,
        step_sizes       = np.array(step_sizes),
    )


def _detection_check(uav_pos, radar_positions, alert_cnts, rng):
    """
    Vectorised detection check with alert amplification.
    Wraps global_detection() from radar_detection.py.
    """
    from radar_detection import global_detection
    return global_detection(
        uav_pos          = uav_pos,
        radar_positions  = radar_positions,
        alert_states     = alert_cnts,
        rng              = rng,
    )


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def run_ensemble(radar_positions, G, rng,
                 n=None, walk="gaussian"):
    """
    Run N independent missions and aggregate statistics.

    Parameters
    ----------
    radar_positions : np.ndarray, shape (N_radars, 2)
    G : nx.Graph
    rng : np.random.Generator
    n : int, optional
        Number of missions. Defaults to config.N_TRAJECTORIES.
    walk : str
        'gaussian' or 'levy'

    Returns
    -------
    EnsembleStats
    results : list of MissionResult
        All individual mission records.
    """
    n    = n or config.N_TRAJECTORIES
    topo = G.nodes[0].get("topo", "unknown") if G.number_of_nodes() > 0 else "none"

    outcomes      = []
    survival_arr  = np.zeros(config.MAX_STEPS + 1)
    all_steps     = []
    det_steps     = []

    results = []

    for _ in range(n):
        r = run_mission(radar_positions, G, rng, walk=walk)
        results.append(r)
        outcomes.append(r.outcome)
        all_steps.extend(r.step_sizes.tolist())

        # Survival curve: mark all steps up to (but not including) detection
        survive_until = r.detection_step if r.outcome == "detected" else r.n_steps
        survival_arr[:survive_until + 1] += 1

        if r.outcome == "detected" and r.detection_step is not None:
            det_steps.append(r.detection_step)

    counts = {
        "success":  outcomes.count("success"),
        "detected": outcomes.count("detected"),
        "timeout":  outcomes.count("timeout"),
    }

    survival_curve = survival_arr / n

    return EnsembleStats(
        n_missions          = n,
        success_rate        = counts["success"]  / n,
        detection_rate      = counts["detected"] / n,
        timeout_rate        = counts["timeout"]  / n,
        mean_survival_steps = float(np.mean(det_steps)) if det_steps else float(config.MAX_STEPS),
        survival_curve      = survival_curve,
        outcome_counts      = counts,
        walk_type           = walk,
        topology            = topo,
        all_step_sizes      = np.array(all_steps),
    ), results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _heatmap(radar_positions, resolution=180):
    L  = config.GRID_SIZE
    xs = np.linspace(0, L, resolution)
    ys = np.linspace(0, L, resolution)
    Xg, Yg = np.meshgrid(xs, ys)
    P_surv = np.ones((resolution, resolution))
    for rp in radar_positions:
        D  = np.sqrt((Xg - rp[0])**2 + (Yg - rp[1])**2)
        Pi = np.where(D <= config.RADAR_RANGE,
                      np.exp(-D / config.DETECTION_LAMBDA), 0.0)
        P_surv *= (1.0 - Pi)
    return Xg, Yg, 1.0 - P_surv


def _draw_trajectory_panel(ax, results, radar_positions, Xg, Yg, P_field,
                            title, n_sample=40):
    """Draw trajectory overview on a single axes."""
    cmap_danger = LinearSegmentedColormap.from_list(
        "danger",
        [(0.0, "#0a0a0a00"), (0.1, "#7f1d1d18"),
         (0.5, "#991b1b70"), (1.0, "#dc2626bb")], N=256
    )
    ax.pcolormesh(Xg, Yg, P_field,
                  cmap=cmap_danger, vmin=0, vmax=1,
                  shading="gouraud", rasterized=True, zorder=1)

    for rp in radar_positions:
        ax.add_patch(plt.Circle(rp, config.RADAR_RANGE,
                                fill=False, edgecolor="#dc262660",
                                linewidth=0.6, linestyle="--", zorder=2))

    ax.scatter(radar_positions[:, 0], radar_positions[:, 1],
               s=28, c="#A32D2D", marker="^", zorder=4,
               edgecolors="#fff", linewidths=0.5)

    colours = {"success": "#0F6E56", "detected": "#993C1D", "timeout": "#5F5E5A"}
    sample  = results[:n_sample]

    for r in sample:
        col = colours[r.outcome]
        ax.plot(r.positions[:, 0], r.positions[:, 1],
                color=col, lw=0.7, alpha=0.55, zorder=3)
        if r.outcome == "detected" and r.detection_pos:
            ax.scatter(*r.detection_pos, s=22, c=col,
                       marker="x", linewidths=1.1, zorder=5)

    ax.scatter(*config.UAV_START,      s=70, c="#185FA5", marker="o",
               zorder=6, edgecolors="#fff", linewidths=1.2)
    ax.add_patch(plt.Circle(config.MISSION_TARGET, config.TARGET_RADIUS,
                            fill=True, facecolor="#3B6D1133",
                            edgecolor="#3B6D11", linewidth=1.2, zorder=5))
    ax.scatter(*config.MISSION_TARGET, s=90, c="#3B6D11",
               marker="*", zorder=6, edgecolors="#fff", linewidths=0.8)

    n  = len(results)
    cs = {k: sum(1 for r in results if r.outcome == k) for k in colours}
    legend_els = [
        mpatches.Patch(fc=colours["success"],  ec="none",
                       label=f"Success  {cs['success']}/{n} "
                             f"({cs['success']/n*100:.0f}%)"),
        mpatches.Patch(fc=colours["detected"], ec="none",
                       label=f"Detected  {cs['detected']}/{n} "
                             f"({cs['detected']/n*100:.0f}%)"),
        mpatches.Patch(fc=colours["timeout"],  ec="none",
                       label=f"Timeout  {cs['timeout']}/{n} "
                             f"({cs['timeout']/n*100:.0f}%)"),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc="upper left",
              framealpha=0.9, edgecolor="#cbd5e1")
    ax.set_xlim(0, config.GRID_SIZE); ax.set_ylim(0, config.GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$ [u]", fontsize=9)
    ax.set_ylabel("$x_2$ [u]", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(True, linewidth=0.25, alpha=0.35)


def plot_trajectory_analysis(radar_positions, graphs, seed=None,
                             n_ensemble=None, save=True):
    """
    Produce the Stage 3 figure — 4 panels:

        [0,0] Sample trajectories — Gaussian walk
        [0,1] Sample trajectories — Lévy flight
        [1,0] Survival curves by topology (Gaussian walk)
        [1,1] Step-size distributions: Gaussian vs Lévy
    """
    n_ensemble = n_ensemble or config.N_TRAJECTORIES
    seed       = seed       if seed is not None else config.RANDOM_SEED
    rng        = np.random.default_rng(seed)

    print(f"  Running ensemble  ({n_ensemble} missions × 2 walks × "
          f"{len(graphs)} topologies)...")

    # ── Heatmap (shared) ──────────────────────────────────────────────────────
    Xg, Yg, P_field = _heatmap(radar_positions)

    # ── Run ensembles ─────────────────────────────────────────────────────────
    # Use ER graph for the trajectory sample panels (representative)
    G_er    = graphs["ER"]
    stats_g, res_g = run_ensemble(radar_positions, G_er, rng,
                                   n=n_ensemble, walk="gaussian")
    stats_l, res_l = run_ensemble(radar_positions, G_er, rng,
                                   n=n_ensemble, walk="levy")

    # Survival curves — Gaussian walk across all three topologies
    surv_by_topo = {}
    for topo, G in graphs.items():
        st, _ = run_ensemble(radar_positions, G, rng,
                             n=n_ensemble, walk="gaussian")
        surv_by_topo[topo] = st

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  {'Topology':<8} {'Walk':<10} {'Success':>8} "
          f"{'Detected':>10} {'Timeout':>9}")
    print("  " + "─" * 50)
    for label, st in [("ER (G)", stats_g), ("ER (L)", stats_l)]:
        print(f"  {label:<8} {'':10} "
              f"{st.success_rate*100:>7.1f}%"
              f"{st.detection_rate*100:>9.1f}%"
              f"{st.timeout_rate*100:>9.1f}%")
    for topo, st in surv_by_topo.items():
        print(f"  {topo:<8} {'Gaussian':10} "
              f"{st.success_rate*100:>7.1f}%"
              f"{st.detection_rate*100:>9.1f}%"
              f"{st.timeout_rate*100:>9.1f}%")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # ── Panel [0,0]: Gaussian trajectories ────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    _draw_trajectory_panel(ax00, res_g, radar_positions, Xg, Yg, P_field,
                           f"Gaussian walk — ER network\n"
                           f"$\\alpha={config.DRIFT_WEIGHT}$, "
                           f"$s={config.UAV_STEP_SIZE}$ u/step",
                           n_sample=50)

    # ── Panel [0,1]: Lévy trajectories ────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    _draw_trajectory_panel(ax01, res_l, radar_positions, Xg, Yg, P_field,
                           f"Lévy flight — ER network\n"
                           f"$\\mu={config.LEVY_ALPHA}$, "
                           f"$s_{{\\min}}={config.LEVY_S_MIN}$, "
                           f"$s_{{\\max}}={config.LEVY_S_MAX}$",
                           n_sample=50)

    # ── Panel [1,0]: Survival curves by topology ──────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    steps = np.arange(config.MAX_STEPS + 1)

    for topo, st in surv_by_topo.items():
        ax10.plot(steps, st.survival_curve * 100,
                  color=TOPO_COLOURS[topo], lw=1.8,
                  label=f"{TOPO_LABELS[topo]}  "
                        f"(S={st.success_rate*100:.1f}%)")

    ax10.set_xlabel("Simulation step  $t$", fontsize=9)
    ax10.set_ylabel("Fraction undetected  [%]", fontsize=9)
    ax10.set_title("Survival curves — Gaussian walk\nby radar network topology",
                   fontsize=9, fontweight="bold")
    ax10.set_xlim(0, config.MAX_STEPS)
    ax10.set_ylim(0, 105)
    ax10.legend(fontsize=8.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax10.grid(True, linewidth=0.3, alpha=0.4)

    # Add success rate annotation
    ax10.axvline(config.MAX_STEPS, color="#94a3b8",
                 lw=0.8, linestyle=":", alpha=0.6)
    ax10.text(config.MAX_STEPS - 5, 102, "$T_{\\max}$",
              ha="right", fontsize=8, color="#64748b")

    # ── Panel [1,1]: Step-size distributions ──────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])

    # Gaussian: all steps are exactly s (deterministic)
    gauss_steps = stats_g.all_step_sizes
    levy_steps  = stats_l.all_step_sizes

    bins = np.logspace(np.log10(config.LEVY_S_MIN),
                       np.log10(config.LEVY_S_MAX + 1), 45)

    ax11.hist(gauss_steps, bins=bins, density=True,
              color="#185FA5", alpha=0.65,
              label=f"Gaussian  ($s={config.UAV_STEP_SIZE}$ u/step, fixed)",
              edgecolor="white", linewidth=0.3)
    ax11.hist(levy_steps, bins=bins, density=True,
              color="#993C1D", alpha=0.65,
              label=f"Lévy  ($\\mu={config.LEVY_ALPHA}$)",
              edgecolor="white", linewidth=0.3)

    # Theoretical Lévy slope reference
    s_ref  = np.logspace(np.log10(config.LEVY_S_MIN),
                         np.log10(config.LEVY_S_MAX), 100)
    p_ref  = s_ref ** (-(config.LEVY_ALPHA + 1))
    p_ref /= p_ref.max()
    p_ref *= ax11.get_ylim()[1] if ax11.get_ylim()[1] > 0 else 1.0

    ax11.set_xscale("log")
    ax11.set_yscale("log")
    ax11.set_xlabel("Step size  $s$  [spatial units]", fontsize=9)
    ax11.set_ylabel("Density", fontsize=9)
    ax11.set_title("Step-size distributions\nGaussian vs Lévy flight",
                   fontsize=9, fontweight="bold")
    ax11.legend(fontsize=8.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax11.grid(True, linewidth=0.3, alpha=0.4, which="both")
    ax11.annotate(f"Lévy tail: $P(s) \\propto s^{{-{config.LEVY_ALPHA+1}}}$",
                  xy=(0.97, 0.92), xycoords="axes fraction",
                  ha="right", fontsize=8, color="#993C1D",
                  bbox=dict(boxstyle="round,pad=0.3",
                            fc="white", ec="#f0957a", lw=0.6))

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Stage 3 — UAV trajectory simulation   "
        f"($N_{{\\mathrm{{radars}}}}={len(radar_positions)}$, "
        f"$N_{{\\mathrm{{missions}}}}={n_ensemble}$, seed {seed})",
        fontsize=11, fontweight="500"
    )

    if save:
        path = os.path.join(config.FIGURES_DIR, "stage3_trajectories.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[Stage 3] Figure saved → {path}")

    return fig, stats_g, stats_l, surv_by_topo


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_mission(radar_positions, G, verbose=True):
    """
    Sanity checks on a single mission.

    Checks:
    - Outcome is one of the three valid states
    - Positions stay within domain bounds
    - Alert history has correct shape
    - Step sizes are within physical bounds
    - Detection step is consistent with outcome
    """
    rng = np.random.default_rng(0)
    r   = run_mission(radar_positions, G, rng, walk="gaussian")
    N   = len(radar_positions)

    checks = {
        "Valid outcome":
            r.outcome in ("success", "detected", "timeout"),
        "Positions within domain":
            bool(np.all(r.positions >= 0) and
                 np.all(r.positions <= config.GRID_SIZE)),
        "Alert history shape correct":
            r.alert_history.shape == (r.n_steps + 1, N),
        "Step sizes within bounds":
            bool(np.all(r.step_sizes >= 0) and
                 np.all(r.step_sizes <= config.LEVY_S_MAX + 1e-6)),
        "Detection step consistent":
            (r.outcome == "detected") == (r.detection_step is not None),
        "Positions array length correct":
            len(r.positions) == r.n_steps + 1,
    }

    all_pass = all(checks.values())

    if verbose:
        print("\n─── Mission validation ───────────────────────────────")
        for label, passed in checks.items():
            print(f"  {'✓' if passed else '✗'}  {label}")
        print(f"  Outcome: {r.outcome} | Steps: {r.n_steps} | "
              f"Walk: {r.walk_type} | Topo: {r.topology}")
        print("──────────────────────────────────────────────────────")

    return all_pass, r


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Stage 3 — UAV trajectory simulation\n")

    rng             = np.random.default_rng(config.RANDOM_SEED)
    radar_positions = place_radars(rng=rng)

    # Build all three networks on the same radar positions
    graphs = {t: build_network(radar_positions, t, rng=rng)
              for t in ("ER", "BA", "WS")}

    print("Validation on single mission:")
    ok, _ = validate_mission(radar_positions, graphs["ER"], verbose=True)
    if not ok:
        print("\n[WARNING] One or more validation checks failed.")
    else:
        print("\n[OK] All validation checks passed.")

    print("\nGenerating trajectory figure...")
    fig, sg, sl, surv = plot_trajectory_analysis(
        radar_positions = radar_positions,
        graphs          = graphs,
        seed            = config.RANDOM_SEED,
        n_ensemble      = config.N_TRAJECTORIES,
        save            = True,
    )

    plt.show()
