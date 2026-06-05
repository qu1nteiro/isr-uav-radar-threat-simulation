"""
uav_elint_trajectory.py — Stage 3b: UAV trajectory with ELINT-based evasion.

The AR3 is equipped with ELINT (Electronic Intelligence) — it intercepts
adversarial radar emissions and detects radar nodes before entering their
coverage zone. This allows the UAV to actively reroute around threats.

Physical model:
  - ELINT detects a radar at distance d <= ELINT_RANGE = 14 u
  - Radar detects the UAV at distance d <= RADAR_RANGE  = 10 u
  - Margin = 4 u → forces aggressive, late evasive manoeuvres
  - Known radar positions are repelled from the direction vector
  - Repulsion weight β is calibrated in this module

Comparison with uav_trajectory.py (blind walks):
  - Blind drone: 0% success, always detected near start
  - ELINT drone: non-zero success rate, visible evasive trajectories

Usage
-----
    python src/uav_elint_trajectory.py
    python src/uav_elint_trajectory.py --beta 0.8 --walk levy
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

import config
from radar_detection import detection_probability, detection_event, global_detection
from network_builder import build_network, place_radars, TOPO_COLOURS, TOPO_LABELS
from uav_trajectory import (MissionResult, EnsembleStats,
                             _levy_step, _gaussian_step,
                             _propagate_alerts, _decay_alerts,
                             run_ensemble)


# ─── ELINT state ──────────────────────────────────────────────────────────────

class ELINTMap:
    """
    Maintains the UAV's real-time knowledge of adversarial radar positions.

    At each step, any radar within ELINT_RANGE is added to the known set.
    The map grows monotonically — once detected, a radar stays known.
    This models the persistent intelligence picture built during the mission.
    """

    def __init__(self, n_radars):
        self.known     = np.zeros(n_radars, dtype=bool)
        self.n_radars  = n_radars
        self.discovery_steps = {}    # radar_id → step first detected

    def update(self, uav_pos, radar_positions, step):
        """Detect any radar within ELINT_RANGE and add to known set."""
        for i, rp in enumerate(radar_positions):
            if not self.known[i]:
                d = np.linalg.norm(uav_pos - rp)
                if d <= config.ELINT_RANGE:
                    self.known[i] = True
                    self.discovery_steps[i] = step

    def known_positions(self, radar_positions):
        """Return positions of all currently known radars."""
        if not np.any(self.known):
            return np.empty((0, 2))
        return radar_positions[self.known]

    @property
    def n_known(self):
        return int(self.known.sum())


# ─── ELINT direction vector ───────────────────────────────────────────────────

def _elint_direction(uav_pos, elint_map, radar_positions, rng, beta):
    """
    Compute movement direction with ELINT-based radar avoidance.

    Extends equation (1) with a repulsion term for each known radar:

        v(t) = α ê_target + (1-α) ê_rand - β Σ_{i∈K(t)} ê_i^rep

    where ê_i^rep points FROM the UAV TOWARD radar i (subtracted = repulsion).

    The repulsion is weighted by proximity: closer radars repel more strongly.
    Radars beyond ELINT_RANGE contribute zero (not yet known).

    Parameters
    ----------
    uav_pos : np.ndarray, shape (2,)
    elint_map : ELINTMap
    radar_positions : np.ndarray, shape (N, 2)
    rng : np.random.Generator
    beta : float
        Evasion weight. Higher β → stronger avoidance, less mission progress.

    Returns
    -------
    direction : np.ndarray, shape (2,), unit vector
    """
    tx, ty  = config.MISSION_TARGET
    target  = np.array([tx, ty])

    # Bias toward target
    diff_t   = target - uav_pos
    dist_t   = np.linalg.norm(diff_t)
    e_target = diff_t / dist_t if dist_t > 1e-9 else np.zeros(2)

    # Random component
    theta  = rng.uniform(0.0, 2.0 * np.pi)
    e_rand = np.array([np.cos(theta), np.sin(theta)])

    v = config.DRIFT_WEIGHT * e_target + (1.0 - config.DRIFT_WEIGHT) * e_rand

    # ELINT repulsion — proximity-weighted
    known_pos = elint_map.known_positions(radar_positions)
    for rp in known_pos:
        diff_r = rp - uav_pos           # vector FROM drone TOWARD radar
        dist_r = np.linalg.norm(diff_r)
        if dist_r > 1e-9:
            # Weight by 1/d: radars closer than ELINT_RANGE repel harder
            weight = config.ELINT_RANGE / (dist_r + 1e-3)
            e_rep  = diff_r / dist_r
            v     -= beta * weight * e_rep

    # Normalise
    v_norm = np.linalg.norm(v)
    return v / v_norm if v_norm > 1e-9 else e_rand


# ─── Single ELINT mission ─────────────────────────────────────────────────────

def run_elint_mission(radar_positions, G, rng, walk="gaussian", beta=None):
    """
    Simulate a single UAV mission with ELINT-based evasion.

    Parameters
    ----------
    radar_positions : np.ndarray, shape (N, 2)
    G : nx.Graph
        Radar communication network.
    rng : np.random.Generator
    walk : str
        'gaussian' (fixed step) or 'levy' (power-law steps).
    beta : float, optional
        Evasion weight. Defaults to config.ELINT_BETA.

    Returns
    -------
    MissionResult
        Compatible with Stage 3 blind walk results for direct comparison.
    elint_map : ELINTMap
        Final ELINT state — radar discovery log.
    """
    if beta is None:
        beta = config.ELINT_BETA

    N        = len(radar_positions)
    step_fn  = _gaussian_step if walk == "gaussian" else _levy_step
    topo     = G.nodes[0].get("topo", "unknown") if G.number_of_nodes() > 0 else "none"

    pos         = np.array(config.UAV_START, dtype=float)
    alert_cnts  = np.zeros(N, dtype=int)
    elint_map   = ELINTMap(N)

    positions  = [pos.copy()]
    p_max_hist = [0.0]
    alert_hist = [alert_cnts.copy()]
    step_sizes = []

    outcome        = "timeout"
    detection_step = None
    detection_pos  = None
    detecting_rad  = None

    for t in range(config.MAX_STEPS):

        # ── Mission success ────────────────────────────────────────────────────
        dist_to_target = np.linalg.norm(pos - np.array(config.MISSION_TARGET))
        if dist_to_target <= config.TARGET_RADIUS:
            outcome = "success"
            break

        # ── ELINT update — scan for radar emissions ────────────────────────────
        elint_map.update(pos, radar_positions, t)

        # ── Move with ELINT-informed direction ─────────────────────────────────
        s         = step_fn(rng)
        direction = _elint_direction(pos, elint_map, radar_positions, rng, beta)

        # ── Step truncation: prevent Lévy jumps from crossing into radar zones ──
        # For each known radar, compute the maximum safe step length along
        # the proposed direction such that ‖pos_final - radar_i‖ >= ELINT_RANGE.
        # This models the AR3 flight management system refusing to enter a
        # known threat zone regardless of the intended step size.
        s_safe = s
        known_pos = elint_map.known_positions(radar_positions)
        for rp in known_pos:
            # Solve: ‖pos + s*dir - rp‖ = ELINT_RANGE for s (quadratic)
            # a*s² + b*s + c = 0  where:
            diff = pos - rp
            a = np.dot(direction, direction)          # always 1 (unit vector)
            b = 2.0 * np.dot(direction, diff)
            c = np.dot(diff, diff) - config.ELINT_RANGE ** 2
            discriminant = b ** 2 - 4 * a * c
            if discriminant < 0:
                continue                              # ray doesn't intersect sphere
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            # t1 is entry into ELINT zone, t2 is exit
            # Only constrain if we are outside and moving inward (t1 > 0)
            if 0 < t1 < s_safe:
                s_safe = t1 * 0.98   # stop just outside the ELINT boundary

        pos = np.clip(pos + s_safe * direction, 0.0, config.GRID_SIZE)
        step_sizes.append(s_safe)

        # ── Detection check ────────────────────────────────────────────────────
        detected, det_ids, p_eff = global_detection(
            uav_pos         = pos,
            radar_positions = radar_positions,
            alert_states    = alert_cnts,
            rng             = rng,
        )
        p_max = float(p_eff.max()) if len(p_eff) > 0 else 0.0

        # ── Alert propagation ──────────────────────────────────────────────────
        if len(det_ids) > 0:
            _propagate_alerts(G, det_ids, alert_cnts)

        _decay_alerts(alert_cnts)

        # ── Record ────────────────────────────────────────────────────────────
        positions.append(pos.copy())
        p_max_hist.append(p_max)
        alert_hist.append(alert_cnts.copy())

        if detected:
            outcome        = "detected"
            detection_step = t + 1
            detection_pos  = tuple(pos)
            detecting_rad  = int(det_ids[0])
            break

    return MissionResult(
        outcome         = outcome,
        n_steps         = len(positions) - 1,
        positions       = np.array(positions),
        p_max_history   = np.array(p_max_hist),
        detection_step  = detection_step,
        detection_pos   = detection_pos,
        detecting_radar = detecting_rad,
        alert_history   = np.array(alert_hist),
        walk_type       = f"elint_{walk}",
        topology        = topo,
        step_sizes      = np.array(step_sizes),
    ), elint_map


# ─── ELINT ensemble ───────────────────────────────────────────────────────────

def run_elint_ensemble(radar_positions, G, rng,
                       n=None, walk="gaussian", beta=None):
    """
    Run N independent ELINT missions and aggregate statistics.

    Returns
    -------
    EnsembleStats, list of MissionResult
    """
    n    = n    or config.N_TRAJECTORIES
    beta = beta if beta is not None else config.ELINT_BETA
    topo = G.nodes[0].get("topo", "unknown") if G.number_of_nodes() > 0 else "none"

    outcomes     = []
    survival_arr = np.zeros(config.MAX_STEPS + 1)
    all_steps    = []
    det_steps    = []
    results      = []

    for _ in range(n):
        r, _ = run_elint_mission(radar_positions, G, rng,
                                  walk=walk, beta=beta)
        results.append(r)
        outcomes.append(r.outcome)
        all_steps.extend(r.step_sizes.tolist())

        survive_until = r.detection_step if r.outcome == "detected" else r.n_steps
        survival_arr[:survive_until + 1] += 1

        if r.outcome == "detected" and r.detection_step is not None:
            det_steps.append(r.detection_step)

    counts = {
        "success":  outcomes.count("success"),
        "detected": outcomes.count("detected"),
        "timeout":  outcomes.count("timeout"),
    }

    return EnsembleStats(
        n_missions          = n,
        success_rate        = counts["success"]  / n,
        detection_rate      = counts["detected"] / n,
        timeout_rate        = counts["timeout"]  / n,
        mean_survival_steps = float(np.mean(det_steps)) if det_steps else float(config.MAX_STEPS),
        survival_curve      = survival_arr / n,
        outcome_counts      = counts,
        walk_type           = f"elint_{walk}",
        topology            = topo,
        all_step_sizes      = np.array(all_steps),
    ), results


# ─── Beta calibration ─────────────────────────────────────────────────────────

def calibrate_beta(radar_positions, G, rng,
                   beta_values=None, n_per_beta=300, walk="gaussian"):
    """
    Sweep β and measure success rate to find the optimal evasion weight.

    Too low β → insufficient evasion → detected
    Too high β → drone circles away from target → timeout
    Optimal β → maximum success rate

    Parameters
    ----------
    beta_values : array-like, optional
        β values to test. Default: 0.1 to 3.0 in 20 steps.
    n_per_beta : int
        Missions per β value.
    walk : str

    Returns
    -------
    beta_values : np.ndarray
    success_rates : np.ndarray
    detection_rates : np.ndarray
    timeout_rates : np.ndarray
    """
    if beta_values is None:
        beta_values = np.linspace(0.1, 3.0, 20)

    success_rates  = []
    detection_rates = []
    timeout_rates  = []

    print(f"\n  Calibrating β ({len(beta_values)} values, "
          f"{n_per_beta} missions each)...")

    for beta in beta_values:
        st, _ = run_elint_ensemble(radar_positions, G, rng,
                                    n=n_per_beta, walk=walk, beta=beta)
        success_rates.append(st.success_rate)
        detection_rates.append(st.detection_rate)
        timeout_rates.append(st.timeout_rate)
        print(f"    β={beta:.2f}  S={st.success_rate*100:>5.1f}%  "
              f"D={st.detection_rate*100:>5.1f}%  "
              f"T={st.timeout_rate*100:>5.1f}%")

    return (np.array(beta_values),
            np.array(success_rates),
            np.array(detection_rates),
            np.array(timeout_rates))


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


def _draw_elint_panel(ax, results, radar_positions, Xg, Yg, P_field,
                      title, n_sample=50):
    """Draw ELINT trajectory panel — identical interface to Stage 3 version."""
    cmap_danger = LinearSegmentedColormap.from_list(
        "danger",
        [(0.0, "#0a0a0a00"), (0.1, "#7f1d1d18"),
         (0.5, "#991b1b70"), (1.0, "#dc2626bb")], N=256
    )
    ax.pcolormesh(Xg, Yg, P_field,
                  cmap=cmap_danger, vmin=0, vmax=1,
                  shading="gouraud", rasterized=True, zorder=1)

    # Radar coverage zones (R) and ELINT detection zones (ELINT_RANGE)
    for rp in radar_positions:
        ax.add_patch(plt.Circle(rp, config.RADAR_RANGE,
                                fill=False, edgecolor="#dc262660",
                                linewidth=0.6, linestyle="--", zorder=2))
        ax.add_patch(plt.Circle(rp, config.ELINT_RANGE,
                                fill=False, edgecolor="#f9731640",
                                linewidth=0.5, linestyle=":", zorder=2))

    ax.scatter(radar_positions[:, 0], radar_positions[:, 1],
               s=28, c="#A32D2D", marker="^", zorder=4,
               edgecolors="#fff", linewidths=0.5)

    colours = {"success": "#0F6E56", "detected": "#993C1D", "timeout": "#5F5E5A"}
    for r in results[:n_sample]:
        col = colours[r.outcome]
        ax.plot(r.positions[:, 0], r.positions[:, 1],
                color=col, lw=0.8, alpha=0.6, zorder=3)
        if r.outcome == "detected" and r.detection_pos:
            ax.scatter(*r.detection_pos, s=22, c=col,
                       marker="x", linewidths=1.1, zorder=5)

    ax.scatter(*config.UAV_START, s=70, c="#185FA5", marker="o",
               zorder=6, edgecolors="#fff", linewidths=1.2)
    ax.add_patch(plt.Circle(config.MISSION_TARGET, config.TARGET_RADIUS,
                            fill=True, facecolor="#3B6D1133",
                            edgecolor="#3B6D11", linewidth=1.2, zorder=5))
    ax.scatter(*config.MISSION_TARGET, s=90, c="#3B6D11",
               marker="*", zorder=6, edgecolors="#fff", linewidths=0.8)

    # ELINT range legend entry
    elint_line = plt.Line2D([0], [0], color="#f97316", lw=0.8,
                             linestyle=":", label=f"ELINT range  ($R_{{\\mathrm{{ELINT}}}}={config.ELINT_RANGE}$ u)")
    radar_line = plt.Line2D([0], [0], color="#dc2626", lw=0.8,
                             linestyle="--", label=f"Radar range  ($R={config.RADAR_RANGE}$ u)")

    n  = len(results)
    cs = {k: sum(1 for r in results if r.outcome == k) for k in colours}
    legend_els = [
        mpatches.Patch(fc=colours["success"],  ec="none",
                       label=f"Success   {cs['success']}/{n} ({cs['success']/n*100:.0f}%)"),
        mpatches.Patch(fc=colours["detected"], ec="none",
                       label=f"Detected  {cs['detected']}/{n} ({cs['detected']/n*100:.0f}%)"),
        mpatches.Patch(fc=colours["timeout"],  ec="none",
                       label=f"Timeout   {cs['timeout']}/{n} ({cs['timeout']/n*100:.0f}%)"),
        radar_line, elint_line,
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc="upper left",
              framealpha=0.92, edgecolor="#cbd5e1")
    ax.set_xlim(0, config.GRID_SIZE); ax.set_ylim(0, config.GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$ [u]", fontsize=9)
    ax.set_ylabel("$x_2$ [u]", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(True, linewidth=0.25, alpha=0.35)


def plot_elint_analysis(radar_positions, graphs, beta=None,
                        n_ensemble=None, seed=None, save=True):
    """
    Produce the ELINT figure — 4 panels:

        [0,0] β calibration curve
        [0,1] Sample trajectories — ELINT Gaussian walk
        [1,0] Sample trajectories — ELINT Lévy flight
        [1,1] Survival curves: blind vs ELINT, by topology
    """
    beta       = beta       if beta       is not None else config.ELINT_BETA
    n_ensemble = n_ensemble if n_ensemble is not None else config.N_TRAJECTORIES
    seed       = seed       if seed       is not None else config.RANDOM_SEED
    rng        = np.random.default_rng(seed)

    Xg, Yg, P_field = _heatmap(radar_positions)
    G_er = graphs["ER"]

    # ── β calibration ─────────────────────────────────────────────────────────
    beta_vals, s_rates, d_rates, t_rates = calibrate_beta(
        radar_positions, G_er, rng,
        beta_values=np.linspace(0.1, 3.0, 20),
        n_per_beta=200, walk="gaussian"
    )
    best_beta = beta_vals[np.argmax(s_rates)]
    print(f"\n  Optimal β = {best_beta:.2f}  "
          f"(S={max(s_rates)*100:.1f}%)")

    # Use optimal beta for all subsequent runs
    beta = best_beta
    config.ELINT_BETA = beta   # update in-memory for this session

    # ── Ensembles ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)   # reset for reproducibility

    stats_eg, res_eg = run_elint_ensemble(
        radar_positions, G_er, rng, n=n_ensemble, walk="gaussian", beta=beta)
    stats_el, res_el = run_elint_ensemble(
        radar_positions, G_er, rng, n=n_ensemble, walk="levy", beta=beta)

    # Survival curves across topologies (ELINT Gaussian)
    surv_elint = {}
    for topo, G in graphs.items():
        st, _ = run_elint_ensemble(
            radar_positions, G, rng, n=n_ensemble,
            walk="gaussian", beta=beta)
        surv_elint[topo] = st

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # ── Panel [0,0]: β calibration ────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(beta_vals, s_rates * 100, color="#0F6E56",
              lw=1.8, label="Success")
    ax00.plot(beta_vals, d_rates * 100, color="#993C1D",
              lw=1.8, label="Detected")
    ax00.plot(beta_vals, t_rates * 100, color="#5F5E5A",
              lw=1.8, label="Timeout", linestyle="--")
    ax00.axvline(best_beta, color="#185FA5", lw=1.2,
                 linestyle=":", label=f"Optimal  $\\beta^*={best_beta:.2f}$")
    ax00.set_xlabel("Evasion weight  $\\beta$", fontsize=9)
    ax00.set_ylabel("Mission rate  [%]", fontsize=9)
    ax00.set_title("ELINT evasion weight calibration\n"
                   "(Gaussian walk, ER network)",
                   fontsize=9, fontweight="bold")
    ax00.set_xlim(beta_vals[0], beta_vals[-1])
    ax00.set_ylim(-2, 102)
    ax00.legend(fontsize=8.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax00.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel [0,1]: ELINT Gaussian trajectories ──────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    _draw_elint_panel(ax01, res_eg, radar_positions, Xg, Yg, P_field,
                      f"ELINT — Gaussian walk  ($\\beta^*={best_beta:.2f}$)\n"
                      f"$R_{{\\mathrm{{ELINT}}}}={config.ELINT_RANGE}$ u  "
                      f"margin $= {config.ELINT_RANGE - config.RADAR_RANGE:.0f}$ u")

    # ── Panel [1,0]: ELINT Lévy trajectories ──────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    _draw_elint_panel(ax10, res_el, radar_positions, Xg, Yg, P_field,
                      f"ELINT — Lévy flight  ($\\mu={config.LEVY_ALPHA}$, "
                      f"$\\beta^*={best_beta:.2f}$)\n"
                      f"$s_{{\\min}}={config.LEVY_S_MIN}$  "
                      f"$s_{{\\max}}={config.LEVY_S_MAX}$")

    # ── Panel [1,1]: Survival curves comparison ────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    steps = np.arange(config.MAX_STEPS + 1)

    for topo, st in surv_elint.items():
        ax11.plot(steps, st.survival_curve * 100,
                  color=TOPO_COLOURS[topo], lw=1.8,
                  label=f"{TOPO_LABELS[topo]}  (S={st.success_rate*100:.1f}%)")

    ax11.set_xlabel("Simulation step  $t$", fontsize=9)
    ax11.set_ylabel("Fraction undetected  [%]", fontsize=9)
    ax11.set_title("Survival curves — ELINT Gaussian walk\n"
                   "by radar network topology",
                   fontsize=9, fontweight="bold")
    ax11.set_xlim(0, config.MAX_STEPS)
    ax11.set_ylim(0, 105)
    ax11.legend(fontsize=8.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax11.grid(True, linewidth=0.3, alpha=0.4)
    ax11.text(config.MAX_STEPS - 5, 102, "$T_{\\max}$",
              ha="right", fontsize=8, color="#64748b")

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Stage 3b — ELINT-equipped UAV trajectory   "
        f"($N_{{\\mathrm{{radars}}}}={len(radar_positions)}$, "
        f"$N_{{\\mathrm{{missions}}}}={n_ensemble}$, seed {seed})",
        fontsize=11, fontweight="500"
    )

    if save:
        path = os.path.join(config.FIGURES_DIR, "stage3b_elint_trajectories.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[Stage 3b] Figure saved → {path}")

    return fig, stats_eg, stats_el, surv_elint, best_beta


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ELINT-equipped AR3 UAV mission simulation."
    )
    parser.add_argument("--beta",  type=float, default=None,
                        help="Fixed β (skips calibration if set)")
    parser.add_argument("--walk",  type=str, default="gaussian",
                        choices=["gaussian", "levy"])
    parser.add_argument("--seed",  type=int, default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else config.RANDOM_SEED
    rng  = np.random.default_rng(seed)

    print("Running Stage 3b — ELINT UAV trajectory simulation\n")

    radar_positions = place_radars(rng=rng)
    graphs = {t: build_network(radar_positions, t, rng=rng)
              for t in ("ER", "BA", "WS")}

    fig, *_ = plot_elint_analysis(
        radar_positions = radar_positions,
        graphs          = graphs,
        beta            = args.beta,
        n_ensemble      = config.N_TRAJECTORIES,
        seed            = seed,
        save            = True,
    )

    plt.show()