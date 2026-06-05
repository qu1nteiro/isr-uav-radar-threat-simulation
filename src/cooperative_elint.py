"""
cooperative_elint.py — Stage 5: cooperative ELINT-equipped UAV missions.

Two AR3 drones fly simultaneously toward the same target, sharing their
ELINT maps in real time. Drones start from different positions (UAV_START_A
and UAV_START_B) to maximise spatial coverage and genuine map sharing.

Both Gaussian and Lévy walk modes are compared for solo and cooperative.

Three analyses:
    1. Trajectory figure  : 30 pairs per walk type, coloured by joint outcome
    2. S(ρ) sweep         : solo vs coop, Gaussian vs Lévy, all topologies
    3. ELINT map coverage  : solo vs coop for both walk types at max ΔS point

Parallelism:
    - Solo sweep  : per-ρ parallel (module-level worker)
    - Coop sweep  : mission-level parallel (flat task list)

Usage
-----
    python src/cooperative_elint.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
import os
import time

import config
from radar_detection import global_detection
from network_builder import build_network, TOPO_COLOURS, TOPO_LABELS
from uav_elint_trajectory import (
    run_elint_ensemble, ELINTMap,
    _propagate_alerts, _decay_alerts,
)

# ─── Walk styling ─────────────────────────────────────────────────────────────

WALK_STYLE = {
    "gaussian": {"label": "Gaussian", "ls_solo": "--", "ls_coop": "-",
                 "lw_solo": 1.5, "lw_coop": 2.0, "marker": "o"},
    "levy":     {"label": f"Lévy ($\\mu={config.LEVY_ALPHA}$)",
                 "ls_solo": ":",  "ls_coop": "-.",
                 "lw_solo": 1.5, "lw_coop": 2.0, "marker": "s"},
}

OUTCOMES = {
    "both_success": {"label": "Both succeed", "ca": "#0F6E56", "cb": "#34d399"},
    "one_success":  {"label": "One succeeds", "ca": "#185FA5", "cb": "#60a5fa"},
    "both_failed":  {"label": "Both fail",    "ca": "#993C1D", "cb": "#f87171"},
}


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class CoopResult:
    """
    Complete record of one cooperative mission.

    coverage_a_pre : Drone A's ELINT map BEFORE merging with B (solo proxy).
    coverage_sh    : Union of both maps AFTER merging (cooperative metric).
    """
    outcome_a:      str
    outcome_b:      str
    joint_outcome:  str
    positions_a:    np.ndarray
    positions_b:    np.ndarray
    det_pos_a:      Optional[tuple]
    det_pos_b:      Optional[tuple]
    coverage_a_pre: np.ndarray
    coverage_sh:    np.ndarray
    n_steps:        int
    n_radars:       int


# ─── Module-level helpers (picklable by joblib) ───────────────────────────────

def _safe_step(pos, direction, s, known_positions):
    """Truncate step so drone never enters ELINT zone of a known radar."""
    s_safe = s
    for rp in known_positions:
        diff = pos - rp
        b    = 2.0 * np.dot(direction, diff)
        c    = np.dot(diff, diff) - config.ELINT_RANGE ** 2
        disc = b * b - 4.0 * c
        if disc < 0:
            continue
        t1 = (-b - np.sqrt(disc)) / 2.0
        if 0.0 < t1 < s_safe:
            s_safe = t1 * 0.98
    return s_safe


def _direction_vector(pos, known_positions, rng, beta):
    """Biased random walk with ELINT repulsion."""
    tx, ty = config.MISSION_TARGET
    diff_t = np.array([tx, ty]) - pos
    dist_t = np.linalg.norm(diff_t)
    e_tgt  = diff_t / dist_t if dist_t > 1e-9 else np.zeros(2)

    theta  = rng.uniform(0.0, 2.0 * np.pi)
    e_rand = np.array([np.cos(theta), np.sin(theta)])

    v = config.DRIFT_WEIGHT * e_tgt + (1.0 - config.DRIFT_WEIGHT) * e_rand
    for rp in known_positions:
        diff_r = rp - pos
        dist_r = np.linalg.norm(diff_r)
        if dist_r > 1e-9:
            weight = config.ELINT_RANGE / (dist_r + 1e-3)
            v     -= beta * weight * (diff_r / dist_r)

    v_norm = np.linalg.norm(v)
    return v / v_norm if v_norm > 1e-9 else e_rand


def _levy_step(rng):
    u = rng.uniform()
    return float(np.clip(
        config.LEVY_S_MIN * (1.0 - u) ** (-1.0 / config.LEVY_ALPHA),
        config.LEVY_S_MIN, config.LEVY_S_MAX,
    ))


def _place_radars(rho, rng):
    L, mg = config.GRID_SIZE, config.RADAR_RANGE * 0.5
    N     = max(1, round(rho * L ** 2))
    return rng.uniform(mg, L - mg, size=(N, 2)), N


# ─── Single cooperative mission ───────────────────────────────────────────────

def run_coop_mission(radar_positions, G, rng_a, rng_b,
                     walk="gaussian", beta=None):
    """
    Simulate one cooperative mission.

    Drone A starts at UAV_START_A, Drone B at UAV_START_B.
    Different starting positions ensure divergent trajectories and
    genuine ELINT map sharing benefit.

    Step order per timestep:
      1. Individual ELINT scans.
      2. Record Drone A coverage BEFORE merging (solo proxy).
      3. Merge maps (real-time sharing).
      4. Record shared coverage AFTER merging (cooperative metric).
      5. Move both drones using shared map.
      6. Check detection.
    """
    beta     = beta if beta is not None else config.ELINT_BETA
    N        = len(radar_positions)
    tgt      = np.array(config.MISSION_TARGET)
    use_levy = (walk == "levy")

    # Different starting positions — key for genuine map sharing
    pos_a = np.array(config.UAV_START_A, dtype=float)
    pos_b = np.array(config.UAV_START_B, dtype=float)

    elint_a    = ELINTMap(N)
    elint_b    = ELINTMap(N)
    alert_cnts = np.zeros(N, dtype=int)

    traj_a    = [pos_a.copy()]
    traj_b    = [pos_b.copy()]
    cov_a_pre = [0.0]
    cov_sh    = [0.0]

    out_a  = "timeout"; out_b  = "timeout"
    det_a  = None;      det_b  = None
    done_a = False;     done_b = False

    for t in range(config.MAX_STEPS):

        # ── Arrival ───────────────────────────────────────────────────────────
        if not done_a and np.linalg.norm(pos_a - tgt) <= config.TARGET_RADIUS:
            out_a = "success"; done_a = True
        if not done_b and np.linalg.norm(pos_b - tgt) <= config.TARGET_RADIUS:
            out_b = "success"; done_b = True
        if done_a and done_b:
            break

        # ── Individual ELINT scans ────────────────────────────────────────────
        if not done_a:
            elint_a.update(pos_a, radar_positions, t)
        if not done_b:
            elint_b.update(pos_b, radar_positions, t)

        # ── Record BEFORE sharing (solo proxy — Drone A only) ─────────────────
        cov_a_pre.append(elint_a.n_known / N)

        # ── Merge maps ────────────────────────────────────────────────────────
        shared = elint_a.known | elint_b.known
        elint_a.known[:] = shared
        elint_b.known[:] = shared

        # ── Record AFTER sharing (cooperative) ────────────────────────────────
        cov_sh.append(float(shared.sum()) / N)

        # ── Move ──────────────────────────────────────────────────────────────
        for done, pos, elint, rng, traj in [
            (done_a, pos_a, elint_a, rng_a, traj_a),
            (done_b, pos_b, elint_b, rng_b, traj_b),
        ]:
            if not done:
                kp  = elint.known_positions(radar_positions)
                d   = _direction_vector(pos, kp, rng, beta)
                s   = _safe_step(pos, d,
                                 _levy_step(rng) if use_levy else config.UAV_STEP_SIZE,
                                 kp)
                pos[:] = np.clip(pos + s * d, 0.0, config.GRID_SIZE)
                traj.append(pos.copy())

        # ── Detection ─────────────────────────────────────────────────────────
        if not done_a:
            det, ids, _ = global_detection(
                uav_pos=pos_a, radar_positions=radar_positions,
                alert_states=alert_cnts, rng=rng_a,
            )
            if det:
                out_a = "detected"; det_a = tuple(pos_a); done_a = True
                if len(ids) > 0:
                    _propagate_alerts(G, ids, alert_cnts)

        if not done_b:
            det, ids, _ = global_detection(
                uav_pos=pos_b, radar_positions=radar_positions,
                alert_states=alert_cnts, rng=rng_b,
            )
            if det:
                out_b = "detected"; det_b = tuple(pos_b); done_b = True
                if len(ids) > 0:
                    _propagate_alerts(G, ids, alert_cnts)

        _decay_alerts(alert_cnts)

        if done_a and done_b:
            break

    # Pad to equal length
    T = max(len(cov_a_pre), len(cov_sh))
    cov_a_pre += [cov_a_pre[-1]] * (T - len(cov_a_pre))
    cov_sh    += [cov_sh[-1]]    * (T - len(cov_sh))

    joint = ("both_success" if out_a == "success" and out_b == "success"
             else "one_success" if out_a == "success" or out_b == "success"
             else "both_failed")

    return CoopResult(
        outcome_a      = out_a,
        outcome_b      = out_b,
        joint_outcome  = joint,
        positions_a    = np.array(traj_a),
        positions_b    = np.array(traj_b),
        det_pos_a      = det_a,
        det_pos_b      = det_b,
        coverage_a_pre = np.array(cov_a_pre),
        coverage_sh    = np.array(cov_sh),
        n_steps        = max(len(traj_a), len(traj_b)) - 1,
        n_radars       = N,
    )


# ─── Module-level parallel workers ───────────────────────────────────────────

def _coop_worker(rho, topology, seed_a, seed_b, walk, beta, seed_base):
    topo_offset = {"ER": 3_000_000, "BA": 1_000_000, "WS": 2_000_000}
    rho_seed    = seed_base + hash(round(rho, 8)) % (2 ** 31) + topo_offset[topology]
    rng_sp      = np.random.default_rng(abs(rho_seed))

    radar_positions, N = _place_radars(rho, rng_sp)
    G = build_network(radar_positions, topology, rng=rng_sp)

    r = run_coop_mission(
        radar_positions, G,
        np.random.default_rng(seed_a),
        np.random.default_rng(seed_b),
        walk=walk, beta=beta,
    )
    return (N, r.joint_outcome, r.outcome_a, r.outcome_b,
            r.coverage_sh, r.coverage_a_pre)


def _solo_worker(rho, topology, n_runs, walk, beta, seed_base):
    topo_offset = {"ER": 3_000_000, "BA": 1_000_000, "WS": 2_000_000}
    rho_seed    = seed_base + hash(round(rho, 8)) % (2 ** 31) + topo_offset[topology]
    rng         = np.random.default_rng(abs(rho_seed))

    radar_positions, N = _place_radars(rho, rng)
    G = build_network(radar_positions, topology, rng=rng)

    stats, _ = run_elint_ensemble(
        radar_positions, G, rng, n=n_runs, walk=walk, beta=beta,
    )
    return N, stats.success_rate


# ─── Density sweeps ───────────────────────────────────────────────────────────

def sweep_solo(topology, rho_vals, n_runs=None,
               walk="gaussian", beta=None, seed_base=None):
    """Solo ELINT sweep — parallel at ρ-point level."""
    n_runs    = n_runs    if n_runs    is not None else config.N_PHASE_RUNS
    beta      = beta      if beta      is not None else config.ELINT_BETA
    seed_base = seed_base if seed_base is not None else config.RANDOM_SEED

    raw = Parallel(n_jobs=config.N_JOBS, verbose=0)(
        delayed(_solo_worker)(rho, topology, n_runs, walk, beta, seed_base)
        for rho in rho_vals
    )

    n_radars = np.array([r[0] for r in raw])
    s_solo   = np.array([r[1] for r in raw])
    ci_solo  = 1.96 * np.sqrt(
        np.maximum(s_solo * (1.0 - s_solo), 1e-9) / n_runs
    )

    for ri, rho in enumerate(rho_vals):
        print(f"    ρ={rho:.5f}  N={n_radars[ri]:>3d}  "
              f"solo={s_solo[ri]*100:>5.1f}%  CI±{ci_solo[ri]*100:.1f}%")

    return rho_vals, s_solo, ci_solo, n_radars


def sweep_cooperative(topology, rho_vals, n_runs=None,
                      walk="gaussian", beta=None, seed_base=None):
    """Cooperative ELINT sweep — parallel at mission level."""
    n_runs    = n_runs    if n_runs    is not None else config.N_PHASE_RUNS
    beta      = beta      if beta      is not None else config.ELINT_BETA
    seed_base = seed_base if seed_base is not None else config.RANDOM_SEED

    all_tasks = []
    for ri, rho in enumerate(rho_vals):
        for run in range(n_runs):
            sa = abs(seed_base + hash(round(rho, 8)) % (2**20) + run * 3 + 1)
            sb = abs(seed_base + hash(round(rho, 8)) % (2**20) + run * 3 + 2)
            all_tasks.append((ri, rho, sa, sb))

    print(f"    {len(all_tasks):,} cooperative missions  "
          f"({n_runs} × {len(rho_vals)} ρ)  "
          f"walk={walk}  cores={config.N_JOBS}")

    raw = Parallel(n_jobs=config.N_JOBS, verbose=0)(
        delayed(_coop_worker)(rho, topology, sa, sb, walk, beta, seed_base)
        for _, rho, sa, sb in all_tasks
    )

    by_rho = defaultdict(list)
    for (ri, *_), r in zip(all_tasks, raw):
        by_rho[ri].append(r)

    n_pts         = len(rho_vals)
    s_both        = np.zeros(n_pts)
    s_one         = np.zeros(n_pts)
    ci_both       = np.zeros(n_pts)
    ci_one        = np.zeros(n_pts)
    n_radars      = np.zeros(n_pts, dtype=int)
    cov_sh_mean   = np.zeros((n_pts, config.MAX_STEPS + 1))
    cov_solo_mean = np.zeros((n_pts, config.MAX_STEPS + 1))

    for ri, missions in by_rho.items():
        N_here       = missions[0][0]
        n_radars[ri] = N_here

        both = float(np.mean([m[1] == "both_success" for m in missions]))
        one  = float(np.mean([m[1] in ("both_success", "one_success")
                               for m in missions]))
        s_both[ri]  = both
        s_one[ri]   = one
        ci_both[ri] = 1.96 * np.sqrt(max(both * (1-both), 1e-9) / n_runs)
        ci_one[ri]  = 1.96 * np.sqrt(max(one  * (1-one),  1e-9) / n_runs)

        def _mean_cov(arrays):
            T   = config.MAX_STEPS + 1
            mat = np.zeros((len(arrays), T))
            for j, arr in enumerate(arrays):
                L = min(len(arr), T)
                mat[j, :L] = arr[:L]
                if L < T:
                    mat[j, L:] = arr[L-1]
            return mat.mean(axis=0)

        cov_sh_mean[ri]   = _mean_cov([m[4] for m in missions])
        cov_solo_mean[ri] = _mean_cov([m[5] for m in missions])

        print(f"    ρ={rho_vals[ri]:.5f}  N={N_here:>3d}  "
              f"both={both*100:>5.1f}%  "
              f"≥1={one*100:>5.1f}%  "
              f"CI±{ci_one[ri]*100:.1f}%")

    return (rho_vals, s_both, s_one, ci_both, ci_one,
            cov_sh_mean, cov_solo_mean, n_radars)


# ─── Trajectory samples ───────────────────────────────────────────────────────

def sample_trajectories(n_radars_target, topology, n_missions=30,
                        walk="gaussian", beta=None, seed=None):
    beta = beta if beta is not None else config.ELINT_BETA
    seed = seed if seed is not None else config.RANDOM_SEED

    rng_sp = np.random.default_rng(seed)
    L, mg  = config.GRID_SIZE, config.RADAR_RANGE * 0.5
    rp     = rng_sp.uniform(mg, L - mg, size=(n_radars_target, 2))
    G      = build_network(rp, topology, rng=rng_sp)

    results = []
    for i in range(n_missions):
        r = run_coop_mission(
            rp, G,
            np.random.default_rng(seed + i*2 + 1),
            np.random.default_rng(seed + i*2 + 2),
            walk=walk, beta=beta,
        )
        results.append(r)
    return results, rp


# ─── Heatmap ──────────────────────────────────────────────────────────────────

def _heatmap(radar_positions, resolution=160):
    L  = config.GRID_SIZE
    xs = np.linspace(0, L, resolution)
    ys = np.linspace(0, L, resolution)
    Xg, Yg    = np.meshgrid(xs, ys)
    P_survive = np.ones((resolution, resolution))
    for rp in radar_positions:
        D  = np.sqrt((Xg - rp[0])**2 + (Yg - rp[1])**2)
        Pi = np.where(D <= config.RADAR_RANGE,
                      np.exp(-D / config.DETECTION_LAMBDA), 0.0)
        P_survive *= (1.0 - Pi)
    return Xg, Yg, 1.0 - P_survive


# ─── Figure 1: Trajectories (Gaussian + Lévy side by side) ───────────────────

def plot_trajectories(missions_g, missions_l, radar_positions, topology,
                      n_radars, seed=None, save=True):
    """
    Four-panel figure: (Drone A / Drone B) × (Gaussian / Lévy).
    """
    Xg, Yg, P = _heatmap(radar_positions)
    cmap_d = LinearSegmentedColormap.from_list(
        "danger",
        [(0.0, "#0a0a0a00"), (0.2, "#7f1d1d22"),
         (0.6, "#991b1b70"), (1.0, "#dc2626bb")], N=256,
    )
    L = config.GRID_SIZE

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)

    walk_missions = [("Gaussian", missions_g), ("Lévy", missions_l)]

    for row, (walk_label, missions) in enumerate(walk_missions):
        counts = defaultdict(int)
        for m in missions:
            counts[m.joint_outcome] += 1
        n = len(missions)

        for col, (drone_label, start_pos) in enumerate(
            [("Drone A", config.UAV_START_A), ("Drone B", config.UAV_START_B)]
        ):
            ax = axes[row, col]
            ax.pcolormesh(Xg, Yg, P, cmap=cmap_d, vmin=0, vmax=1,
                          shading="gouraud", rasterized=True, zorder=1)

            for rp in radar_positions:
                ax.add_patch(plt.Circle(rp, config.RADAR_RANGE,
                                        fill=False, edgecolor="#dc262655",
                                        lw=0.6, linestyle="--", zorder=2))
                ax.add_patch(plt.Circle(rp, config.ELINT_RANGE,
                                        fill=False, edgecolor="#f9731630",
                                        lw=0.4, linestyle=":", zorder=2))

            ax.scatter(radar_positions[:, 0], radar_positions[:, 1],
                       s=22, c="#A32D2D", marker="^", zorder=4,
                       edgecolors="#fff", linewidths=0.4)

            for m in missions:
                jout = m.joint_outcome
                traj = m.positions_a if col == 0 else m.positions_b
                dpos = m.det_pos_a   if col == 0 else m.det_pos_b
                clr  = OUTCOMES[jout]["ca"] if col == 0 else OUTCOMES[jout]["cb"]
                ax.plot(traj[:, 0], traj[:, 1],
                        color=clr, lw=0.8, alpha=0.50, zorder=3)
                if dpos:
                    ax.scatter(*dpos, s=25, c=clr,
                               marker="x", linewidths=1.1, zorder=5)

            # Start markers (different for A and B)
            ax.scatter(*config.UAV_START_A, s=70, c="#185FA5", marker="o",
                       zorder=6, edgecolors="#fff", linewidths=1.2,
                       label="Start A")
            ax.scatter(*config.UAV_START_B, s=70, c="#9333ea", marker="D",
                       zorder=6, edgecolors="#fff", linewidths=1.2,
                       label="Start B")
            ax.add_patch(plt.Circle(config.MISSION_TARGET, config.TARGET_RADIUS,
                                    fill=True, facecolor="#3B6D1133",
                                    edgecolor="#3B6D11", lw=1.2, zorder=5))
            ax.scatter(*config.MISSION_TARGET, s=90, c="#3B6D11",
                       marker="*", zorder=6, edgecolors="#fff", linewidths=0.8)

            ax.set_xlim(0, L); ax.set_ylim(0, L)
            ax.set_aspect("equal")
            ax.set_xlabel("$x_1$ [u]", fontsize=9)
            ax.set_ylabel("$x_2$ [u]", fontsize=9)
            ax.set_title(
                f"{walk_label} — {drone_label} — {TOPO_LABELS[topology]}\n"
                f"$N={n_radars}$ radars  "
                f"both={counts['both_success']/n*100:.0f}%  "
                f"≥1={(counts['both_success']+counts['one_success'])/n*100:.0f}%",
                fontsize=8.5, fontweight="bold",
                color=TOPO_COLOURS[topology]
            )
            ax.grid(True, linewidth=0.25, alpha=0.3)

            if row == 0 and col == 0:
                legend_els = [
                    mpatches.Patch(fc=OUTCOMES["both_success"]["ca"], ec="none",
                                   label=f"Both succeed"),
                    mpatches.Patch(fc=OUTCOMES["one_success"]["ca"],  ec="none",
                                   label=f"One succeeds"),
                    mpatches.Patch(fc=OUTCOMES["both_failed"]["ca"],  ec="none",
                                   label=f"Both fail"),
                    plt.Line2D([0],[0], color="#f97316", lw=0.8, linestyle=":",
                               label=f"ELINT range ({config.ELINT_RANGE} u)"),
                    plt.Line2D([0],[0], color="#185FA5", marker="o", lw=0,
                               markersize=6, label="Start A (5,5)"),
                    plt.Line2D([0],[0], color="#9333ea", marker="D", lw=0,
                               markersize=6,
                               label=f"Start B (5,{int(config.UAV_START_B[1])})"),
                ]
                ax.legend(handles=legend_els, fontsize=7.5, loc="upper left",
                          framealpha=0.92, edgecolor="#cbd5e1")

    seed_val = seed if seed is not None else config.RANDOM_SEED
    fig.suptitle(
        f"Stage 5 — Cooperative ELINT trajectories   "
        f"($\\beta^*={config.ELINT_BETA}$, "
        f"$\\Delta$start={int(config.UAV_START_B[1]-config.UAV_START_A[1])} u, "
        f"seed {seed_val})",
        fontsize=11, fontweight="500",
    )

    if save:
        path = os.path.join(config.FIGURES_DIR,
                            "stage5_cooperative_trajectories.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[Stage 5] Trajectories saved → {path}")
    return fig


# ─── Figure 2: Cooperative analysis ──────────────────────────────────────────

def plot_cooperative_analysis(rho_vals, solo_results, coop_results,
                               seed=None, save=True):
    """
    Three-panel analysis figure — Gaussian and Lévy compared.

        [0] S(ρ) — ER network: solo G, solo L, coop G, coop L
        [1] ΔS cooperation benefit — Gaussian vs Lévy, all topologies
        [2] ELINT map coverage — solo vs coop for G and L at max ΔS point
    """
    L       = config.GRID_SIZE
    rho_now = config.N_RADARS_DEFAULT / L ** 2
    topos   = ["ER", "BA", "WS"]
    walks   = ["gaussian", "levy"]

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Topo colours reused; walk distinguished by linestyle
    # ── Panel 0: S(ρ) for ER — all 4 combinations ────────────────────────────
    ax0   = fig.add_subplot(gs[0])
    topo  = "ER"
    cc    = TOPO_COLOURS[topo]

    for walk in walks:
        ws  = WALK_STYLE[walk]
        sv  = solo_results[topo][walk]
        cv  = coop_results[topo][walk]
        rho = rho_vals * 1e3

        s_solo_sm = gaussian_filter1d(sv["s_solo"], sigma=1.5)
        s_one_sm  = gaussian_filter1d(cv["s_one"],  sigma=1.5)
        ci_solo   = sv["ci_solo"]
        ci_one    = cv["ci_one"]

        # CI bands
        ax0.fill_between(rho,
                         np.clip(s_solo_sm*100 - ci_solo*100, 0, 100),
                         np.clip(s_solo_sm*100 + ci_solo*100, 0, 100),
                         color=cc, alpha=0.07)
        ax0.fill_between(rho,
                         np.clip(s_one_sm*100  - ci_one*100,  0, 100),
                         np.clip(s_one_sm*100  + ci_one*100,  0, 100),
                         color=cc, alpha=0.07)

        ax0.plot(rho, s_solo_sm * 100,
                 color=cc, lw=ws["lw_solo"], linestyle=ws["ls_solo"],
                 alpha=0.70, label=f"Solo — {ws['label']}")
        ax0.plot(rho, s_one_sm * 100,
                 color=cc, lw=ws["lw_coop"], linestyle=ws["ls_coop"],
                 label=f"Coop (≥1) — {ws['label']}")

    ax0.axvline(rho_now*1e3, color="#dc2626", lw=1.0,
                linestyle="-.", alpha=0.8, label="Current ρ")
    ax0.set_xlabel("Radar density  $\\rho$  [$\\times 10^{-3}$ u$^{-2}$]",
                   fontsize=9)
    ax0.set_ylabel("Mission success rate  $S(\\rho)$  [%]", fontsize=9)
    ax0.set_title(f"Solo vs cooperative — {TOPO_LABELS[topo]}\n"
                  f"Gaussian vs Lévy walk",
                  fontsize=9, fontweight="bold", color=cc)
    ax0.set_ylim(-2, 102)
    ax0.set_xlim(rho_vals[0]*1e3, rho_vals[-1]*1e3)
    ax0.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax0.grid(True, linewidth=0.3, alpha=0.4)

    ax0t = ax0.twiny()
    n_ticks   = np.array([2, 5, 10, 15, 20, 30])
    rho_ticks = n_ticks / L**2
    valid     = (rho_ticks >= rho_vals[0]) & (rho_ticks <= rho_vals[-1])
    ax0t.set_xlim(ax0.get_xlim())
    ax0t.set_xticks(rho_ticks[valid]*1e3)
    ax0t.set_xticklabels([str(n) for n in n_ticks[valid]], fontsize=7)
    ax0t.set_xlabel("$N$ radars", fontsize=7.5, labelpad=3)

    # ── Panel 1: ΔS — Gaussian vs Lévy, all topologies ───────────────────────
    ax1 = fig.add_subplot(gs[1])

    # Two walk groups: solid=Gaussian, dashed=Lévy
    walk_ls = {"gaussian": "-", "levy": "--"}

    for topo in topos:
        cc = TOPO_COLOURS[topo]
        for walk in walks:
            sv  = solo_results[topo][walk]
            cv  = coop_results[topo][walk]
            rho = rho_vals * 1e3

            s_solo_sm = gaussian_filter1d(sv["s_solo"], sigma=1.5)
            s_one_sm  = gaussian_filter1d(cv["s_one"],  sigma=1.5)
            delta     = (s_one_sm - s_solo_sm) * 100
            ws_label  = WALK_STYLE[walk]["label"]

            ax1.plot(rho, delta, color=cc, lw=1.8,
                     linestyle=walk_ls[walk],
                     label=f"{TOPO_LABELS[topo]} — {ws_label}")
            ax1.fill_between(rho, 0.0, np.maximum(delta, 0),
                             color=cc, alpha=0.06)

    ax1.axhline(0, color="#94a3b8", lw=0.8, linestyle="--", alpha=0.7)
    ax1.axvline(rho_now*1e3, color="#dc2626", lw=1.0,
                linestyle="-.", alpha=0.8, label="Current ρ")
    ax1.set_xlabel("Radar density  $\\rho$  [$\\times 10^{-3}$ u$^{-2}$]",
                   fontsize=9)
    ax1.set_ylabel("$\\Delta S = S_{\\mathrm{coop}} - S_{\\mathrm{solo}}$  [pp]",
                   fontsize=9)
    ax1.set_title("Cooperation benefit  $\\Delta S(\\rho)$\n"
                  "solid = Gaussian  |  dashed = Lévy",
                  fontsize=9, fontweight="bold")
    ax1.set_xlim(rho_vals[0]*1e3, rho_vals[-1]*1e3)
    ax1.legend(fontsize=7, framealpha=0.9, edgecolor="#cbd5e1", ncols=2)
    ax1.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel 2: ELINT coverage — Gaussian vs Lévy, solo vs coop ─────────────
    ax2 = fig.add_subplot(gs[2])

    cov_colours = {"gaussian": TOPO_COLOURS["ER"],
                   "levy":     TOPO_COLOURS["WS"]}

    for walk in walks:
        cc    = cov_colours[walk]
        ws    = WALK_STYLE[walk]
        cv_er = coop_results["ER"][walk]
        sv_er = solo_results["ER"][walk]

        # Find max ΔS point for this walk
        s_solo_sm = gaussian_filter1d(sv_er["s_solo"], sigma=1.5)
        s_one_sm  = gaussian_filter1d(cv_er["s_one"],  sigma=1.5)
        delta_er  = s_one_sm - s_solo_sm
        idx_c     = int(np.argmax(delta_er))
        N_c       = int(cv_er["n_radars"][idx_c])
        delta_pp  = float(delta_er[idx_c] * 100)

        cov_sh_c   = cv_er["cov_sh_mean"][idx_c]
        cov_solo_c = cv_er["cov_solo_mean"][idx_c]
        steps      = np.arange(len(cov_sh_c))

        ax2.plot(steps, cov_sh_c * 100,
                 color=cc, lw=2.0, linestyle=ws["ls_coop"],
                 label=f"Coop — {ws['label']} (N={N_c}, Δ={delta_pp:.0f}pp)")
        ax2.plot(steps, cov_solo_c * 100,
                 color=cc, lw=1.5, linestyle=ws["ls_solo"], alpha=0.75,
                 label=f"Solo — {ws['label']}")

        # Annotate t_50%
        for cov, ls in [(cov_sh_c, ws["ls_coop"]),
                         (cov_solo_c, ws["ls_solo"])]:
            idxs = np.where(np.array(cov) >= 0.5)[0]
            if len(idxs) > 0:
                t50 = idxs[0]
                ax2.axvline(t50, color=cc, lw=0.7, linestyle=ls, alpha=0.4)

    ax2.axhline(100, color="#94a3b8", lw=0.6, linestyle=":", alpha=0.5)
    ax2.set_xlabel("Simulation step  $t$", fontsize=9)
    ax2.set_ylabel("Radars identified  [%]", fontsize=9)
    ax2.set_title("ELINT map coverage rate — ER network\n"
                  "at max $\\Delta S$ point for each walk type",
                  fontsize=9, fontweight="bold")
    ax2.set_xlim(0, min(800, config.MAX_STEPS))
    ax2.set_ylim(-2, 108)
    ax2.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax2.grid(True, linewidth=0.3, alpha=0.4)

    seed_val = seed if seed is not None else config.RANDOM_SEED
    fig.suptitle(
        f"Stage 5 — Cooperative ELINT analysis   "
        f"($\\beta^*={config.ELINT_BETA}$, "
        f"start sep. {int(config.UAV_START_B[1]-config.UAV_START_A[1])} u, "
        f"seed {seed_val})",
        fontsize=11, fontweight="500",
    )

    if save:
        path = os.path.join(config.FIGURES_DIR,
                            "stage5_cooperative_analysis.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[Stage 5] Analysis saved → {path}")
    return fig


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0   = time.time()
    seed = config.RANDOM_SEED

    rho_vals = np.linspace(0.0003, 0.002, 20)
    n_runs   = 500
    topos    = ["ER", "BA", "WS"]
    walks    = ["gaussian", "levy"]

    total = n_runs * len(rho_vals) * len(topos) * len(walks) * 2
    print("Running Stage 5 — Cooperative ELINT mission simulation")
    print(f"  ρ range   : {rho_vals[0]:.4f} – {rho_vals[-1]:.4f}  "
          f"({len(rho_vals)} points)")
    print(f"  Walks     : Gaussian + Lévy")
    print(f"  Start A   : {config.UAV_START_A}")
    print(f"  Start B   : {config.UAV_START_B}  "
          f"(sep={int(config.UAV_START_B[1]-config.UAV_START_A[1])} u)")
    print(f"  Total     : {total:,} missions\n")

    solo_results = {t: {} for t in topos}
    coop_results = {t: {} for t in topos}

    for topo in topos:
        print(f"\n{'='*60}")
        print(f"  TOPOLOGY: {TOPO_LABELS[topo]}")
        print(f"{'='*60}")

        for walk in walks:
            print(f"\n  [Solo ELINT — {TOPO_LABELS[topo]} — {walk}]")
            _, s_solo, ci_solo, n_r = sweep_solo(
                topo, rho_vals, n_runs=n_runs, walk=walk, seed_base=seed,
            )
            solo_results[topo][walk] = {
                "s_solo":   s_solo,
                "ci_solo":  ci_solo,
                "n_radars": n_r,
            }

            print(f"\n  [Cooperative ELINT — {TOPO_LABELS[topo]} — {walk}]")
            (_, s_both, s_one, ci_both, ci_one,
             cov_sh_mean, cov_solo_mean, n_r_c) = sweep_cooperative(
                topo, rho_vals, n_runs=n_runs, walk=walk, seed_base=seed,
            )
            coop_results[topo][walk] = {
                "s_both":        s_both,
                "s_one":         s_one,
                "ci_both":       ci_both,
                "ci_one":        ci_one,
                "cov_sh_mean":   cov_sh_mean,
                "cov_solo_mean": cov_solo_mean,
                "n_radars":      n_r_c,
            }

    print(f"\n  Sweeps completed in {(time.time()-t0)/60:.1f} minutes.")

    # ── Trajectory samples — both walk types ──────────────────────────────────
    print("\n  Generating trajectory samples (ER, N=13, Gaussian + Lévy)...")
    missions_g, rp = sample_trajectories(13, "ER", 30, "gaussian", seed=seed)
    missions_l, _  = sample_trajectories(13, "ER", 30, "levy",     seed=seed)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n  Generating figures...")
    plot_trajectories(missions_g, missions_l, rp, "ER",
                      n_radars=13, seed=seed, save=True)
    plot_cooperative_analysis(rho_vals, solo_results, coop_results,
                               seed=seed, save=True)

    plt.show()
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} minutes.")
