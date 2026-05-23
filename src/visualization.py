"""
visualize_model.py — Mission overview figure for the ISR UAV project.

Produces a single publication-quality figure showing:
  - Combined P(d) heatmap of the full radar network
  - N_TRAJ trajectory samples from start to target
  - Outcome colour-coding: success / detected / timeout
  - Detection markers and radar nodes with coverage rings

NOTE: This module uses a self-contained minimal random walk for
visualisation only. When Stage 3 (uav_trajectory.py) is complete,
replace _minimal_walk() with the proper imported function.

Usage
-----
    python src/visualize_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

import config
from radar_detection import detection_probability, detection_event


# ─── Colour palette (aligned with project aesthetic) ──────────────────────────

C_SUCCESS   = "#0F6E56"   # teal  — mission complete
C_DETECTED  = "#993C1D"   # coral — detected
C_TIMEOUT   = "#5F5E5A"   # gray  — timeout
C_RADAR     = "#A32D2D"   # red   — radar node
C_UAV_START = "#185FA5"   # blue  — start marker
C_TARGET    = "#3B6D11"   # green — target marker


# ─── Minimal random walk (Stage 1 visualisation only) ─────────────────────────

def _minimal_walk(rng):
    """
    Biased random walk from config.UAV_START toward config.MISSION_TARGET.
    Implements equations (1) and (2) of the formulation.

    Returns
    -------
    xs, ys : np.ndarray
        x and y coordinates of the trajectory.
    outcome : str
        'success' | 'detected' | 'timeout'
    detection_pos : tuple or None
        (x, y) at the step where detection occurred, or None.
    """
    x, y         = config.UAV_START
    tx, ty       = config.MISSION_TARGET
    s            = config.UAV_STEP_SIZE
    alpha        = config.DRIFT_WEIGHT

    xs = [x]
    ys = [y]
    outcome       = "timeout"
    detection_pos = None

    for _ in range(config.MAX_STEPS):
        # Unit vector toward target
        dx, dy = tx - x, ty - y
        dist_to_target = np.sqrt(dx**2 + dy**2)

        if dist_to_target <= config.TARGET_RADIUS:
            outcome = "success"
            break

        e_target = np.array([dx, dy]) / dist_to_target

        # Random unit vector
        theta   = rng.uniform(0, 2 * np.pi)
        e_rand  = np.array([np.cos(theta), np.sin(theta)])

        # Weighted combination → normalise → scale
        v = alpha * e_target + (1 - alpha) * e_rand
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            v = e_rand
        else:
            v = v / v_norm

        x += s * v[0]
        y += s * v[1]

        # Clip to domain
        x = np.clip(x, 0, config.GRID_SIZE)
        y = np.clip(y, 0, config.GRID_SIZE)

        xs.append(x)
        ys.append(y)

        # Detection check against all radars
        uav_pos = np.array([x, y])
        for rp in radar_positions:
            d = np.linalg.norm(uav_pos - rp)
            p = detection_probability(d, rng=rng)
            if detection_event(p, rng=rng):
                outcome       = "detected"
                detection_pos = (x, y)
                break

        if outcome == "detected":
            break

    return np.array(xs), np.array(ys), outcome, detection_pos


# ─── Heatmap ──────────────────────────────────────────────────────────────────

def _build_heatmap(radar_positions, resolution=300):
    """
    Compute the combined detection probability field over the domain.

    At each grid point, the effective P is:
        P_combined = 1 - prod_i (1 - P_i)   [probability of at least one detection]

    This gives the 'worst-case' field — the heatmap shows where the network
    as a whole is sensitive, not just individual radars.
    """
    L   = config.GRID_SIZE
    xs  = np.linspace(0, L, resolution)
    ys  = np.linspace(0, L, resolution)
    Xg, Yg = np.meshgrid(xs, ys)

    # Vectorised: distances from every grid point to every radar
    P_survive = np.ones((resolution, resolution))   # probability of NOT being detected

    for rp in radar_positions:
        D  = np.sqrt((Xg - rp[0])**2 + (Yg - rp[1])**2)
        Pi = np.where(D <= config.RADAR_RANGE,
                      np.exp(-D / config.DETECTION_LAMBDA),
                      0.0)
        P_survive *= (1.0 - Pi)

    P_combined = 1.0 - P_survive
    return Xg, Yg, P_combined


# ─── Main figure ──────────────────────────────────────────────────────────────

def plot_mission_overview(n_traj=40, n_radars=None, seed=None, save=True):
    """
    Generate the mission overview figure.

    Parameters
    ----------
    n_traj : int
        Number of trajectories to simulate and plot.
    n_radars : int, optional
        Number of radar nodes. Defaults to config.N_RADARS_DEFAULT.
    seed : int, optional
        Random seed. Defaults to config.RANDOM_SEED.
    save : bool
        If True, saves the figure to config.FIGURES_DIR.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    global radar_positions   # used by _minimal_walk

    n_radars = n_radars if n_radars is not None else config.N_RADARS_DEFAULT
    rng      = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
    L        = config.GRID_SIZE

    # ── Place radars randomly in the domain interior ───────────────────────────
    margin           = config.RADAR_RANGE * 0.5
    radar_positions  = rng.uniform(margin, L - margin, size=(n_radars, 2))

    # ── Heatmap ────────────────────────────────────────────────────────────────
    Xg, Yg, P_field = _build_heatmap(radar_positions, resolution=280)

    # ── Trajectories ──────────────────────────────────────────────────────────
    trajectories = []
    counts = {"success": 0, "detected": 0, "timeout": 0}

    for _ in range(n_traj):
        xs, ys, outcome, det_pos = _minimal_walk(rng)
        trajectories.append((xs, ys, outcome, det_pos))
        counts[outcome] += 1

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)

    # ── Heatmap layer ─────────────────────────────────────────────────────────
    cmap_danger = LinearSegmentedColormap.from_list(
        "danger",
        [(0.0, "#0a0a0a00"),     # transparent at P=0
         (0.1, "#7f1d1d22"),
         (0.5, "#991b1b88"),
         (1.0, "#dc2626cc")],    # opaque red at P=1
        N=256
    )

    im = ax.pcolormesh(Xg, Yg, P_field,
                       cmap=cmap_danger, vmin=0, vmax=1,
                       shading="gouraud", rasterized=True, zorder=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Combined detection probability  $P_{\\mathrm{net}}$",
                   fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    # ── Radar coverage rings ───────────────────────────────────────────────────
    for rp in radar_positions:
        ring = plt.Circle(rp, config.RADAR_RANGE,
                          fill=False, edgecolor="#dc2626",
                          linewidth=0.5, linestyle="--",
                          alpha=0.35, zorder=2)
        ax.add_patch(ring)

    # ── Radar nodes ───────────────────────────────────────────────────────────
    ax.scatter(radar_positions[:, 0], radar_positions[:, 1],
               s=40, c=C_RADAR, marker="^", zorder=4,
               linewidths=0.8, edgecolors="#fff",
               label=f"Radar node ($N={n_radars}$)")

    # ── Trajectories ──────────────────────────────────────────────────────────
    style = {
        "success":  dict(color=C_SUCCESS,  lw=0.9, alpha=0.75, zorder=3),
        "detected": dict(color=C_DETECTED, lw=0.9, alpha=0.65, zorder=3),
        "timeout":  dict(color=C_TIMEOUT,  lw=0.7, alpha=0.45, zorder=3),
    }

    for xs, ys, outcome, det_pos in trajectories:
        ax.plot(xs, ys, **style[outcome])

        if outcome == "detected" and det_pos is not None:
            ax.scatter(*det_pos, s=28, c=C_DETECTED,
                       marker="x", linewidths=1.2, zorder=5)

    # ── Start and target markers ───────────────────────────────────────────────
    sx, sy = config.UAV_START
    tx, ty = config.MISSION_TARGET

    ax.scatter(sx, sy, s=120, c=C_UAV_START, marker="o", zorder=6,
               edgecolors="#fff", linewidths=1.5)
    ax.annotate("AR3 start", xy=(sx, sy),
                xytext=(sx + 5, sy + 3),
                fontsize=8, color=C_UAV_START,
                arrowprops=dict(arrowstyle="-", color=C_UAV_START,
                                lw=0.7, shrinkA=4, shrinkB=0))

    target_ring = plt.Circle((tx, ty), config.TARGET_RADIUS,
                              fill=True, facecolor=C_TARGET + "33",
                              edgecolor=C_TARGET, linewidth=1.2, zorder=5)
    ax.add_patch(target_ring)
    ax.scatter(tx, ty, s=120, c=C_TARGET, marker="*", zorder=6,
               edgecolors="#fff", linewidths=1.0)
    ax.annotate("Mission target", xy=(tx, ty),
                xytext=(tx - 30, ty - 8),
                fontsize=8, color=C_TARGET,
                arrowprops=dict(arrowstyle="-", color=C_TARGET,
                                lw=0.7, shrinkA=6, shrinkB=0))

    # ── Legend (outcome lines) ─────────────────────────────────────────────────
    pct = lambda k: f"{counts[k]}/{n_traj} ({counts[k]/n_traj*100:.0f}%)"

    legend_elements = [
        mpatches.Patch(fc=C_SUCCESS,  ec="none", label=f"Success — {pct('success')}"),
        mpatches.Patch(fc=C_DETECTED, ec="none", label=f"Detected — {pct('detected')}"),
        mpatches.Patch(fc=C_TIMEOUT,  ec="none", label=f"Timeout — {pct('timeout')}"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=C_RADAR,
                   markersize=7, markeredgecolor="#fff", markeredgewidth=0.6,
                   label=f"Radar node ($N={n_radars}$)"),
        plt.Line2D([0], [0], marker="x", color=C_DETECTED,
                   markersize=6, linewidth=0,
                   label="Detection event"),
    ]

    ax.legend(handles=legend_elements, fontsize=8.5,
              loc="upper left", framealpha=0.92,
              edgecolor="#cbd5e1", facecolor="white")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$ [spatial units  ≈  km]", fontsize=10)
    ax.set_ylabel("$x_2$ [spatial units  ≈  km]", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.3, color="#888")

    # ── Title block ───────────────────────────────────────────────────────────
    ax.set_title(
        f"Mission overview — ISR UAV in contested radar environment\n"
        f"$N={n_radars}$ radars · $R={config.RADAR_RANGE}$ u · "
        f"$\\lambda={config.DETECTION_LAMBDA}$ · "
        f"$\\alpha={config.DRIFT_WEIGHT}$ · "
        f"{n_traj} trajectories",
        fontsize=10, pad=10
    )

    # ── Parameter annotation (bottom right) ───────────────────────────────────
    param_text = (
        f"$s = {config.UAV_STEP_SIZE}$ u/step\n"
        f"$T_{{\\max}} = {config.MAX_STEPS}$ steps\n"
        f"$r^* = {config.TARGET_RADIUS}$ u"
    )
    ax.text(0.985, 0.015, param_text,
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5,
            color="#64748b",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#cbd5e1", lw=0.6, alpha=0.85))

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        path = os.path.join(config.FIGURES_DIR, "mission_overview.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"[Visualize] Figure saved → {path}")

    return fig


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating mission overview figure...")
    plot_mission_overview(
        n_traj   = 40,
        n_radars = config.N_RADARS_DEFAULT,
        seed     = config.RANDOM_SEED,
        save     = True
    )
    plt.show()
