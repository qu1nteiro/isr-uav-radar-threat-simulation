"""
animate_mission.py — Step-by-step animation of a single UAV mission.

Renders the AR3 moving through the contested radar environment frame by frame.
Saves the result as mission_animation.gif in results/figures/.

Usage
-----
    python src/animate_mission.py              # default seed
    python src/animate_mission.py --seed 7     # specific seed
    python src/animate_mission.py --radars 8   # number of radars
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os
import sys

import config
from radar_detection import detection_probability, detection_event


# ─── Colours ──────────────────────────────────────────────────────────────────

C_SUCCESS  = "#0F6E56"
C_DETECTED = "#993C1D"
C_TIMEOUT  = "#5F5E5A"
C_RADAR    = "#A32D2D"
C_UAV      = "#185FA5"
C_TARGET   = "#3B6D11"
C_TRAIL    = "#2563eb"
C_ALERT    = "#dc2626"

TRAIL_LEN  = 40      # number of past positions to show as trail


# ─── Pre-compute full trajectory ──────────────────────────────────────────────

def run_mission(radar_positions, rng):
    """
    Simulate a full mission and return every step.

    Returns
    -------
    steps : list of dict
        One dict per step with keys:
            x, y         — UAV position
            p_max        — highest P_eff at this step
            detected     — bool, was the UAV detected this step?
            det_radar_id — index of detecting radar, or None
    outcome : str  — 'success' | 'detected' | 'timeout'
    """
    x, y   = config.UAV_START
    tx, ty = config.MISSION_TARGET
    s      = config.UAV_STEP_SIZE
    alpha  = config.DRIFT_WEIGHT

    steps   = [{"x": x, "y": y, "p_max": 0.0,
                 "detected": False, "det_radar_id": None}]
    outcome = "timeout"

    for _ in range(config.MAX_STEPS):
        dx, dy   = tx - x, ty - y
        dist_tgt = np.sqrt(dx**2 + dy**2)

        if dist_tgt <= config.TARGET_RADIUS:
            outcome = "success"
            break

        e_tgt  = np.array([dx, dy]) / dist_tgt
        theta  = rng.uniform(0, 2 * np.pi)
        e_rand = np.array([np.cos(theta), np.sin(theta)])

        v     = alpha * e_tgt + (1 - alpha) * e_rand
        v_n   = np.linalg.norm(v)
        v     = v / v_n if v_n > 1e-9 else e_rand

        x = float(np.clip(x + s * v[0], 0, config.GRID_SIZE))
        y = float(np.clip(y + s * v[1], 0, config.GRID_SIZE))

        uav_pos = np.array([x, y])
        p_max   = 0.0
        step_detected   = False
        det_radar_id    = None

        for i, rp in enumerate(radar_positions):
            d = float(np.linalg.norm(uav_pos - rp))
            p = float(detection_probability(d, rng=rng))
            p_max = max(p_max, p)
            if not step_detected and detection_event(p, rng=rng):
                step_detected = True
                det_radar_id  = i

        steps.append({"x": x, "y": y, "p_max": p_max,
                      "detected": step_detected,
                      "det_radar_id": det_radar_id})

        if step_detected:
            outcome = "detected"
            break

    return steps, outcome


# ─── Heatmap ──────────────────────────────────────────────────────────────────

def build_heatmap(radar_positions, resolution=200):
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


# ─── Animation ────────────────────────────────────────────────────────────────

def animate(n_radars=8, seed=None, save=True, show=True, interval=30):
    """
    Build and optionally save the mission animation.

    Parameters
    ----------
    n_radars : int
        Number of radar nodes.
    seed : int or None
    save : bool
        Save as .gif to results/figures/.
    show : bool
        Call plt.show() after building.
    interval : int
        Milliseconds between frames.
    """
    seed = seed if seed is not None else config.RANDOM_SEED
    rng  = np.random.default_rng(seed)
    L    = config.GRID_SIZE

    # ── Place radars ──────────────────────────────────────────────────────────
    margin          = config.RADAR_RANGE * 0.6
    radar_positions = rng.uniform(margin, L - margin, size=(n_radars, 2))

    # ── Simulate ──────────────────────────────────────────────────────────────
    steps, outcome = run_mission(radar_positions, rng)
    n_frames       = len(steps)

    print(f"  Seed {seed} | {n_radars} radars | {n_frames} steps | outcome: {outcome}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    Xg, Yg, P_field = build_heatmap(radar_positions)

    cmap_danger = LinearSegmentedColormap.from_list(
        "danger",
        [(0.0, "#0a0a0a00"),
         (0.1, "#7f1d1d22"),
         (0.5, "#991b1b88"),
         (1.0, "#dc2626cc")],
        N=256
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 8.6), constrained_layout=False)
    fig.patch.set_facecolor("#0d1117")

    # Main axes (map)
    ax = fig.add_axes([0.07, 0.13, 0.82, 0.82])
    ax.set_facecolor("#0d1117")

    # ── Static layers ─────────────────────────────────────────────────────────
    ax.pcolormesh(Xg, Yg, P_field,
                  cmap=cmap_danger, vmin=0, vmax=1,
                  shading="gouraud", rasterized=True, zorder=1)

    for rp in radar_positions:
        ring = plt.Circle(rp, config.RADAR_RANGE,
                          fill=False, edgecolor="#dc262688",
                          linewidth=0.8, linestyle="--", zorder=2)
        ax.add_patch(ring)

    ax.scatter(radar_positions[:, 0], radar_positions[:, 1],
               s=60, c=C_RADAR, marker="^", zorder=4,
               linewidths=0.8, edgecolors="#ffffff88")

    # Target
    target_ring = plt.Circle(config.MISSION_TARGET, config.TARGET_RADIUS,
                              fill=True, facecolor=C_TARGET + "44",
                              edgecolor=C_TARGET, linewidth=1.5, zorder=3)
    ax.add_patch(target_ring)
    ax.scatter(*config.MISSION_TARGET, s=120, c=C_TARGET,
               marker="*", zorder=5, edgecolors="#fff", linewidths=0.8)

    # Start
    ax.scatter(*config.UAV_START, s=80, c=C_UAV,
               marker="o", zorder=5, edgecolors="#fff", linewidths=1.0,
               alpha=0.6)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$  [spatial units  ≈  km]",
                  fontsize=9, color="#94a3b8")
    ax.set_ylabel("$x_2$  [spatial units  ≈  km]",
                  fontsize=9, color="#94a3b8")
    ax.tick_params(colors="#64748b", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")
    ax.grid(True, linewidth=0.25, alpha=0.25, color="#94a3b8")

    # ── Dynamic elements (initialised empty) ──────────────────────────────────
    trail_line, = ax.plot([], [], lw=1.2, color=C_TRAIL,
                          alpha=0.6, zorder=6, solid_capstyle="round")

    uav_dot, = ax.plot([], [], "o", ms=8, color=C_UAV,
                       mec="#fff", mew=1.2, zorder=8)

    detection_flash = plt.Circle((0, 0), 0, fill=True,
                                 facecolor="#dc262633",
                                 edgecolor=C_ALERT,
                                 linewidth=1.5, zorder=7, visible=False)
    ax.add_patch(detection_flash)

    det_markers, = ax.plot([], [], "x", ms=7, color=C_ALERT,
                           mew=1.5, zorder=7, ls="none")
    det_xs, det_ys = [], []

    # ── Status bar (bottom) ───────────────────────────────────────────────────
    bar_ax = fig.add_axes([0.07, 0.03, 0.82, 0.072])
    bar_ax.set_facecolor("#161b22")
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")

    txt_step    = bar_ax.text(0.02, 0.62, "", fontsize=8.5,
                              color="#94a3b8", va="center",
                              fontfamily="monospace")
    txt_pmax    = bar_ax.text(0.02, 0.22, "", fontsize=8.5,
                              color="#94a3b8", va="center",
                              fontfamily="monospace")
    txt_outcome = bar_ax.text(0.5, 0.5, "", fontsize=11,
                              color="#94a3b8", va="center", ha="center",
                              fontweight="bold")

    # P bar background
    bar_ax.add_patch(mpatches.FancyBboxPatch(
        (0.38, 0.15), 0.58, 0.35,
        boxstyle="round,pad=0.01",
        facecolor="#1e293b", edgecolor="#334155", lw=0.5))
    p_bar_bg = bar_ax.add_patch(mpatches.FancyBboxPatch(
        (0.385, 0.18), 0.0, 0.28,
        boxstyle="square,pad=0",
        facecolor=C_TRAIL, edgecolor="none"))
    txt_p_label = bar_ax.text(0.385, 0.62, "P_max", fontsize=7,
                              color="#64748b", va="center")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             f"AR3 ISR Mission Simulation  —  "
             f"$N={n_radars}$ radars · seed {seed}",
             ha="center", va="top", fontsize=10,
             color="#e2e8f0", fontweight="500")

    outcome_colour = {
        "success":  C_SUCCESS,
        "detected": C_DETECTED,
        "timeout":  C_TIMEOUT,
    }

    flash_radius_max = 12.0
    flash_duration   = 12    # frames for the flash effect

    def init():
        trail_line.set_data([], [])
        uav_dot.set_data([], [])
        det_markers.set_data([], [])
        detection_flash.set_visible(False)
        txt_step.set_text("")
        txt_pmax.set_text("")
        txt_outcome.set_text("")
        p_bar_bg.set_width(0.0)
        return (trail_line, uav_dot, detection_flash,
                det_markers, txt_step, txt_pmax, txt_outcome, p_bar_bg)

    def update(frame):
        # Hold on last frame
        f = min(frame, n_frames - 1)
        step = steps[f]

        x, y   = step["x"], step["y"]
        p_max  = step["p_max"]

        # Trail
        t_start = max(0, f - TRAIL_LEN)
        txs     = [s["x"] for s in steps[t_start:f+1]]
        tys     = [s["y"] for s in steps[t_start:f+1]]
        trail_line.set_data(txs, tys)

        # UAV dot colour — changes on detection
        if step["detected"]:
            uav_dot.set_color(C_ALERT)
            det_xs.append(x)
            det_ys.append(y)
            det_markers.set_data(det_xs, det_ys)
        else:
            uav_dot.set_color(C_UAV)

        uav_dot.set_data([x], [y])

        # Detection flash — expanding ring
        frames_since = f - (n_frames - 1) if outcome == "detected" else -1
        if step["detected"]:
            detection_flash.center = (x, y)
            detection_flash.set_radius(flash_radius_max * 0.3)
            detection_flash.set_visible(True)
        elif f > 0 and steps[f-1]["detected"]:
            detection_flash.set_visible(False)
        else:
            detection_flash.set_visible(False)

        # Status bar text
        elapsed_min = f * config.STEP_DURATION_S / 60
        txt_step.set_text(
            f"Step {f:>4d} / {config.MAX_STEPS}   "
            f"│   t = {elapsed_min:>5.1f} min   "
            f"│   pos = ({x:>6.1f}, {y:>5.1f})"
        )
        txt_pmax.set_text(f"Max P(d) this step: {p_max*100:>5.1f}%")

        # P bar
        bar_w = 0.565 * min(p_max, 1.0)
        p_bar_bg.set_width(bar_w)
        p_bar_bg.set_facecolor(
            C_ALERT if p_max > 0.6 else
            "#f97316" if p_max > 0.3 else
            C_TRAIL
        )

        # Outcome message (only at the end)
        if f == n_frames - 1:
            msgs = {
                "success":  "✓  Mission complete",
                "detected": "✗  UAV detected",
                "timeout":  "⚠  Mission timeout",
            }
            txt_outcome.set_text(msgs[outcome])
            txt_outcome.set_color(outcome_colour[outcome])
        else:
            txt_outcome.set_text("")

        return (trail_line, uav_dot, detection_flash,
                det_markers, txt_step, txt_pmax, txt_outcome, p_bar_bg)

    # Hold last frame for 60 extra frames
    total_frames = n_frames + 60

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        init_func=init, blit=True, interval=interval
    )

    if save:
        path = os.path.join(config.FIGURES_DIR, "mission_animation.gif")
        print(f"  Saving animation → {path}  (this may take ~30s...)")
        writer = animation.PillowWriter(fps=30)
        anim.save(path, writer=writer, dpi=110)
        print(f"  Saved.")

    if show:
        plt.show()

    return anim


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate a single AR3 ISR mission."
    )
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed (default: config.RANDOM_SEED)")
    parser.add_argument("--radars", type=int, default=8,
                        help="Number of radar nodes (default: 8)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving the .gif")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plt.show()")
    parser.add_argument("--interval", type=int, default=30,
                        help="ms between frames (default: 30)")

    args = parser.parse_args()

    print("Running mission animation...")
    animate(
        n_radars = args.radars,
        seed     = args.seed,
        save     = not args.no_save,
        show     = not args.no_show,
        interval = args.interval,
    )
