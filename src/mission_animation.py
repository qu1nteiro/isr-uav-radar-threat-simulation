"""
animate_mission.py — GIF animations for solo and cooperative ELINT missions.

Visual style: military/ISR surveillance software aesthetic.

Generates 4 animated GIFs:
    solo_gaussian.gif   — single ELINT drone, Gaussian walk
    solo_levy.gif       — single ELINT drone, Lévy walk
    coop_gaussian.gif   — two cooperative ELINT drones, Gaussian walk
    coop_levy.gif       — two cooperative ELINT drones, Lévy walk

Usage
-----
    python src/animate_mission.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import config
from radar_detection import global_detection
from network_builder import build_network
from uav_elint_trajectory import ELINTMap, _propagate_alerts, _decay_alerts
from cooperative_elint import _safe_step, _direction_vector, _levy_step

# ─── Palette ──────────────────────────────────────────────────────────────────
BG          = "#0d1117"
GRID_COL    = "#1f2937"
AXES_COL    = "#374151"
TEXT_DIM    = "#6b7280"
TEXT_MID    = "#94a3b8"
TEXT_BRIGHT = "#e2e8f0"
DRONE_A     = "#38bdf8"
DRONE_B     = "#c084fc"
KNOWN_COL   = "#f97316"
UNKNOWN_COL = "#dc262630"
SUCCESS_COL = "#22c55e"
DETECT_COL  = "#ef4444"
TIMEOUT_COL = "#6b7280"
BAR_OK      = "#22c55e"
BAR_DET     = "#ef4444"

N_FRAMES = 200
FPS      = 20
DPI      = 110


# ─── Detection probability helper ────────────────────────────────────────────

def _max_pd(pos, radar_positions, alert_states):
    """Maximum instantaneous detection probability across all radars."""
    d = np.linalg.norm(radar_positions - pos, axis=1)
    p = np.where(d <= config.RADAR_RANGE,
                 np.exp(-d / config.DETECTION_LAMBDA), 0.0)
    amp = 1.0 + config.ALERT_GAMMA * (alert_states > 0).astype(float)
    return float(np.clip(p * amp, 0, 1).max()) if len(p) else 0.0


# ─── Heatmap helper ───────────────────────────────────────────────────────────

def _heatmap(radar_positions, res=120):
    L  = config.GRID_SIZE
    xs = np.linspace(0, L, res)
    ys = np.linspace(0, L, res)
    Xg, Yg = np.meshgrid(xs, ys)
    P = np.ones((res, res))
    for rp in radar_positions:
        D  = np.sqrt((Xg-rp[0])**2 + (Yg-rp[1])**2)
        Pi = np.where(D <= config.RADAR_RANGE,
                      np.exp(-D / config.DETECTION_LAMBDA), 0.0)
        P *= (1.0 - Pi)
    return Xg, Yg, 1.0 - P


# ─── Mission recorders ────────────────────────────────────────────────────────

def record_solo(radar_positions, G, rng, walk="gaussian", beta=None):
    beta     = beta or config.ELINT_BETA
    N        = len(radar_positions)
    tgt      = np.array(config.MISSION_TARGET)
    use_levy = (walk == "levy")

    pos    = np.array(config.UAV_START, dtype=float)
    elint  = ELINTMap(N)
    alerts = np.zeros(N, dtype=int)

    positions    = [pos.copy()]
    known_masks  = [elint.known.copy()]
    alert_states = [alerts.copy()]
    pd_history   = [0.0]

    outcome = "timeout"; det_pos = None

    for t in range(config.MAX_STEPS):
        if np.linalg.norm(pos - tgt) <= config.TARGET_RADIUS:
            outcome = "success"; break

        elint.update(pos, radar_positions, t)
        kp = elint.known_positions(radar_positions)
        d  = _direction_vector(pos, kp, rng, beta)
        s  = _safe_step(pos, d,
                        _levy_step(rng) if use_levy else config.UAV_STEP_SIZE, kp)
        pos = np.clip(pos + s * d, 0.0, config.GRID_SIZE)

        det, ids, _ = global_detection(
            uav_pos=pos, radar_positions=radar_positions,
            alert_states=alerts, rng=rng)
        if det:
            outcome = "detected"; det_pos = tuple(pos); break
        if len(ids) > 0:
            _propagate_alerts(G, ids, alerts)
        _decay_alerts(alerts)

        positions.append(pos.copy())
        known_masks.append(elint.known.copy())
        alert_states.append(alerts.copy())
        pd_history.append(_max_pd(pos, radar_positions, alerts))

    return {"positions": np.array(positions), "known_masks": known_masks,
            "alert_states": alert_states, "pd_history": pd_history,
            "outcome": outcome, "det_pos": det_pos,
            "n_steps": len(positions)-1, "n_radars": N}


def record_coop(radar_positions, G, rng_a, rng_b, walk="gaussian", beta=None):
    beta     = beta or config.ELINT_BETA
    N        = len(radar_positions)
    tgt      = np.array(config.MISSION_TARGET)
    use_levy = (walk == "levy")

    pos_a = np.array(config.UAV_START_A, dtype=float)
    pos_b = np.array(config.UAV_START_B, dtype=float)
    ea    = ELINTMap(N); eb = ELINTMap(N)
    alerts = np.zeros(N, dtype=int)

    traj_a = [pos_a.copy()]; traj_b = [pos_b.copy()]
    known_sh = [(ea.known | eb.known).copy()]
    pd_a_hist = [0.0]; pd_b_hist = [0.0]

    out_a="timeout"; out_b="timeout"; det_a=None; det_b=None
    done_a=False;    done_b=False

    for t in range(config.MAX_STEPS):
        if not done_a and np.linalg.norm(pos_a-tgt) <= config.TARGET_RADIUS:
            out_a="success"; done_a=True
        if not done_b and np.linalg.norm(pos_b-tgt) <= config.TARGET_RADIUS:
            out_b="success"; done_b=True
        if done_a and done_b: break

        if not done_a: ea.update(pos_a, radar_positions, t)
        if not done_b: eb.update(pos_b, radar_positions, t)

        shared = ea.known | eb.known
        ea.known[:] = shared; eb.known[:] = shared

        if not done_a:
            kp = ea.known_positions(radar_positions)
            d  = _direction_vector(pos_a, kp, rng_a, beta,
                                   partner_pos=pos_b if not done_b else None)
            s  = _safe_step(pos_a, d,
                            _levy_step(rng_a) if use_levy else config.UAV_STEP_SIZE, kp)
            pos_a = np.clip(pos_a + s*d, 0, config.GRID_SIZE)
            traj_a.append(pos_a.copy())

        if not done_b:
            kp = eb.known_positions(radar_positions)
            d  = _direction_vector(pos_b, kp, rng_b, beta,
                                   partner_pos=pos_a if not done_a else None)
            s  = _safe_step(pos_b, d,
                            _levy_step(rng_b) if use_levy else config.UAV_STEP_SIZE, kp)
            pos_b = np.clip(pos_b + s*d, 0, config.GRID_SIZE)
            traj_b.append(pos_b.copy())

        if not done_a:
            det, ids, _ = global_detection(
                uav_pos=pos_a, radar_positions=radar_positions,
                alert_states=alerts, rng=rng_a)
            if det:
                out_a="detected"; det_a=tuple(pos_a); done_a=True
                if len(ids)>0: _propagate_alerts(G, ids, alerts)

        if not done_b:
            det, ids, _ = global_detection(
                uav_pos=pos_b, radar_positions=radar_positions,
                alert_states=alerts, rng=rng_b)
            if det:
                out_b="detected"; det_b=tuple(pos_b); done_b=True
                if len(ids)>0: _propagate_alerts(G, ids, alerts)

        _decay_alerts(alerts)
        T = max(len(traj_a), len(traj_b))
        known_sh.append(shared.copy())
        pd_a_hist.append(_max_pd(pos_a, radar_positions, alerts) if not done_a else pd_a_hist[-1])
        pd_b_hist.append(_max_pd(pos_b, radar_positions, alerts) if not done_b else pd_b_hist[-1])

        if done_a and done_b: break

    T = max(len(traj_a), len(traj_b))
    while len(traj_a)<T: traj_a.append(traj_a[-1])
    while len(traj_b)<T: traj_b.append(traj_b[-1])
    while len(known_sh)<T: known_sh.append(known_sh[-1])

    joint = ("both_success" if out_a=="success" and out_b=="success"
             else "one_success" if out_a=="success" or out_b=="success"
             else "both_failed")

    return {"positions_a": np.array(traj_a), "positions_b": np.array(traj_b),
            "known_shared": known_sh, "pd_a": pd_a_hist, "pd_b": pd_b_hist,
            "out_a": out_a, "out_b": out_b, "joint": joint,
            "det_a": det_a, "det_b": det_b,
            "n_steps": T-1, "n_radars": N}


# ─── Shared figure setup (ISR/surveillance aesthetic) ────────────────────────

def _make_fig(N_radars, seed, subtitle="", n_hud_lines=3):
    """
    Create the ISR display figure.
    HUD shows live coordinates, heading, P(d) — no progress bar.
    n_hud_lines: 3 for solo, 4 for coop (extra drone B line).
    """
    L       = config.GRID_SIZE
    hud_h   = 0.04 + n_hud_lines * 0.038   # dynamic height for HUD
    plot_b  = hud_h + 0.02                 # bottom of main axes
    fig     = plt.figure(figsize=(8.8, 9.2), dpi=DPI)
    fig.patch.set_facecolor(BG)

    # Main radar display
    ax = fig.add_axes([0.08, plot_b + 0.01, 0.88, 0.97 - plot_b - 0.07])
    ax.set_facecolor(BG)
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$  [spatial units ≈ km]", color=TEXT_DIM, fontsize=8)
    ax.set_ylabel("$x_2$  [spatial units ≈ km]", color=TEXT_DIM, fontsize=8, labelpad=2)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(AXES_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.3, alpha=0.8, zorder=0)
    for y in np.arange(0, L, 20):
        ax.axhspan(y, y+10, color="white", alpha=0.007, zorder=0)

    # Title bar
    fig.text(0.50, 0.985,
             f"AR3 ISR Mission Simulation  —  N = {N_radars} radars · seed {seed}",
             ha="center", va="top", color=TEXT_BRIGHT, fontsize=9.5,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.50, 0.966, subtitle,
             ha="center", va="top", color=TEXT_MID, fontsize=8,
             fontfamily="monospace")

    # HUD divider
    fig.add_artist(plt.matplotlib.lines.Line2D(
        [0.04, 0.96], [plot_b, plot_b], transform=fig.transFigure,
        color=AXES_COL, lw=0.8))

    # HUD text lines (dynamic)
    line_ys  = [plot_b - 0.006 - i * 0.038 for i in range(n_hud_lines)]
    hud_texts = [fig.text(0.05, y, "", color=TEXT_BRIGHT,
                          fontsize=8, fontfamily="monospace", va="top")
                 for y in line_ys]

    # ELINT coverage bar — matplotlib Rectangle (guaranteed rendering)
    bar_ax = fig.add_axes([0.05, 0.012, 0.90, 0.022])
    bar_ax.set_facecolor("#111827")
    bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    bar_fill = bar_ax.add_patch(
        __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (0, 0.1), 0.001, 0.8, color=DRONE_A, zorder=3))
    bar_label = bar_ax.text(0.5, 0.5, "ELINT MAP  0 / 0   0%",
                             ha="center", va="center", color=TEXT_DIM,
                             fontsize=7, fontfamily="monospace",
                             transform=bar_ax.transAxes, zorder=4)
    # Border
    for spine in ["top","bottom","left","right"]:
        bar_ax.spines[spine].set_color(AXES_COL)
        bar_ax.spines[spine].set_linewidth(0.5)
        bar_ax.spines[spine].set_visible(True)

    return fig, ax, hud_texts, bar_fill, bar_label


def _update_hud_solo(texts, bar_fill, bar_label,
                     step, total_steps, pos, pd, n_known, N, outcome=None):
    """Update 3-line solo HUD + matplotlib bar."""
    t     = step * config.STEP_DURATION_S / 60.0
    t_tot = total_steps * config.STEP_DURATION_S / 60.0
    dx    = config.MISSION_TARGET[0] - pos[0]
    dy    = config.MISSION_TARGET[1] - pos[1]
    hdg   = int(np.degrees(np.arctan2(dx, dy)) % 360)
    status = f"  ►  {outcome.upper()}" if outcome else ""
    col0   = (SUCCESS_COL if outcome=="success"
              else DETECT_COL if outcome=="detected" else TEXT_BRIGHT)
    pd_col = DETECT_COL if pd>0.3 else "#f59e0b" if pd>0.1 else TEXT_DIM

    texts[0].set_text(
        f"STEP  {step:>5d} / {total_steps}   |   T = {t:>6.1f} / {t_tot:.0f} min{status}")
    texts[0].set_color(col0)
    texts[1].set_text(
        f"x1 = {pos[0]:>7.2f} km     x2 = {pos[1]:>7.2f} km     HDG = {hdg:>03d} deg")
    texts[1].set_color(TEXT_MID)
    texts[2].set_text(f"Max P(d) = {pd*100:>5.1f}%   |   ELINT  {n_known:>2d} / {N} radars identified")
    texts[2].set_color(pd_col)

    # matplotlib bar
    frac = n_known / max(N, 1)
    bar_fill.set_width(max(frac, 0.002))
    bar_fill.set_color(SUCCESS_COL if outcome=="success"
                       else DETECT_COL if outcome=="detected" else DRONE_A)
    bar_label.set_text(
        f"ELINT MAP   {n_known} / {N} radars   {int(frac*100)}%")


def _update_hud_coop(texts, bar_fill, bar_label,
                     step, total_steps,
                     pos_a, pos_b, pd_a, pd_b,
                     n_shared, N, out_a=None, out_b=None):
    """Update 4-line coop HUD + matplotlib bar."""
    t      = step * config.STEP_DURATION_S / 60.0
    t_tot  = total_steps * config.STEP_DURATION_S / 60.0
    j_ok   = out_a=="success" and out_b=="success"
    any_ok = out_a=="success" or out_b=="success"
    status = ("  ►  BOTH SUCCESS" if j_ok
              else "  ►  ONE SUCCESS" if any_ok
              else "  ►  BOTH DETECTED" if (out_a and out_b) else "")
    col0   = SUCCESS_COL if j_ok else "#f59e0b" if any_ok else TEXT_BRIGHT

    def _line(pos, pd, out, marker):
        col = (SUCCESS_COL if out=="success"
               else DETECT_COL if out=="detected" else TEXT_MID)
        alert = " [ALERT]" if pd>0.3 else " [WARN] " if pd>0.1 else "        "
        return (f"{marker}  x1={pos[0]:>7.2f}  x2={pos[1]:>7.2f}"
                f"  P(d)={pd*100:>4.1f}%{alert}"), col

    la, ca = _line(pos_a, pd_a, out_a, "A *")
    lb, cb = _line(pos_b, pd_b, out_b, "B *")

    texts[0].set_text(
        f"STEP  {step:>5d} / {total_steps}   |   T = {t:>6.1f} / {t_tot:.0f} min{status}")
    texts[0].set_color(col0)
    texts[1].set_text(la); texts[1].set_color(ca)
    texts[2].set_text(lb); texts[2].set_color(cb)
    texts[3].set_text(f"COOPERATIVE ELINT MAP:  {n_shared} / {N} radars  [{int(n_shared/max(N,1)*100)}% identified]")
    texts[3].set_color(TEXT_DIM)

    # matplotlib bar
    frac = n_shared / max(N, 1)
    bar_fill.set_width(max(frac, 0.002))
    bar_fill.set_color(SUCCESS_COL if j_ok else "#f59e0b" if any_ok else DRONE_B)
    bar_label.set_text(f"SHARED MAP   {n_shared} / {N}   {int(frac*100)}%")


# ─── Solo GIF ─────────────────────────────────────────────────────────────────

def make_solo_gif(rec, radar_positions, walk_label, seed, output_path):
    from matplotlib.colors import LinearSegmentedColormap
    L   = config.GRID_SIZE
    N   = rec["n_radars"]
    Xg, Yg, P_heat = _heatmap(radar_positions)

    step_idx = np.unique(np.round(
        np.linspace(0, len(rec["positions"])-1, N_FRAMES)
    ).astype(int))

    cmap_d = LinearSegmentedColormap.from_list("d",
        [(0,"#dc262600"),(0.3,"#dc262620"),
         (0.6,"#dc262660"),(1,"#dc2626cc")], N=256)

    outcome = rec["outcome"]

    fig, ax, hud_texts, bar_fill, bar_label = _make_fig(
        N, seed, n_hud_lines=3,
        subtitle=f"Solo ELINT  ·  {walk_label}  ·  β* = {config.ELINT_BETA}"
    )

    ax.pcolormesh(Xg, Yg, P_heat, cmap=cmap_d, vmin=0, vmax=1,
                  shading="gouraud", rasterized=True, zorder=1)

    # ELINT rings
    for rp in radar_positions:
        ax.add_patch(Circle(rp, config.ELINT_RANGE, fill=False,
                            edgecolor="#f9731618", lw=0.6,
                            linestyle=":", zorder=2))

    # Target
    ax.add_patch(Circle(config.MISSION_TARGET, config.TARGET_RADIUS,
                        fill=True, facecolor="#22c55e18",
                        edgecolor="#22c55e", lw=1.2, zorder=3))
    ax.scatter(*config.MISSION_TARGET, s=80, c=SUCCESS_COL,
               marker="*", zorder=10, edgecolors=BG, lw=0.8)

    # Start
    ax.scatter(*config.UAV_START, s=55, c=DRONE_A,
               marker="o", zorder=10, edgecolors=BG, lw=1.0)

    # Animated elements
    trail, = ax.plot([], [], color=DRONE_A, lw=1.0, alpha=0.6, zorder=5)
    dot,   = ax.plot([], [], "o", color=DRONE_A, ms=7, zorder=11,
                     mec=BG, mew=1.0)

    radar_sc = ax.scatter(radar_positions[:,0], radar_positions[:,1],
                          s=22, c=[UNKNOWN_COL]*N, marker="^",
                          zorder=6, edgecolors="none")

    det_rings = [Circle(rp, config.RADAR_RANGE, fill=False,
                        edgecolor=UNKNOWN_COL, lw=0.5,
                        linestyle="--", zorder=4)
                 for rp in radar_positions]
    for r in det_rings: ax.add_patch(r)

    # Range indicator
    range_circle = ax.add_patch(
        Circle((0,0), config.ELINT_RANGE, fill=False,
               edgecolor="#f9731640", lw=0.6, linestyle=":", zorder=3)
    )

    total_steps = rec["n_steps"]

    def _update(fi):
        si   = step_idx[fi]
        si   = min(si, len(rec["positions"])-1)
        pos  = rec["positions"][si]
        mask = rec["known_masks"][min(si, len(rec["known_masks"])-1)]
        alrt = rec["alert_states"][min(si, len(rec["alert_states"])-1)]
        pd   = rec["pd_history"][min(si, len(rec["pd_history"])-1)]

        lo = max(0, si-80)
        trail.set_data(rec["positions"][lo:si+1, 0],
                       rec["positions"][lo:si+1, 1])
        dot.set_data([pos[0]], [pos[1]])
        range_circle.center = pos

        # Alert flicker
        dot.set_color(DETECT_COL if pd > 0.3 else
                      "#f59e0b"  if pd > 0.1 else DRONE_A)

        cols = [KNOWN_COL if mask[i] else
                ("#ef444460" if alrt[i] > 0 else UNKNOWN_COL)
                for i in range(N)]
        radar_sc.set_color(cols)

        for i, ring in enumerate(det_rings):
            if mask[i]:
                ring.set_edgecolor(KNOWN_COL); ring.set_alpha(0.7)
                ring.set_linewidth(0.9)
            elif alrt[i] > 0:
                ring.set_edgecolor("#ef4444"); ring.set_alpha(0.5)
                ring.set_linewidth(0.7)
            else:
                ring.set_edgecolor(UNKNOWN_COL); ring.set_alpha(0.3)
                ring.set_linewidth(0.4)

        fin_outcome = outcome if fi == len(step_idx)-1 else None
        _update_hud_solo(hud_texts, bar_fill, bar_label,
                         si, total_steps, pos, pd,
                         int(mask.sum()), N, outcome=fin_outcome)

        return trail, dot, radar_sc, range_circle, *hud_texts, *det_rings

    ani = animation.FuncAnimation(fig, _update, frames=len(step_idx),
                                  interval=1000//FPS, blit=True)
    ani.save(output_path, writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ─── Cooperative GIF ──────────────────────────────────────────────────────────

def make_coop_gif(rec, radar_positions, walk_label, seed, output_path):
    from matplotlib.colors import LinearSegmentedColormap
    L   = config.GRID_SIZE
    N   = rec["n_radars"]
    Xg, Yg, P_heat = _heatmap(radar_positions)

    step_idx = np.unique(np.round(
        np.linspace(0, len(rec["positions_a"])-1, N_FRAMES)
    ).astype(int))

    cmap_d = LinearSegmentedColormap.from_list("d",
        [(0,"#dc262600"),(0.3,"#dc262620"),
         (0.6,"#dc262660"),(1,"#dc2626cc")], N=256)

    joint = rec["joint"]

    fig, ax, hud_texts, bar_fill, bar_label = _make_fig(
        N, seed, n_hud_lines=4,
        subtitle=f"Cooperative ELINT  ·  {walk_label}  ·  β* = {config.ELINT_BETA}"
                 f"  ·  ΔStart = {int(config.UAV_START_B[1]-config.UAV_START_A[1])} u"
    )

    ax.pcolormesh(Xg, Yg, P_heat, cmap=cmap_d, vmin=0, vmax=1,
                  shading="gouraud", rasterized=True, zorder=1)

    for rp in radar_positions:
        ax.add_patch(Circle(rp, config.ELINT_RANGE, fill=False,
                            edgecolor="#f9731618", lw=0.6,
                            linestyle=":", zorder=2))

    ax.add_patch(Circle(config.MISSION_TARGET, config.TARGET_RADIUS,
                        fill=True, facecolor="#22c55e18",
                        edgecolor="#22c55e", lw=1.2, zorder=3))
    ax.scatter(*config.MISSION_TARGET, s=80, c=SUCCESS_COL,
               marker="*", zorder=10, edgecolors=BG, lw=0.8)
    ax.scatter(*config.UAV_START_A, s=50, c=DRONE_A,
               marker="o", zorder=10, edgecolors=BG, lw=1.0)
    ax.scatter(*config.UAV_START_B, s=50, c=DRONE_B,
               marker="D", zorder=10, edgecolors=BG, lw=1.0)

    trail_a, = ax.plot([], [], color=DRONE_A, lw=1.0, alpha=0.55, zorder=5)
    trail_b, = ax.plot([], [], color=DRONE_B, lw=1.0, alpha=0.55, zorder=5)
    dot_a,   = ax.plot([], [], "o", color=DRONE_A, ms=7, zorder=11,
                       mec=BG, mew=1.0)
    dot_b,   = ax.plot([], [], "D", color=DRONE_B, ms=6, zorder=11,
                       mec=BG, mew=1.0)

    radar_sc = ax.scatter(radar_positions[:,0], radar_positions[:,1],
                          s=22, c=[UNKNOWN_COL]*N, marker="^",
                          zorder=6, edgecolors="none")

    det_rings = [Circle(rp, config.RADAR_RANGE, fill=False,
                        edgecolor=UNKNOWN_COL, lw=0.5,
                        linestyle="--", zorder=4)
                 for rp in radar_positions]
    for r in det_rings: ax.add_patch(r)

    rc_a = ax.add_patch(Circle((0,0), config.ELINT_RANGE, fill=False,
                                edgecolor=f"{DRONE_A}50", lw=0.6,
                                linestyle=":", zorder=3))
    rc_b = ax.add_patch(Circle((0,0), config.ELINT_RANGE, fill=False,
                                edgecolor=f"{DRONE_B}50", lw=0.6,
                                linestyle=":", zorder=3))

    # Legend
    leg = ax.legend(handles=[
        Line2D([0],[0], color=DRONE_A, lw=1.5, marker="o", ms=5,
               label=f"Drone A  ({int(config.UAV_START_A[0])},{int(config.UAV_START_A[1])})"),
        Line2D([0],[0], color=DRONE_B, lw=1.5, marker="D", ms=5,
               label=f"Drone B  ({int(config.UAV_START_B[0])},{int(config.UAV_START_B[1])})"),
    ], loc="upper left", fontsize=7, facecolor="#1e293b",
       edgecolor=AXES_COL, labelcolor=TEXT_MID, framealpha=0.9)

    total_steps = rec["n_steps"]

    def _update(fi):
        si  = step_idx[fi]
        pa  = rec["positions_a"]
        pb  = rec["positions_b"]
        sh  = rec["known_shared"]
        mask = sh[min(si, len(sh)-1)]
        pda  = rec["pd_a"][min(si, len(rec["pd_a"])-1)]
        pdb  = rec["pd_b"][min(si, len(rec["pd_b"])-1)]

        lo = max(0, si-80)
        ia = min(si, len(pa)-1); ib = min(si, len(pb)-1)
        trail_a.set_data(pa[lo:ia+1, 0], pa[lo:ia+1, 1])
        trail_b.set_data(pb[lo:ib+1, 0], pb[lo:ib+1, 1])
        dot_a.set_data([pa[ia,0]], [pa[ia,1]])
        dot_b.set_data([pb[ib,0]], [pb[ib,1]])
        rc_a.center = pa[ia]; rc_b.center = pb[ib]

        dot_a.set_color(DETECT_COL if pda>0.3 else "#f59e0b" if pda>0.1 else DRONE_A)
        dot_b.set_color(DETECT_COL if pdb>0.3 else "#f59e0b" if pdb>0.1 else DRONE_B)

        cols = [KNOWN_COL if mask[i] else UNKNOWN_COL for i in range(N)]
        radar_sc.set_color(cols)
        for i, ring in enumerate(det_rings):
            if mask[i]:
                ring.set_edgecolor(KNOWN_COL); ring.set_alpha(0.7)
                ring.set_linewidth(0.9)
            else:
                ring.set_edgecolor(UNKNOWN_COL); ring.set_alpha(0.3)
                ring.set_linewidth(0.4)

        oa = rec["out_a"] if fi==len(step_idx)-1 else None
        ob = rec["out_b"] if fi==len(step_idx)-1 else None
        _update_hud_coop(hud_texts, bar_fill, bar_label,
                         si, total_steps,
                         pa[ia], pb[ib], pda, pdb,
                         int(mask.sum()), N, out_a=oa, out_b=ob)

        return (trail_a, trail_b, dot_a, dot_b, radar_sc,
                rc_a, rc_b, *hud_texts, *det_rings)

    ani = animation.FuncAnimation(fig, _update, frames=len(step_idx),
                                  interval=1000//FPS, blit=True)
    ani.save(output_path, writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _t

    t0   = _t.time()
    seed = 59        # verified: both_success on Gaussian AND Lévy, N=9
    N_R  = 9

    print("Generating AR3 ISR mission animations")
    print(f"  N={N_R} radars, seed={seed}, {N_FRAMES} frames @ {FPS}fps\n")

    rng_sp = np.random.default_rng(seed)
    L, mg  = config.GRID_SIZE, config.RADAR_RANGE * 0.5
    rp     = rng_sp.uniform(mg, L-mg, size=(N_R, 2))
    G      = build_network(rp, "ER", rng=rng_sp)

    out    = config.FIGURES_DIR

    print("[1/4] Solo — Gaussian...")
    rec = record_solo(rp, G, np.random.default_rng(seed+1), "gaussian")
    print(f"      {rec['outcome']}  steps={rec['n_steps']}  known={rec['known_masks'][-1].sum()}/{N_R}")
    make_solo_gif(rec, rp, "Gaussian walk", seed,
                  os.path.join(out, "solo_gaussian.gif"))

    print("[2/4] Solo — Lévy...")
    rec = record_solo(rp, G, np.random.default_rng(seed+2), "levy")
    print(f"      {rec['outcome']}  steps={rec['n_steps']}  known={rec['known_masks'][-1].sum()}/{N_R}")
    make_solo_gif(rec, rp, f"Lévy walk (μ={config.LEVY_ALPHA})", seed,
                  os.path.join(out, "solo_levy.gif"))

    print("[3/4] Cooperative — Gaussian...")
    rec = record_coop(rp, G, np.random.default_rng(seed+3),
                      np.random.default_rng(seed+4), "gaussian")
    print(f"      A={rec['out_a']} B={rec['out_b']} ({rec['joint']})  shared={rec['known_shared'][-1].sum()}/{N_R}")
    make_coop_gif(rec, rp, "Gaussian walk", seed,
                  os.path.join(out, "coop_gaussian.gif"))

    print("[4/4] Cooperative — Lévy...")
    rec = record_coop(rp, G, np.random.default_rng(seed+5),
                      np.random.default_rng(seed+6), "levy")
    print(f"      A={rec['out_a']} B={rec['out_b']} ({rec['joint']})  shared={rec['known_shared'][-1].sum()}/{N_R}")
    make_coop_gif(rec, rp, f"Lévy walk (μ={config.LEVY_ALPHA})", seed,
                  os.path.join(out, "coop_levy.gif"))

    print(f"\nDone in {(_t.time()-t0)/60:.1f} minutes.")
