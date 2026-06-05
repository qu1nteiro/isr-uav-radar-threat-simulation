"""
phase_transition.py — Stage 4: phase transition analysis.

Sweeps radar spatial density ρ = N/L² from RHO_MIN to RHO_MAX and measures
the mission success rate S(ρ) for three walk modes:
    - Blind Gaussian (no ELINT)
    - ELINT Gaussian
    - ELINT Lévy

Four analyses are performed:
    1. S(ρ) curves with 95% confidence bands
    2. Logistic fit S(ρ) = 1/(1+exp(k(ρ−ρ_c))) and sharpness parameter k
    3. Alert propagation contribution: γ=0 vs γ=0.5
    4. Spatial detection heatmaps near ρ_c

Usage
-----
    python src/phase_transition.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import os
import time

import config
from network_builder import build_network, TOPO_COLOURS, TOPO_LABELS
from uav_trajectory import run_ensemble
from uav_elint_trajectory import run_elint_ensemble


# ─── Walk configuration ───────────────────────────────────────────────────────

WALKS = {
    "blind_gaussian": {
        "label":     "Blind — Gaussian",
        "colour":    "#5F5E5A",
        "linestyle": "--",
        "elint":     False,
        "walk":      "gaussian",
    },
    "elint_gaussian": {
        "label":     "ELINT — Gaussian",
        "colour":    "#185FA5",
        "linestyle": "-",
        "elint":     True,
        "walk":      "gaussian",
    },
    "elint_levy": {
        "label":     f"ELINT — Lévy  ($\\mu={config.LEVY_ALPHA}$)",
        "colour":    "#0F6E56",
        "linestyle": "-",
        "elint":     True,
        "walk":      "levy",
    },
}


# ─── Radar placement ──────────────────────────────────────────────────────────

def _place_radars_for_rho(rho, rng):
    """Place N = round(ρ·L²) radars uniformly in the domain interior."""
    L  = config.GRID_SIZE
    N  = max(1, round(rho * L ** 2))
    mg = config.RADAR_RANGE * 0.5
    return rng.uniform(mg, L - mg, size=(N, 2)), N


# ─── Single-ρ worker (parallelised) ──────────────────────────────────────────

def _run_single_rho(rho, topology, use_elint, walk_type,
                    n_runs, beta, seed_base, gamma_override=None):
    """
    Run all missions for one (ρ, topology, walk) combination.

    gamma_override : float or None
        If set, temporarily replaces config.ALERT_GAMMA for this worker.
        Safe in multiprocessing because each worker has its own config copy.
    """
    topo_offset = {"ER": 3_000_000, "BA": 1_000_000, "WS": 2_000_000}
    rho_seed    = seed_base + hash(round(rho, 8)) % (2 ** 31) + topo_offset[topology]
    rng_local   = np.random.default_rng(abs(rho_seed))

    # Temporarily override alert amplification if requested
    original_gamma = config.ALERT_GAMMA
    if gamma_override is not None:
        config.ALERT_GAMMA = gamma_override

    try:
        radar_positions, N = _place_radars_for_rho(rho, rng_local)
        G = build_network(radar_positions, topology, rng=rng_local)

        if use_elint:
            stats, results = run_elint_ensemble(
                radar_positions, G, rng_local,
                n=n_runs, walk=walk_type, beta=beta
            )
        else:
            stats, results = run_ensemble(
                radar_positions, G, rng_local,
                n=n_runs, walk=walk_type
            )

        # Collect detection positions for heatmap
        det_positions = [
            r.detection_pos for r in results
            if r.detection_pos is not None
        ]

    finally:
        config.ALERT_GAMMA = original_gamma

    return (N, stats.success_rate, stats.detection_rate,
            stats.timeout_rate, det_positions)


# ─── Density sweep ────────────────────────────────────────────────────────────

def sweep_density(walk_key, topology, rng,
                  rho_vals=None, n_runs=None, beta=None,
                  gamma_override=None, verbose=True):
    """
    Sweep radar density and measure S(ρ) with 95% confidence intervals.

    Returns
    -------
    rho_vals, success_rates, ci_95, detection_rates, timeout_rates,
    n_radars, detection_positions_by_rho
    """
    from joblib import Parallel, delayed

    if rho_vals is None:
        rho_vals = np.linspace(config.RHO_MIN, config.RHO_MAX, config.RHO_STEPS)
    if n_runs is None:
        n_runs = config.N_PHASE_RUNS
    if beta is None:
        beta = config.ELINT_BETA

    walk_cfg  = WALKS[walk_key]
    use_elint = walk_cfg["elint"]
    walk_type = walk_cfg["walk"]

    n_pts = len(rho_vals)
    success_rates   = np.zeros(n_pts)
    ci_95           = np.zeros(n_pts)
    detection_rates = np.zeros(n_pts)
    timeout_rates   = np.zeros(n_pts)
    n_radars_arr    = np.zeros(n_pts, dtype=int)
    det_pos_by_rho  = [[] for _ in range(n_pts)]

    raw = Parallel(n_jobs=config.N_JOBS, verbose=0)(
        delayed(_run_single_rho)(
            rho, topology, use_elint, walk_type,
            n_runs, beta, config.RANDOM_SEED, gamma_override
        )
        for rho in rho_vals
    )

    for idx, (N, s, d, t, dpos) in enumerate(raw):
        n_radars_arr[idx]    = N
        success_rates[idx]   = s
        detection_rates[idx] = d
        timeout_rates[idx]   = t
        det_pos_by_rho[idx]  = dpos
        # 95% CI for a proportion: 1.96 * sqrt(p*(1-p)/n)
        ci_95[idx] = 1.96 * np.sqrt(max(s * (1 - s), 1e-9) / n_runs)

        if verbose:
            print(f"    ρ={rho_vals[idx]:.5f}  N={N:>3d}  "
                  f"S={s*100:>5.1f}%  "
                  f"D={d*100:>5.1f}%  "
                  f"T={t*100:>5.1f}%  "
                  f"CI±{ci_95[idx]*100:.1f}%")

    return (rho_vals, success_rates, ci_95, detection_rates,
            timeout_rates, n_radars_arr, det_pos_by_rho)


# ─── Critical density and logistic fit ────────────────────────────────────────

def _logistic(rho, k, rho_c):
    """Logistic decay: S = 1 / (1 + exp(k·(ρ − ρ_c)))."""
    return 1.0 / (1.0 + np.exp(np.clip(k * (rho - rho_c), -500, 500)))


def find_rho_c(rho_vals, success_rates, smooth_sigma=1.5):
    """
    Estimate ρ_c from inflection point of smoothed S(ρ).

    Returns rho_c, s_at_rho_c, s_smooth, ds_drho.
    """
    s_smooth = gaussian_filter1d(success_rates, sigma=smooth_sigma)
    ds_drho  = np.gradient(s_smooth, rho_vals)
    idx_c    = np.argmin(ds_drho)
    return (float(rho_vals[idx_c]), float(s_smooth[idx_c]),
            s_smooth, ds_drho)


def fit_logistic(rho_vals, success_rates, smooth_sigma=1.5):
    """
    Fit a logistic function to S(ρ) and extract sharpness k and ρ_c.

    The fit is performed on smoothed data for numerical stability.
    k characterises the transition width: larger k = sharper = more critical.

    Returns
    -------
    k_fit : float       sharpness parameter
    rho_c_fit : float   critical density from fit
    s_fit : np.ndarray  fitted curve
    rmse : float        root mean square error of fit
    success : bool      True if fit converged
    """
    s_smooth = gaussian_filter1d(success_rates, sigma=smooth_sigma)
    ds       = np.gradient(s_smooth, rho_vals)
    rho_c0   = float(rho_vals[np.argmin(ds)])
    k0       = 5000.0

    try:
        popt, _ = curve_fit(
            _logistic, rho_vals, s_smooth,
            p0=[k0, rho_c0],
            bounds=([0, rho_vals[0]], [1e8, rho_vals[-1]]),
            maxfev=20000,
        )
        k_fit, rho_c_fit = popt
        s_fit  = _logistic(rho_vals, *popt)
        rmse   = float(np.sqrt(np.mean((s_smooth - s_fit) ** 2)))
        return k_fit, rho_c_fit, s_fit, rmse, True
    except Exception:
        return k0, rho_c0, s_smooth, np.nan, False


# ─── Full analysis ────────────────────────────────────────────────────────────

def run_phase_analysis(seed=None, n_runs=None, verbose=True):
    """
    Run 3 walks × 3 topologies × RHO_STEPS density values.
    Stores CI bands, logistic fit, and detection positions.
    """
    seed   = seed   if seed   is not None else config.RANDOM_SEED
    n_runs = n_runs if n_runs is not None else config.N_PHASE_RUNS
    rng    = np.random.default_rng(seed)
    rho_vals = np.linspace(config.RHO_MIN, config.RHO_MAX, config.RHO_STEPS)
    results  = {}

    for walk_key, walk_cfg in WALKS.items():
        results[walk_key] = {}
        for topo in ("ER", "BA", "WS"):
            print(f"\n  [{walk_cfg['label']}  ×  {TOPO_LABELS[topo]}]")

            (rho_v, s_r, ci, d_r, t_r,
             n_r, dpos) = sweep_density(
                walk_key, topo, rng,
                rho_vals=rho_vals, n_runs=n_runs, verbose=verbose,
            )

            rho_c, s_at_c, s_smooth, ds_drho = find_rho_c(rho_v, s_r)
            k_fit, rho_c_fit, s_fit, rmse, ok = fit_logistic(rho_v, s_r)

            results[walk_key][topo] = {
                "rho_vals":        rho_v,
                "success_rates":   s_r,
                "ci_95":           ci,
                "detection_rates": d_r,
                "timeout_rates":   t_r,
                "n_radars":        n_r,
                "rho_c":           rho_c,
                "s_at_rho_c":      s_at_c,
                "s_smooth":        s_smooth,
                "ds_drho":         ds_drho,
                "k_fit":           k_fit,
                "rho_c_fit":       rho_c_fit,
                "s_fit":           s_fit,
                "fit_rmse":        rmse,
                "fit_ok":          ok,
                "det_positions":   dpos,
            }

            print(f"    → ρ_c = {rho_c:.5f}  "
                  f"(N_c ≈ {round(rho_c * config.GRID_SIZE**2)} radars)  "
                  f"k = {k_fit:.0f}  RMSE = {rmse:.4f}")

    return results


# ─── Gamma comparison ─────────────────────────────────────────────────────────

def run_gamma_comparison(seed=None, n_runs=None, verbose=True):
    """
    Run ELINT Gaussian sweep with γ=0 (independent radars) across all
    topologies and compare with γ=0.5 (alert propagation active).

    This isolates the contribution of network communication to defence.

    Returns
    -------
    gamma_results : dict
        gamma_results[topo] → {rho_vals, s_gamma0, s_gamma05, ci_gamma0,
                                ci_gamma05, delta_s}
    """
    seed   = seed   if seed   is not None else config.RANDOM_SEED
    n_runs = n_runs if n_runs is not None else config.N_PHASE_RUNS
    rng    = np.random.default_rng(seed + 999_999)
    rho_vals = np.linspace(config.RHO_MIN, config.RHO_MAX, config.RHO_STEPS)

    gamma_results = {}

    for topo in ("ER", "BA", "WS"):
        print(f"\n  [γ comparison  ×  {TOPO_LABELS[topo]}]")

        # γ = 0 — radars detect independently, no alert propagation
        print(f"    γ = 0.0  (no alert propagation):")
        (_, s0, ci0, _, _, _, _) = sweep_density(
            "elint_gaussian", topo, rng,
            rho_vals=rho_vals, n_runs=n_runs,
            gamma_override=0.0, verbose=verbose,
        )

        # γ = 0.5 — full alert propagation (re-run for consistent seed)
        print(f"    γ = {config.ALERT_GAMMA}  (alert propagation active):")
        (_, s05, ci05, _, _, _, _) = sweep_density(
            "elint_gaussian", topo, rng,
            rho_vals=rho_vals, n_runs=n_runs,
            gamma_override=config.ALERT_GAMMA, verbose=verbose,
        )

        s0_sm  = gaussian_filter1d(s0,  sigma=1.5)
        s05_sm = gaussian_filter1d(s05, sigma=1.5)

        gamma_results[topo] = {
            "rho_vals":   rho_vals,
            "s_gamma0":   s0,
            "s_gamma05":  s05,
            "s0_smooth":  s0_sm,
            "s05_smooth": s05_sm,
            "ci_gamma0":  ci0,
            "ci_gamma05": ci05,
            "delta_s":    (s05_sm - s0_sm) * 100,   # pp — how much γ hurts/helps
        }

    return gamma_results


# ─── Detection heatmaps ───────────────────────────────────────────────────────

def compute_detection_heatmaps(results, n_heatmap=500, seed=None):
    """
    For each topology, run n_heatmap ELINT Gaussian missions at ρ ≈ ρ_c
    and build a 2D histogram of detection positions.

    Parameters
    ----------
    results : dict   from run_phase_analysis()
    n_heatmap : int  missions to run per topology

    Returns
    -------
    heatmaps : dict
        heatmaps[topo] → {"H": 2D array, "rho_c": float, "n_det": int}
    """
    seed = seed if seed is not None else config.RANDOM_SEED
    L    = config.GRID_SIZE
    bins = 40
    heatmaps = {}

    for topo in ("ER", "BA", "WS"):
        rho_c = results["elint_gaussian"][topo]["rho_c"]
        print(f"\n  [Heatmap  ×  {TOPO_LABELS[topo]}  at ρ_c={rho_c:.5f}]")

        rng_local = np.random.default_rng(seed + hash(topo) % (2 ** 20))
        radar_positions, N = _place_radars_for_rho(rho_c, rng_local)
        G = build_network(radar_positions, topo, rng=rng_local)

        # Collect detection positions from all missions
        all_det = []
        _, mission_results = run_elint_ensemble(
            radar_positions, G, rng_local,
            n=n_heatmap, walk="gaussian", beta=config.ELINT_BETA
        )
        for r in mission_results:
            if r.detection_pos is not None:
                all_det.append(r.detection_pos)

        n_det = len(all_det)
        print(f"    Detections: {n_det}/{n_heatmap} ({n_det/n_heatmap*100:.1f}%)")

        if n_det > 0:
            xs = [p[0] for p in all_det]
            ys = [p[1] for p in all_det]
            H, xedges, yedges = np.histogram2d(
                xs, ys,
                bins=bins, range=[[0, L], [0, L]]
            )
            H = H / n_heatmap   # normalise by total missions
        else:
            H = np.zeros((bins, bins))
            xedges = np.linspace(0, L, bins + 1)
            yedges = np.linspace(0, L, bins + 1)

        heatmaps[topo] = {
            "H":       H,
            "xedges":  xedges,
            "yedges":  yedges,
            "rho_c":   rho_c,
            "N":       N,
            "n_det":   n_det,
            "n_total": n_heatmap,
            "radar_positions": radar_positions,
        }

    return heatmaps


# ─── Figure 1: Main phase transition (with CI bands) ─────────────────────────

def plot_phase_transition(results, seed=None, save=True):
    """
    Main Stage 4 figure — 2 rows × 3 cols.
    S(ρ) curves with 95% CI bands, ρ_c bars, derivative, ΔS.
    """
    rho_arr = results["blind_gaussian"]["ER"]["rho_vals"]
    L       = config.GRID_SIZE
    rho_now = 20 / L ** 2

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # ── Row 0: S(ρ) per topology with CI ─────────────────────────────────────
    for col, topo in enumerate(("ER", "BA", "WS")):
        ax = fig.add_subplot(gs[0, col])

        for walk_key, walk_cfg in WALKS.items():
            r    = results[walk_key][topo]
            cc   = walk_cfg["colour"]
            ls   = walk_cfg["linestyle"]
            rho  = r["rho_vals"] * 1e3
            s_sm = r["s_smooth"] * 100
            ci   = r["ci_95"] * 100

            # CI band on smoothed curve
            ax.fill_between(rho,
                            np.clip(s_sm - ci, 0, 100),
                            np.clip(s_sm + ci, 0, 100),
                            color=cc, alpha=0.12)
            # Raw data (faint)
            ax.plot(rho, r["success_rates"] * 100,
                    color=cc, lw=0.5, alpha=0.25, linestyle=ls)
            # Smoothed
            ax.plot(rho, s_sm,
                    color=cc, lw=2.0, linestyle=ls,
                    label=walk_cfg["label"])
            # ρ_c marker
            ax.axvline(r["rho_c"] * 1e3, color=cc,
                       lw=0.8, linestyle=":", alpha=0.7)

        ax.axvline(rho_now * 1e3, color="#dc2626", lw=1.0,
                   linestyle="-.", alpha=0.8, label="Current ρ (N=20)")

        ax.set_xlabel("Radar density  $\\rho$  [$\\times 10^{-3}$ u$^{-2}$]",
                      fontsize=9)
        ax.set_ylabel("Success rate  $S(\\rho)$  [%]", fontsize=9)
        ax.set_title(f"{TOPO_LABELS[topo]} network",
                     fontsize=9, fontweight="bold",
                     color=TOPO_COLOURS[topo])
        ax.set_xlim(rho_arr[0] * 1e3, rho_arr[-1] * 1e3)
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cbd5e1",
                  loc="upper right")
        ax.grid(True, linewidth=0.3, alpha=0.4)

        # Second axis: N radars
        ax2 = ax.twiny()
        n_ticks   = np.array([5, 10, 20, 30, 50])
        rho_ticks = n_ticks / L ** 2
        valid     = (rho_ticks >= rho_arr[0]) & (rho_ticks <= rho_arr[-1])
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(rho_ticks[valid] * 1e3)
        ax2.set_xticklabels([str(n) for n in n_ticks[valid]], fontsize=7)
        ax2.set_xlabel("$N$ radars", fontsize=7.5, labelpad=3)

    # ── Row 1, col 0: ρ_c bar chart ───────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 0])
    topos   = ["ER", "BA", "WS"]
    x       = np.arange(len(topos))
    width   = 0.25

    for wi, (wk, off) in enumerate(zip(WALKS, [-width, 0, width])):
        vals = [results[wk][t]["rho_c"] * 1e3 for t in topos]
        bars = ax_bar.bar(x + off, vals, width * 0.9,
                          color=WALKS[wk]["colour"], alpha=0.85,
                          label=WALKS[wk]["label"],
                          edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.2f}",
                        ha="center", va="bottom",
                        fontsize=6.5, color=WALKS[wk]["colour"])

    ax_bar.axhline(rho_now * 1e3, color="#dc2626", lw=1.0,
                   linestyle="-.", alpha=0.8, label="Current ρ")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([TOPO_LABELS[t] for t in topos], fontsize=8)
    ax_bar.set_ylabel("$\\rho_c$  [$\\times 10^{-3}$ u$^{-2}$]", fontsize=9)
    ax_bar.set_title("Critical density $\\rho_c$\nby walk and topology",
                     fontsize=9, fontweight="bold")
    ax_bar.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax_bar.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    # ── Row 1, col 1: dS/dρ derivative (ER) ──────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    for walk_key, walk_cfg in WALKS.items():
        r = results[walk_key]["ER"]
        ax_d.plot(r["rho_vals"] * 1e3, r["ds_drho"] * 1e3,
                  color=walk_cfg["colour"], lw=1.8,
                  linestyle=walk_cfg["linestyle"],
                  label=walk_cfg["label"])
        ax_d.axvline(r["rho_c"] * 1e3, color=walk_cfg["colour"],
                     lw=0.8, linestyle=":", alpha=0.7)

    ax_d.axhline(0, color="#94a3b8", lw=0.6, linestyle="--", alpha=0.5)
    ax_d.set_xlabel("Radar density  $\\rho$  [$\\times 10^{-3}$ u$^{-2}$]",
                    fontsize=9)
    ax_d.set_ylabel("$dS/d\\rho$  [$\\times 10^{-3}$]", fontsize=9)
    ax_d.set_title("Derivative $dS/d\\rho$ — ER network\n(inflection = $\\rho_c$)",
                   fontsize=9, fontweight="bold")
    ax_d.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax_d.grid(True, linewidth=0.3, alpha=0.4)
    ax_d.set_xlim(rho_arr[0] * 1e3, rho_arr[-1] * 1e3)

    # ── Row 1, col 2: ΔS ELINT benefit ───────────────────────────────────────
    ax_ds = fig.add_subplot(gs[1, 2])

    for topo in topos:
        s_blind = results["blind_gaussian"][topo]["s_smooth"]
        s_eg    = results["elint_gaussian"][topo]["s_smooth"]
        s_el    = results["elint_levy"][topo]["s_smooth"]

        ax_ds.plot(rho_arr * 1e3, (s_eg - s_blind) * 100,
                   color=TOPO_COLOURS[topo], lw=1.5, linestyle="-",
                   label=f"{TOPO_LABELS[topo]} — ELINT Gauss")
        ax_ds.plot(rho_arr * 1e3, (s_el - s_blind) * 100,
                   color=TOPO_COLOURS[topo], lw=1.5, linestyle="--",
                   label=f"{TOPO_LABELS[topo]} — ELINT Lévy")

    ax_ds.axhline(0, color="#94a3b8", lw=0.8, linestyle="--", alpha=0.6)
    ax_ds.axvline(rho_now * 1e3, color="#dc2626", lw=1.0,
                  linestyle="-.", alpha=0.8, label="Current ρ")
    ax_ds.set_xlabel("Radar density  $\\rho$  [$\\times 10^{-3}$ u$^{-2}$]",
                     fontsize=9)
    ax_ds.set_ylabel("$\\Delta S$  [pp]", fontsize=9)
    ax_ds.set_title("ELINT benefit over blind walk\n$\\Delta S(\\rho)$",
                    fontsize=9, fontweight="bold")
    ax_ds.legend(fontsize=7, framealpha=0.9, edgecolor="#cbd5e1", ncols=2)
    ax_ds.grid(True, linewidth=0.3, alpha=0.4)
    ax_ds.set_xlim(rho_arr[0] * 1e3, rho_arr[-1] * 1e3)

    seed_val = seed if seed is not None else config.RANDOM_SEED
    fig.suptitle(
        f"Stage 4 — Phase transition analysis   "
        f"($N_{{\\mathrm{{runs}}}}={config.N_PHASE_RUNS}$, seed {seed_val})",
        fontsize=11, fontweight="500"
    )

    if save:
        path = os.path.join(config.FIGURES_DIR, "stage4_phase_transition.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[Stage 4] Figure saved → {path}")

    return fig


# ─── Figure 2: Advanced analysis ─────────────────────────────────────────────

def plot_advanced_analysis(results, gamma_results, heatmaps, seed=None, save=True):
    """
    Advanced Stage 4 figure — 2 rows × 3 cols:

        Row 0: Detection heatmaps at ρ_c  (ER, BA, WS)
        Row 1: Logistic fit + k parameter | k bar chart | γ comparison
    """
    topos   = ["ER", "BA", "WS"]
    rho_arr = results["blind_gaussian"]["ER"]["rho_vals"]
    L       = config.GRID_SIZE
    rho_now = 20 / L ** 2

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    cmap_heat = LinearSegmentedColormap.from_list(
        "heat",
        [(0.0, "#0d1117"), (0.3, "#7f1d1d"),
         (0.7, "#dc2626"), (1.0, "#fbbf24")],
        N=256
    )

    # ── Row 0: Detection heatmaps ─────────────────────────────────────────────
    for col, topo in enumerate(topos):
        ax  = fig.add_subplot(gs[0, col])
        hm  = heatmaps[topo]
        H   = hm["H"].T   # transpose for correct orientation
        rp  = hm["radar_positions"]

        im = ax.pcolormesh(
            hm["xedges"], hm["yedges"], H,
            cmap=cmap_heat, shading="auto"
        )
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01,
                     label="Detection density")

        # Radar coverage rings
        for r in rp:
            ax.add_patch(plt.Circle(r, config.RADAR_RANGE,
                                    fill=False, edgecolor="#dc262650",
                                    lw=0.5, linestyle="--"))
        ax.scatter(rp[:, 0], rp[:, 1],
                   s=20, c="#dc2626", marker="^",
                   zorder=3, edgecolors="#fff", linewidths=0.4)
        ax.scatter(*config.UAV_START, s=60, c="#185FA5",
                   marker="o", zorder=4, edgecolors="#fff", linewidths=1.0)
        ax.scatter(*config.MISSION_TARGET, s=70, c="#3B6D11",
                   marker="*", zorder=4, edgecolors="#fff", linewidths=0.8)

        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$ [u]", fontsize=9)
        ax.set_ylabel("$x_2$ [u]", fontsize=9)
        n_det = hm["n_det"]; n_tot = hm["n_total"]
        ax.set_title(
            f"Detection heatmap — {TOPO_LABELS[topo]}\n"
            f"at $\\rho_c$ (N={hm['N']} radars)  "
            f"det. rate {n_det/n_tot*100:.1f}%",
            fontsize=8.5, fontweight="bold", color=TOPO_COLOURS[topo]
        )
        ax.grid(True, linewidth=0.2, alpha=0.2)

    # ── Row 1, col 0: Logistic fits (BA — most interesting topology) ──────────
    ax_fit = fig.add_subplot(gs[1, 0])
    topo_fit = "BA"

    for walk_key, walk_cfg in WALKS.items():
        r   = results[walk_key][topo_fit]
        cc  = walk_cfg["colour"]
        ls  = walk_cfg["linestyle"]
        rho = r["rho_vals"] * 1e3

        ax_fit.scatter(rho, r["success_rates"] * 100,
                       color=cc, s=10, alpha=0.4, zorder=2)
        if r["fit_ok"]:
            ax_fit.plot(rho, r["s_fit"] * 100,
                        color=cc, lw=2.0, linestyle=ls,
                        label=f"{walk_cfg['label']}  "
                              f"(k={r['k_fit']:.0f})")
        else:
            ax_fit.plot(rho, r["s_smooth"] * 100,
                        color=cc, lw=2.0, linestyle=ls,
                        label=walk_cfg["label"] + " (no fit)")

    ax_fit.set_xlabel("Radar density  $\\rho$  [$\\times 10^{-3}$ u$^{-2}$]",
                      fontsize=9)
    ax_fit.set_ylabel("$S(\\rho)$  [%]", fontsize=9)
    ax_fit.set_title(f"Logistic fit  $S = 1/(1+e^{{k(\\rho-\\rho_c)}})$\n"
                     f"{TOPO_LABELS[topo_fit]} network",
                     fontsize=9, fontweight="bold")
    ax_fit.set_ylim(-2, 102)
    ax_fit.set_xlim(rho_arr[0] * 1e3, rho_arr[-1] * 1e3)
    ax_fit.legend(fontsize=8, framealpha=0.9, edgecolor="#cbd5e1")
    ax_fit.grid(True, linewidth=0.3, alpha=0.4)

    # ── Row 1, col 1: k parameter bar chart ───────────────────────────────────
    ax_k = fig.add_subplot(gs[1, 1])
    x     = np.arange(len(topos))
    width = 0.25

    for wi, (wk, off) in enumerate(zip(WALKS, [-width, 0, width])):
        k_vals = []
        for t in topos:
            r = results[wk][t]
            k_vals.append(r["k_fit"] if r["fit_ok"] else 0.0)
        bars = ax_k.bar(x + off, k_vals, width * 0.9,
                        color=WALKS[wk]["colour"], alpha=0.85,
                        label=WALKS[wk]["label"],
                        edgecolor="white", linewidth=0.5)
        for bar, kv in zip(bars, k_vals):
            if kv > 0:
                ax_k.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 50,
                          f"{kv:.0f}", ha="center", va="bottom",
                          fontsize=6.5, color=WALKS[wk]["colour"])

    ax_k.set_xticks(x)
    ax_k.set_xticklabels([TOPO_LABELS[t] for t in topos], fontsize=8)
    ax_k.set_ylabel("Sharpness  $k$  [u²]", fontsize=9)
    ax_k.set_title("Transition sharpness $k$\nlarger = more critical-like",
                   fontsize=9, fontweight="bold")
    ax_k.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cbd5e1")
    ax_k.grid(True, linewidth=0.3, alpha=0.4, axis="y")
    ax_k.annotate("Higher k → sharper\ntransition → more critical",
                  xy=(0.97, 0.97), xycoords="axes fraction",
                  ha="right", va="top", fontsize=7.5, color="#374151",
                  bbox=dict(boxstyle="round,pad=0.3",
                            fc="white", ec="#cbd5e1", lw=0.6))

    # ── Row 1, col 2: γ=0 vs γ=0.5 comparison ────────────────────────────────
    ax_g = fig.add_subplot(gs[1, 2])

    for topo in topos:
        gr  = gamma_results[topo]
        rho = gr["rho_vals"] * 1e3
        cc  = TOPO_COLOURS[topo]

        ax_g.fill_between(rho,
                          np.clip(gr["s0_smooth"] * 100 - gr["ci_gamma0"] * 100, 0, 100),
                          np.clip(gr["s0_smooth"] * 100 + gr["ci_gamma0"] * 100, 0, 100),
                          color=cc, alpha=0.08)
        ax_g.fill_between(rho,
                          np.clip(gr["s05_smooth"] * 100 - gr["ci_gamma05"] * 100, 0, 100),
                          np.clip(gr["s05_smooth"] * 100 + gr["ci_gamma05"] * 100, 0, 100),
                          color=cc, alpha=0.08)

        ax_g.plot(rho, gr["s0_smooth"] * 100,
                  color=cc, lw=1.5, linestyle="--",
                  label=f"{TOPO_LABELS[topo]}  γ=0")
        ax_g.plot(rho, gr["s05_smooth"] * 100,
                  color=cc, lw=2.0, linestyle="-",
                  label=f"{TOPO_LABELS[topo]}  γ={config.ALERT_GAMMA}")

    ax_g.axvline(rho_now * 1e3, color="#dc2626", lw=1.0,
                 linestyle="-.", alpha=0.8, label="Current ρ")
    ax_g.set_xlabel("Radar density  $\\rho$  [$\\times 10^{-3}$ u$^{-2}$]",
                    fontsize=9)
    ax_g.set_ylabel("$S(\\rho)$  [%]", fontsize=9)
    ax_g.set_title(f"Alert propagation contribution\n"
                   f"$\\gamma=0$ vs $\\gamma={config.ALERT_GAMMA}$ — ELINT Gaussian",
                   fontsize=9, fontweight="bold")
    ax_g.set_ylim(-2, 102)
    ax_g.set_xlim(rho_arr[0] * 1e3, rho_arr[-1] * 1e3)
    ax_g.legend(fontsize=7, framealpha=0.9, edgecolor="#cbd5e1", ncols=2)
    ax_g.grid(True, linewidth=0.3, alpha=0.4)

    seed_val = seed if seed is not None else config.RANDOM_SEED
    fig.suptitle(
        f"Stage 4 — Advanced analysis   "
        f"($N_{{\\mathrm{{runs}}}}={config.N_PHASE_RUNS}$, seed {seed_val})",
        fontsize=11, fontweight="500"
    )

    if save:
        path = os.path.join(config.FIGURES_DIR,
                            "stage4_advanced_analysis.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[Stage 4] Advanced figure saved → {path}")

    return fig


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(results):
    """Print ρ_c and k for all walk × topology combinations."""
    topos = ["ER", "BA", "WS"]
    cw    = 20
    sep   = "─" * (24 + cw * 3)

    print(f"\n{'':24}" +
          "".join(f"{TOPO_LABELS[t]:>{cw}}" for t in topos))
    print(sep)

    for wk, wc in WALKS.items():
        row_rc = f"  {wc['label'][:18]:<18}  ρ_c"
        row_k  = f"  {'':18}  k  "
        for t in topos:
            r   = results[wk][t]
            rc  = r["rho_c"] * 1e3
            nc  = round(r["rho_c"] * config.GRID_SIZE ** 2)
            kv  = r["k_fit"] if r["fit_ok"] else float("nan")
            row_rc += f"{rc:>{cw-6}.3f} (N={nc:>2d})  "
            row_k  += f"{kv:>{cw-2}.0f}  "
        print(row_rc)
        print(row_k)

    print(sep)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Stage 4 — phase transition analysis")
    print(f"  RHO_MIN={config.RHO_MIN}  RHO_MAX={config.RHO_MAX}  "
          f"RHO_STEPS={config.RHO_STEPS}  N_PHASE_RUNS={config.N_PHASE_RUNS}")

    n_sims = len(WALKS) * 3 * config.RHO_STEPS * config.N_PHASE_RUNS
    # gamma comparison adds 2 × 3 × RHO_STEPS × N_PHASE_RUNS
    n_gamma = 2 * 3 * config.RHO_STEPS * config.N_PHASE_RUNS
    print(f"  Main simulations:  {n_sims:>8,}")
    print(f"  Gamma comparison:  {n_gamma:>8,}")
    print(f"  Total:             {n_sims+n_gamma:>8,}")
    print(f"  Estimated time:    ~"
          f"{(n_sims+n_gamma)*config.MAX_STEPS/4e6:.0f}–"
          f"{(n_sims+n_gamma)*config.MAX_STEPS/2.5e6:.0f} min\n")

    t0 = time.time()

    # ── 1. Main sweep ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  MAIN SWEEP")
    print("=" * 60)
    results = run_phase_analysis(
        seed=config.RANDOM_SEED, n_runs=config.N_PHASE_RUNS
    )

    # ── 2. Gamma comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GAMMA COMPARISON  (γ=0 vs γ={})".format(config.ALERT_GAMMA))
    print("=" * 60)
    gamma_results = run_gamma_comparison(
        seed=config.RANDOM_SEED, n_runs=config.N_PHASE_RUNS
    )

    # ── 3. Detection heatmaps ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DETECTION HEATMAPS")
    print("=" * 60)
    heatmaps = compute_detection_heatmaps(
        results, n_heatmap=500, seed=config.RANDOM_SEED
    )

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} minutes.")

    print_summary_table(results)

    # ── 4. Figures ─────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_phase_transition(results, seed=config.RANDOM_SEED, save=True)
    plot_advanced_analysis(results, gamma_results, heatmaps,
                           seed=config.RANDOM_SEED, save=True)

    plt.show()
