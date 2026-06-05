"""
additional_analysis.py — Two formal analyses linking the group project
to the MSc course assignments (Project 3 and Project 5).

    Addition 1 — Lévy distribution verification
        Analogue of Project 3, Task 3.1.
        Output: stage_additional_levy_verification.png

    Addition 2 — Enriched network metrics (clustering + Pearson)
        Analogue of Project 5, Tasks 2–3.
        Output: stage_additional_network_metrics.png

Usage
-----
    python src/additional_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import config
from network_builder import build_network, network_metrics, TOPO_COLOURS, TOPO_LABELS

SEED       = config.RANDOM_SEED
OUT_DIR    = config.FIGURES_DIR
TOPOLOGIES = ["ER", "BA", "WS"]


# ══════════════════════════════════════════════════════════════════════════════
#  Addition 1 — Lévy distribution verification
# ══════════════════════════════════════════════════════════════════════════════

def _levy_samples(n, rng):
    """Inverse-transform sampling — identical to the simulation code."""
    mu    = config.LEVY_ALPHA
    s_min = config.LEVY_S_MIN
    s_max = config.LEVY_S_MAX
    u     = rng.uniform(size=n)
    return np.clip(s_min * (1.0 - u) ** (-1.0 / mu), s_min, s_max)


def _levy_pdf(l_vals):
    """Normalised P(l) = C · l^{-(μ+1)} on [s_min, s_max]."""
    mu    = config.LEVY_ALPHA
    s_min = config.LEVY_S_MIN
    s_max = config.LEVY_S_MAX
    norm  = (s_min ** (-mu) - s_max ** (-mu)) / mu
    return (1.0 / norm) * l_vals ** (-(mu + 1))


def plot_levy_verification(n_samples=1_000_000, seed=None, save=True):
    """
    Log-log comparison of empirical P(l) vs theoretical power law.

    Note: the uppermost histogram bin is excluded from the plot because
    samples clipped at l_max = config.LEVY_S_MAX pile up there artificially,
    making that bin unrepresentative of the true density.
    """
    rng   = np.random.default_rng(seed if seed is not None else SEED)
    mu    = config.LEVY_ALPHA
    s_min = config.LEVY_S_MIN
    s_max = config.LEVY_S_MAX

    samples = _levy_samples(n_samples, rng)

    # Histogram — exclude the last bin (clipping boundary artefact)
    n_bins          = 55
    bins            = np.logspace(np.log10(s_min), np.log10(s_max), n_bins + 1)
    counts, edges   = np.histogram(samples, bins=bins)
    widths          = np.diff(edges)
    mid             = 0.5 * (edges[:-1] + edges[1:])
    density         = counts / (n_samples * widths)

    # Drop the last bin (pile-up at l_max) and any empty bins
    interior        = (mid < s_max * 0.97) & (counts > 0)

    # Theoretical
    l_ref = np.logspace(np.log10(s_min), np.log10(s_max * 0.97), 400)
    p_ref = _levy_pdf(l_ref)

    # Goodness-of-fit on interior bins
    p_theo_mid  = _levy_pdf(mid[interior])
    rel_err     = np.abs(density[interior] - p_theo_mid) / p_theo_mid * 100
    mean_err    = rel_err.mean()
    max_err     = rel_err.max()

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]},
                             constrained_layout=True)

    # Left panel: main log-log plot
    ax = axes[0]
    ax.scatter(mid[interior], density[interior],
               s=30, color="#185FA5", alpha=0.80, zorder=4,
               label=f"Numerical  ($N = {n_samples:,}$ samples)")
    ax.plot(l_ref, p_ref,
            color="#993C1D", lw=2.2, zorder=5,
            label=f"Theory  $P(l) = C\\,l^{{-{mu+1:.1f}}}$  ($\\mu = {mu}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step length  $l$  [spatial units]", fontsize=10)
    ax.set_ylabel("Probability density  $P(l)$", fontsize=10)
    ax.set_title(
        f"Lévy step distribution: numerical verification\n"
        f"$\\mu = {mu}$,   "
        f"$l_{{\\min}} = {s_min}$,   "
        f"$l_{{\\max}} = {s_max}$",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9, framealpha=0.92, edgecolor="#cbd5e1",
              loc="lower left")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    # Slope annotation — placed in top-right, away from data
    ax.text(0.97, 0.95, f"slope $= -{mu + 1:.1f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#993C1D",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#f0957a", lw=0.8))

    # Right panel: relative error per bin (no boundary artefact)
    ax2 = axes[1]
    ax2.scatter(mid[interior], rel_err,
                s=22, color="#185FA5", alpha=0.75, zorder=4)
    ax2.axhline(0, color="#993C1D", lw=1.2, linestyle="--", zorder=5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Step length  $l$", fontsize=9)
    ax2.set_ylabel("Relative error  [%]", fontsize=9)
    ax2.set_title("Residuals\n(interior bins only)",
                  fontsize=9, fontweight="bold")
    ax2.grid(True, which="both", linewidth=0.3, alpha=0.4)

    # Stats in lower-right corner of residuals panel
    ax2.text(0.97, 0.04,
             f"Mean |err| = {mean_err:.2f}%\nMax |err| = {max_err:.2f}%",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="#cbd5e1", lw=0.7))

    fig.suptitle(
        f"Addition 1 — Lévy step distribution verification   "
        f"(seed {seed if seed is not None else SEED})",
        fontsize=11, fontweight="500"
    )

    if save:
        path = os.path.join(OUT_DIR, "stage_additional_levy_verification.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Addition 1] Saved → {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Addition 2 — Enriched network metrics (clustering + Pearson)
# ══════════════════════════════════════════════════════════════════════════════

def _enriched_metrics(G):
    m = network_metrics(G)
    m["pearson"] = nx.degree_assortativity_coefficient(G)
    return m


def plot_network_metrics(seed=None, n_radars=None, save=True):
    rng = np.random.default_rng(seed if seed is not None else SEED)
    N   = n_radars if n_radars is not None else config.N_RADARS_DEFAULT
    L, mg = config.GRID_SIZE, config.RADAR_RANGE * 0.5

    radar_positions = rng.uniform(mg, L - mg, size=(N, 2))
    graphs  = {t: build_network(radar_positions, t, rng=rng) for t in TOPOLOGIES}
    metrics = {t: _enriched_metrics(graphs[t])               for t in TOPOLOGIES}

    colours = [TOPO_COLOURS[t] for t in TOPOLOGIES]
    labels  = [TOPO_LABELS[t]  for t in TOPOLOGIES]
    x       = np.arange(len(TOPOLOGIES))

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=[1.0, 1.25])

    # ── Panel [0,0]: Clustering coefficient ──────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    c_vals = [metrics[t]["clustering"] for t in TOPOLOGIES]
    bars   = ax0.bar(x, c_vals, color=colours, alpha=0.82,
                     edgecolor="white", linewidth=0.7, width=0.55)

    # Y-limit with generous headroom for labels
    ax0.set_ylim(0, max(c_vals) * 1.45)
    for bar, val in zip(bars, c_vals):
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 val + max(c_vals) * 0.03,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    # ER theoretical reference line
    k_mean = metrics["ER"]["mean_degree"]
    c_er_th = k_mean / (N - 1)
    ax0.axhline(c_er_th, color=TOPO_COLOURS["ER"], lw=1.3,
                linestyle="--", alpha=0.75,
                label=f"ER theory  $C \\approx \\langle k\\rangle/N = {c_er_th:.3f}$")
    ax0.legend(fontsize=8, framealpha=0.92, edgecolor="#cbd5e1",
               loc="upper right")

    ax0.set_xticks(x); ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("Clustering coefficient  $C$", fontsize=9)
    ax0.set_title("Clustering coefficient\n(analogue: Project 5, Task 2)",
                  fontsize=9, fontweight="bold")
    ax0.grid(True, axis="y", linewidth=0.3, alpha=0.4)

    # Caption below x-axis (no overlap with bars)
    ax0.set_xlabel(
        "ER: $C \\approx c/N \\to 0$ as $N\\to\\infty$   "
        "WS: large $C$ (small-world)   BA: intermediate $C$",
        fontsize=7.5, color="#374151"
    )

    # ── Panel [0,1]: Pearson coefficient ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    rho_vals = [metrics[t]["pearson"] for t in TOPOLOGIES]
    bars     = ax1.bar(x, rho_vals, color=colours, alpha=0.82,
                       edgecolor="white", linewidth=0.7, width=0.55)

    rng_abs = max(abs(min(rho_vals)), abs(max(rho_vals)))
    ax1.set_ylim(-rng_abs * 1.5, rng_abs * 0.6)
    for bar, val in zip(bars, rho_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val - rng_abs * 0.06,
                 f"{val:.3f}", ha="center", va="top",
                 fontsize=9, fontweight="bold")

    ax1.axhline(0, color="#374151", lw=1.2, linestyle="--",
                alpha=0.7, label="$\\rho = 0$ (neutral / random wiring)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Pearson coefficient  $\\rho$", fontsize=9)
    ax1.set_title("Degree-degree correlation\n(analogue: Project 5, Task 3)",
                  fontsize=9, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.92, edgecolor="#cbd5e1",
               loc="upper right")
    ax1.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax1.set_xlabel(
        "$\\rho < 0$: disassortative — hubs connect to low-degree nodes",
        fontsize=7.5, color="#374151"
    )

    # ── Panel [0,2]: C vs ρ scatter ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for t in TOPOLOGIES:
        col = TOPO_COLOURS[t]
        rho = metrics[t]["pearson"]
        cc  = metrics[t]["clustering"]
        ax2.scatter(rho, cc, s=140, color=col, zorder=5,
                    edgecolors="white", linewidth=0.8,
                    label=TOPO_LABELS[t])

    # Annotate topology labels offset to avoid overlap with dots
    offsets = {"ER": (8, -14), "BA": (8, 6), "WS": (-60, 6)}
    for t in TOPOLOGIES:
        ax2.annotate(
            TOPO_LABELS[t],
            (metrics[t]["pearson"], metrics[t]["clustering"]),
            textcoords="offset points",
            xytext=offsets.get(t, (8, 4)),
            fontsize=8.5, color=TOPO_COLOURS[t], fontweight="bold"
        )

    ax2.axvline(0, color="#94a3b8", lw=0.8, linestyle=":", alpha=0.6)
    ax2.set_xlabel("Pearson  $\\rho$", fontsize=9)
    ax2.set_ylabel("Clustering  $C$", fontsize=9)
    ax2.set_title("Topology fingerprint\n($C$ vs $\\rho$)",
                  fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.92, edgecolor="#cbd5e1")
    ax2.grid(True, linewidth=0.3, alpha=0.4)

    # ── Row 1: Full metrics table ─────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")

    row_defs = [
        ("n_nodes",         "Nodes  N"),
        ("n_edges",         "Edges  |E|"),
        ("mean_degree",     "Mean degree  <k>"),
        ("degree_std",      "Degree std  σ_k"),
        ("max_degree",      "Max degree  k_max"),
        ("clustering",      "Clustering  C  ★"),
        ("pearson",         "Pearson  ρ  ★"),
        ("lcc_fraction",    "LCC fraction"),
        ("avg_path_length", "Avg path length"),
        ("diameter",        "Diameter"),
        ("is_connected",    "Connected"),
    ]

    def _fmt(key, val):
        if key in ("clustering", "pearson", "density"):
            return f"{val:.4f}"
        if key == "avg_path_length":
            return (f"{val:.3f}"
                    if isinstance(val, float) and not np.isnan(val)
                    else "n/a")
        if key == "is_connected":
            return "Yes" if val else "No"
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    col_labels = ["Metric"] + [TOPO_LABELS[t] for t in TOPOLOGIES]
    table_data = [[label] + [_fmt(k, metrics[t][k]) for t in TOPOLOGIES]
                  for k, label in row_defs]

    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    # Header
    header_cols = ["#1e293b"] + [TOPO_COLOURS[t] for t in TOPOLOGIES]
    for j, hc in enumerate(header_cols):
        cell = tbl[0, j]
        cell.set_facecolor(hc)
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight clustering and Pearson rows
    highlight = {i + 1 for i, (k, _) in enumerate(row_defs)
                 if k in ("clustering", "pearson")}
    for i in range(1, len(row_defs) + 1):
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            if i in highlight:
                cell.set_facecolor("#fef9c3" if j == 0 else "#fefce8")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#f1f5f9" if i % 2 == 0 else "white")

    ax_tbl.set_title(
        "Full metrics comparison — ER · BA · WS   "
        "(★ = new metrics, analogues of Project 5 Tasks 2–3)",
        fontsize=9.5, fontweight="bold", pad=8, loc="left"
    )

    fig.suptitle(
        f"Addition 2 — Enriched network metrics   "
        f"($N = {N}$ radars, seed {seed if seed is not None else SEED})",
        fontsize=11, fontweight="500"
    )

    if save:
        path = os.path.join(OUT_DIR, "stage_additional_network_metrics.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Addition 2] Saved → {path}")
    return fig, metrics


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("Running additional analyses...\n")

    print("  Addition 1 — Lévy distribution verification")
    print(f"    mu={config.LEVY_ALPHA},  "
          f"l_min={config.LEVY_S_MIN},  l_max={config.LEVY_S_MAX},  "
          f"N=1,000,000 samples")
    fig1 = plot_levy_verification(n_samples=1_000_000, seed=SEED, save=True)
    plt.close(fig1)

    print("\n  Addition 2 — Enriched network metrics (clustering + Pearson)")
    print(f"    N_radars={config.N_RADARS_DEFAULT},  "
          f"topologies: ER · BA · WS,  seed={SEED}")
    fig2, metrics = plot_network_metrics(
        seed=SEED, n_radars=config.N_RADARS_DEFAULT, save=True)

    print("\n  Summary:")
    print(f"  {'Topology':>18}  {'C':>8}  {'rho':>8}")
    for t in TOPOLOGIES:
        print(f"  {TOPO_LABELS[t]:>18}  "
              f"{metrics[t]['clustering']:>8.4f}  "
              f"{metrics[t]['pearson']:>8.4f}")
    plt.close(fig2)

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Figures saved to {OUT_DIR}/")
