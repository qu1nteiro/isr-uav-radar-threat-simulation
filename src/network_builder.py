"""
network_builder.py — Stage 2: radar network construction and analysis.

Builds three network topologies (Erdős–Rényi, Barabási–Albert, Watts–Strogatz)
on top of the physical radar positions defined in Stage 1. Computes standard
graph-theoretic metrics and robustness curves under random failure and
targeted hub attacks.

Public API (imported by Stages 3, 4, 5)
-----------------------------------------
    build_network(radar_positions, topology, rng)  →  nx.Graph
    network_metrics(G)                             →  dict
    robustness_curve(G, mode, rng)                 →  np.ndarray

Usage
-----
    python src/network_builder.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import os
import warnings

import config


# ─── Topology identifiers ─────────────────────────────────────────────────────

TOPOLOGIES = ("ER", "BA", "WS")

TOPO_LABELS = {
    "ER": "Erdős–Rényi",
    "BA": "Barabási–Albert",
    "WS": "Watts–Strogatz",
}

TOPO_COLOURS = {
    "ER": "#185FA5",   # blue
    "BA": "#993C1D",   # coral
    "WS": "#0F6E56",   # teal
}


# ─── Network construction ─────────────────────────────────────────────────────

def build_network(radar_positions, topology, rng=None, seed=None):
    """
    Build a radar communication network with the given topology.

    Each node i corresponds to radar i at position radar_positions[i].
    Physical positions are stored as node attribute 'pos' for plotting
    and as a numpy array for distance computations in later stages.

    Parameters
    ----------
    radar_positions : np.ndarray, shape (N, 2)
        Physical positions of radar nodes in the simulation domain.
    topology : str
        One of 'ER', 'BA', 'WS'.
    rng : np.random.Generator, optional
        Project RNG. Used to derive an integer seed for NetworkX.
    seed : int, optional
        Direct integer seed for NetworkX generators (overrides rng).

    Returns
    -------
    G : nx.Graph
        Graph with N nodes. Node attributes:
            'pos'   — (x, y) tuple  [spatial units]
            'topo'  — topology string
        Edge attributes:
            'weight' — 1.0 (uniform for now)
    """
    N = len(radar_positions)

    # Derive a reproducible integer seed for NetworkX
    if seed is None:
        if rng is not None:
            seed = int(rng.integers(0, 2**31))
        else:
            seed = config.RANDOM_SEED if config.RANDOM_SEED is not None else 0

    topology = topology.upper()
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, got '{topology}'")

    # ── Generate graph ────────────────────────────────────────────────────────
    if topology == "ER":
        G = nx.erdos_renyi_graph(N, config.ER_P, seed=seed)

    elif topology == "BA":
        G = nx.barabasi_albert_graph(N, config.BA_M, seed=seed)

    elif topology == "WS":
        G = nx.watts_strogatz_graph(N, config.WS_K, config.WS_BETA, seed=seed)

    # ── Attach physical positions and metadata ────────────────────────────────
    for i in range(N):
        G.nodes[i]["pos"]  = tuple(radar_positions[i])
        G.nodes[i]["topo"] = topology

    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    return G


def place_radars(n_radars=None, rng=None, seed=None):
    """
    Randomly place radar nodes in the simulation domain interior.

    Parameters
    ----------
    n_radars : int, optional
        Defaults to config.N_RADARS_DEFAULT.
    rng : np.random.Generator, optional
    seed : int, optional

    Returns
    -------
    radar_positions : np.ndarray, shape (N, 2)
    """
    n_radars = n_radars or config.N_RADARS_DEFAULT
    if rng is None:
        rng = np.random.default_rng(
            seed if seed is not None else config.RANDOM_SEED
        )
    margin = config.RADAR_RANGE * 0.6
    L      = config.GRID_SIZE
    return rng.uniform(margin, L - margin, size=(n_radars, 2))


# ─── Network metrics ──────────────────────────────────────────────────────────

def network_metrics(G):
    """
    Compute standard graph-theoretic metrics for a radar network.

    Metrics are computed on the largest connected component (LCC) where
    required (average path length, diameter), since these are undefined
    for disconnected graphs.

    Parameters
    ----------
    G : nx.Graph

    Returns
    -------
    metrics : dict with keys:
        n_nodes           — total nodes
        n_edges           — total edges
        density           — edge density = 2|E| / (N(N-1))
        mean_degree       — <k>
        degree_std        — std(k)
        max_degree        — max degree (hub size)
        clustering        — average clustering coefficient
        lcc_size          — size of largest connected component
        lcc_fraction      — lcc_size / N
        avg_path_length   — average shortest path length in LCC
        diameter          — diameter of LCC
        is_connected      — bool
    """
    N      = G.number_of_nodes()
    degrees = [d for _, d in G.degree()]

    lcc_nodes = max(nx.connected_components(G), key=len)
    lcc       = G.subgraph(lcc_nodes).copy()

    # Average path length — can be slow for large graphs; suppress warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            apl = nx.average_shortest_path_length(lcc)
            dia = nx.diameter(lcc)
        except nx.NetworkXError:
            apl = float("nan")
            dia = float("nan")

    return {
        "n_nodes":         N,
        "n_edges":         G.number_of_edges(),
        "density":         nx.density(G),
        "mean_degree":     float(np.mean(degrees)),
        "degree_std":      float(np.std(degrees)),
        "max_degree":      int(np.max(degrees)),
        "clustering":      nx.average_clustering(G),
        "lcc_size":        len(lcc_nodes),
        "lcc_fraction":    len(lcc_nodes) / N,
        "avg_path_length": apl,
        "diameter":        dia,
        "is_connected":    nx.is_connected(G),
    }


def print_metrics(metrics_dict):
    """Pretty-print a metrics comparison table for all three topologies."""
    keys = [
        ("n_nodes",         "Nodes"),
        ("n_edges",         "Edges"),
        ("density",         "Density"),
        ("mean_degree",     "Mean degree <k>"),
        ("degree_std",      "Degree std"),
        ("max_degree",      "Max degree (hub)"),
        ("clustering",      "Avg clustering"),
        ("lcc_fraction",    "LCC fraction"),
        ("avg_path_length", "Avg path length"),
        ("diameter",        "Diameter"),
        ("is_connected",    "Connected"),
    ]
    col_w = 22
    header = f"{'Metric':<{col_w}}" + "".join(
        f"{t:>{col_w}}" for t in metrics_dict
    )
    print("─" * (col_w + col_w * len(metrics_dict)))
    print(header)
    print("─" * (col_w + col_w * len(metrics_dict)))
    for key, label in keys:
        row = f"{label:<{col_w}}"
        for topo in metrics_dict:
            val = metrics_dict[topo].get(key, "—")
            if isinstance(val, float):
                row += f"{val:>{col_w}.3f}"
            elif isinstance(val, bool):
                row += f"{'Yes' if val else 'No':>{col_w}}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)
    print("─" * (col_w + col_w * len(metrics_dict)))


# ─── Robustness ───────────────────────────────────────────────────────────────

def robustness_curve(G, mode="random", rng=None):
    """
    Measure network robustness by sequentially removing nodes and tracking
    the size of the largest connected component (LCC).

    This directly implements the percolation-style robustness analysis:
    - Random removal models random radar failures (e.g. equipment fault)
    - Targeted removal models an adversarial attack on hub nodes

    Parameters
    ----------
    G : nx.Graph
    mode : str
        'random'   — nodes removed in random order
        'targeted' — nodes removed in descending degree order (hub-first)
    rng : np.random.Generator, optional

    Returns
    -------
    fractions_removed : np.ndarray, shape (N+1,)
        Fraction of nodes removed at each step (0 to 1).
    lcc_fractions : np.ndarray, shape (N+1,)
        LCC size / N at each step.
    """
    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)

    H = G.copy()
    N = G.number_of_nodes()

    if mode == "random":
        order = rng.permutation(list(H.nodes())).tolist()
    elif mode == "targeted":
        order = sorted(H.nodes(), key=lambda n: H.degree(n), reverse=True)
    else:
        raise ValueError(f"mode must be 'random' or 'targeted', got '{mode}'")

    fracs_removed = [0.0]
    lcc_fracs     = [len(max(nx.connected_components(H), key=len)) / N]

    for step, node in enumerate(order):
        H.remove_node(node)
        if H.number_of_nodes() == 0:
            lcc = 0
        else:
            lcc = len(max(nx.connected_components(H), key=len))
        fracs_removed.append((step + 1) / N)
        lcc_fracs.append(lcc / N)

    return np.array(fracs_removed), np.array(lcc_fracs)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _draw_single_network(ax, G, radar_positions, topo, metrics, L, show_labels=True):
    """Helper — draw one network topology on a given axes."""
    col = TOPO_COLOURS[topo]
    N   = len(radar_positions)

    # Coverage zones
    for rp in radar_positions:
        ring = plt.Circle(rp, config.RADAR_RANGE,
                          fill=True, facecolor="#dc262606",
                          edgecolor="#dc262618", linewidth=0.4, zorder=1)
        ax.add_patch(ring)

    # Edges
    for u, v in G.edges():
        x0, y0 = radar_positions[u]
        x1, y1 = radar_positions[v]
        ax.plot([x0, x1], [y0, y1],
                color=col, lw=0.7, alpha=0.5, zorder=2)

    # Nodes — size proportional to degree
    degrees = np.array([G.degree(i) for i in range(N)])
    sizes   = 12 + 6 * degrees
    ax.scatter(radar_positions[:, 0], radar_positions[:, 1],
               s=sizes, c=col, zorder=3,
               edgecolors="white", linewidths=0.5)

    # Start / target
    ax.scatter(*config.UAV_START,      s=55, c="#185FA5", marker="o",
               zorder=5, edgecolors="#fff", linewidths=0.9)
    ax.scatter(*config.MISSION_TARGET, s=65, c="#3B6D11", marker="*",
               zorder=5, edgecolors="#fff", linewidths=0.7)

    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$ [u]", fontsize=8)
    ax.set_ylabel("$x_2$ [u]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.25, alpha=0.35)

    m = metrics[topo]
    title_str = (
        TOPO_LABELS[topo] + "\n"
        + f"$\\langle k \\rangle={m['mean_degree']:.1f}$  "
        + f"$k_{{\\max}}={m['max_degree']}$  "
        + f"$C={m['clustering']:.2f}$"
    )
    ax.set_title(title_str, fontsize=8.5, fontweight="bold", color=col)


def plot_network_analysis(radar_positions=None, seed=None, save=True):
    """
    Produce the Stage 2 figure — 8 panels in a 2×4 grid:

        Row 0: ER layout | BA layout | WS layout | Combined layout
        Row 1: Degree ER+BA (log-log) | Degree WS (linear) |
               Robustness random | Robustness targeted
    """
    if seed is None:
        seed = config.RANDOM_SEED
    rng = np.random.default_rng(seed)

    if radar_positions is None:
        radar_positions = place_radars(rng=rng)

    N = len(radar_positions)

    # ── Build networks ────────────────────────────────────────────────────────
    graphs  = {t: build_network(radar_positions, t, rng=rng) for t in TOPOLOGIES}
    metrics = {t: network_metrics(graphs[t])                 for t in TOPOLOGIES}

    print("\n[Stage 2] Network metrics:")
    print_metrics(metrics)

    # ── Figure: 2 rows × 4 cols ───────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 4, figure=fig)

    L = config.GRID_SIZE

    # ── Row 0: individual network layouts ─────────────────────────────────────
    for col_idx, topo in enumerate(TOPOLOGIES):
        ax = fig.add_subplot(gs[0, col_idx])
        _draw_single_network(ax, graphs[topo], radar_positions,
                             topo, metrics, L)

    # Combined overlay (col 3)
    ax_comb = fig.add_subplot(gs[0, 3])
    offsets = {"ER": (-0.5, 0.5), "BA": (0.0, 0.0), "WS": (0.5, -0.5)}

    for rp in radar_positions:
        ring = plt.Circle(rp, config.RADAR_RANGE,
                          fill=True, facecolor="#dc262605",
                          edgecolor="#dc262615", linewidth=0.4, zorder=1)
        ax_comb.add_patch(ring)

    for topo, G in graphs.items():
        col    = TOPO_COLOURS[topo]
        ox, oy = offsets[topo]
        for u, v in G.edges():
            x0, y0 = radar_positions[u][0] + ox, radar_positions[u][1] + oy
            x1, y1 = radar_positions[v][0] + ox, radar_positions[v][1] + oy
            ax_comb.plot([x0, x1], [y0, y1],
                         color=col, lw=0.5, alpha=0.4, zorder=2)
        ax_comb.scatter(radar_positions[:, 0] + ox,
                        radar_positions[:, 1] + oy,
                        s=14, c=col, zorder=3,
                        edgecolors="white", linewidths=0.3,
                        label=TOPO_LABELS[topo])

    ax_comb.scatter(*config.UAV_START,      s=55, c="#185FA5",
                    marker="o", zorder=5, edgecolors="#fff", linewidths=0.9)
    ax_comb.scatter(*config.MISSION_TARGET, s=65, c="#3B6D11",
                    marker="*", zorder=5, edgecolors="#fff", linewidths=0.7)

    ax_comb.set_xlim(0, L); ax_comb.set_ylim(0, L)
    ax_comb.set_aspect("equal")
    ax_comb.set_xlabel("$x_1$ [u]", fontsize=8)
    ax_comb.set_ylabel("$x_2$ [u]", fontsize=8)
    ax_comb.tick_params(labelsize=7)
    ax_comb.grid(True, linewidth=0.25, alpha=0.35)
    ax_comb.set_title("All topologies — combined", fontsize=8.5,
                       fontweight="bold", color="#374151")
    ax_comb.legend(fontsize=7, loc="lower right",
                   framealpha=0.9, edgecolor="#cbd5e1")

    # ── Row 1, col 0: Degree distribution ER + BA (log-log) ───────────────────
    ax_deg1 = fig.add_subplot(gs[1, 0])

    for topo in ("ER", "BA"):
        G       = graphs[topo]
        col     = TOPO_COLOURS[topo]
        degrees = [d for _, d in G.degree()]
        k_max   = max(degrees)
        counts  = np.bincount(degrees, minlength=k_max + 2)[1:]
        prob    = counts / counts.sum()
        k_vals  = np.arange(1, k_max + 2)
        mask    = prob > 0
        ax_deg1.plot(k_vals[mask], prob[mask],
                     "o-", color=col, lw=1.3, ms=4.5,
                     label=TOPO_LABELS[topo], alpha=0.88)

    ax_deg1.set_xscale("log")
    ax_deg1.set_yscale("log")
    ax_deg1.set_xlabel("Degree  $k$", fontsize=9)
    ax_deg1.set_ylabel("$P(k)$", fontsize=9)
    ax_deg1.set_title("Degree distribution — ER & BA\n(log-log scale)",
                       fontsize=8.5, fontweight="bold")
    ax_deg1.legend(fontsize=8, framealpha=0.9, edgecolor="#cbd5e1")
    ax_deg1.grid(True, linewidth=0.3, alpha=0.4, which="both")
    ax_deg1.annotate("BA: power-law tail\n$P(k) \\propto k^{-3}$",
                     xy=(0.55, 0.15), xycoords="axes fraction",
                     fontsize=7.5, color=TOPO_COLOURS["BA"],
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", ec="#f0957a", lw=0.6))

    # ── Row 1, col 1: Degree distribution WS (linear bar) ─────────────────────
    ax_deg2 = fig.add_subplot(gs[1, 1])

    G_ws    = graphs["WS"]
    col_ws  = TOPO_COLOURS["WS"]
    deg_ws  = [d for _, d in G_ws.degree()]
    k_vals_ws, counts_ws = np.unique(deg_ws, return_counts=True)
    prob_ws = counts_ws / counts_ws.sum()

    ax_deg2.bar(k_vals_ws, prob_ws,
                color=col_ws, alpha=0.78,
                edgecolor="white", linewidth=0.6, width=0.55)
    ax_deg2.axvline(np.mean(deg_ws), color="#374151", lw=1.2,
                    linestyle="--",
                    label=f"$\\langle k \\rangle = {np.mean(deg_ws):.1f}$")

    ax_deg2.set_xlabel("Degree  $k$", fontsize=9)
    ax_deg2.set_ylabel("$P(k)$", fontsize=9)
    ax_deg2.set_title("Degree distribution — WS\n(linear scale)",
                       fontsize=8.5, fontweight="bold",
                       color=col_ws)
    ax_deg2.legend(fontsize=8, framealpha=0.9, edgecolor="#cbd5e1")
    ax_deg2.grid(True, linewidth=0.3, alpha=0.4, axis="y")
    ax_deg2.set_xticks(k_vals_ws)

    ws_std = np.std(deg_ws)
    ax_deg2.annotate(
        f"Narrow distribution\n$\\sigma_k = {ws_std:.2f}$\n(small-world property)",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=7.5, color="#374151",
        bbox=dict(boxstyle="round,pad=0.3",
                  fc="white", ec="#a7f3d0", lw=0.6))

    # ── Row 1, cols 2-3: Robustness curves ────────────────────────────────────
    for col_idx, (mode, title) in enumerate([
        ("random",   "Robustness — random node failure"),
        ("targeted", "Robustness — targeted hub attack"),
    ]):
        ax = fig.add_subplot(gs[1, col_idx + 2])
        for topo, G in graphs.items():
            c = TOPO_COLOURS[topo]
            fr, lcc = robustness_curve(G, mode=mode, rng=rng)
            ax.plot(fr * 100, lcc * 100,
                    color=c, lw=1.7, label=TOPO_LABELS[topo])
            idx50 = np.argmax(lcc <= 0.5)
            if 0 < idx50 < len(fr):
                ax.axvline(fr[idx50] * 100, color=c,
                           lw=0.7, linestyle=":", alpha=0.55)

        ax.axhline(50, color="#94a3b8", lw=0.8,
                   linestyle="--", alpha=0.6, label="50% LCC threshold")
        ax.set_xlabel("Fraction of nodes removed  [%]", fontsize=9)
        ax.set_ylabel("Largest connected component  [%]", fontsize=9)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.set_xlim(0, 100); ax.set_ylim(0, 105)
        ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cbd5e1")
        ax.grid(True, linewidth=0.3, alpha=0.4)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Stage 2 — Radar network analysis   ($N={N}$, seed {seed})",
        fontsize=12, fontweight="500"
    )

    if save:
        path = os.path.join(config.FIGURES_DIR, "stage2_network_analysis.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[Stage 2] Figure saved → {path}")

    return fig, graphs, metrics


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_networks(graphs, metrics, verbose=True):
    """
    Sanity checks for all three networks.

    Checks:
    - All graphs have N nodes
    - BA is always connected (guaranteed by construction)
    - Mean degree is in a physically plausible range
    - No self-loops

    Returns True if all pass.
    """
    N      = config.N_RADARS_DEFAULT
    checks = {}

    for topo, G in graphs.items():
        m = metrics[topo]
        checks[f"{topo}: correct node count"]  = (G.number_of_nodes() == N)
        checks[f"{topo}: no self-loops"]       = (nx.number_of_selfloops(G) == 0)
        checks[f"{topo}: mean degree > 0"]     = (m["mean_degree"] > 0)
        checks[f"{topo}: LCC fraction > 0.5"]  = (m["lcc_fraction"] > 0.5)

    checks["BA: always connected"] = metrics["BA"]["is_connected"]

    all_pass = all(checks.values())

    if verbose:
        print("\n─── Network validation ───────────────────────────────")
        for label, passed in checks.items():
            print(f"  {'✓' if passed else '✗'}  {label}")
        print(f"──────────────────────────────────────────────────────")

    return all_pass


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Stage 2 — radar network construction and analysis\n")

    rng             = np.random.default_rng(config.RANDOM_SEED)
    radar_positions = place_radars(rng=rng)

    fig, graphs, metrics = plot_network_analysis(
        radar_positions=radar_positions,
        seed=config.RANDOM_SEED,
        save=True
    )

    ok = validate_networks(graphs, metrics, verbose=True)
    if not ok:
        print("\n[WARNING] One or more validation checks failed.")
    else:
        print("\n[OK] All validation checks passed.")

    plt.show()
