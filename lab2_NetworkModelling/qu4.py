# TMM Lab 02 - Q4*: beta-skeletons on a torus (doubly periodic tile), after
# Espinosa, Connors & Wilson. No boundary => no edge effects. Load with uniform
# all-to-all demand on shortest routes (ties split equally) and study how the
# metrics vary with beta, for (i) uniformly random and (ii) jittered-grid nodes.
#
# Torus conventions used throughout:
#   - nodes live in the tile [0, W)^2; node k has 'pos'
#   - each edge (u, v) with u < v stores 'vec' = displacement from u to the nearest
#     periodic image of v used by that street, and 'weight' = |vec|
#   - straight-line distance between nodes is the minimum-image distance

# %% Imports
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import shortest_path

from qu1 import edge_flows, uniform_demand
from qu3 import random_nodes, jittered_grid, blocked

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# %% Torus geometry
def tile_3x3(pts, W):
    """The point set plus its 8 periodic copies, (9N, 2). Copy 4 is the original."""
    offsets = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]) * W
    return np.vstack([pts + o for o in offsets])


def min_image(d, W):
    """Wrap displacement(s) d into [-W/2, W/2) per axis (nearest periodic image)."""
    return d - W * np.round(d / W)


def torus_distance_matrix(pts, W):
    """N x N straight-line (minimum-image) distances on the torus."""
    d = min_image(pts[:, None, :] - pts[None, :, :], W)
    return np.sqrt((d**2).sum(axis=2))


def empty_torus_graph(pts):
    G = nx.Graph()
    for k, xy in enumerate(pts):
        G.add_node(k, pos=tuple(xy))
    return G


def add_torus_edge(G, i, j, vec):
    """Add street i -> (image of) j with displacement vec, stored with u < v."""
    u, v, vec = (i, j, vec) if i < j else (j, i, -vec)
    G.add_edge(u, v, weight=float(np.linalg.norm(vec)), vec=vec)


# %% Torus beta-skeleton
def torus_skeleton_delaunay(pts, beta, W, kind="lune"):
    """beta >= 1: candidates = periodic Delaunay edges (Delaunay of the 3x3 tiling, keeping
    edges with an endpoint in the central copy). Forbidden region tested against ALL tiled
    points, so blockers just across the 'seam' count too."""
    N = len(pts)
    tiled = tile_3x3(pts, W)
    G = empty_torus_graph(pts)
    s = Delaunay(tiled).simplices
    cand = np.unique(np.sort(np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]]), axis=1), axis=0)
    for a, b in cand:
        if not 4 * N <= a < 5 * N:
            a, b = b, a                        # make a the central-copy endpoint
            if not 4 * N <= a < 5 * N:
                continue
        i, j = a % N, b % N
        if i == j or G.has_edge(i, j):         # self-image, or seen from the other end
            continue
        if not blocked(tiled, a, b, beta, kind):
            add_torus_edge(G, i, j, tiled[b] - tiled[a])
    return G


def torus_skeleton_bruteforce(pts, beta, W):
    """beta <= 1 (lune and circle coincide): keep (p, q) iff no r has angle prq > theta,
    theta = pi - arcsin(beta)  <=>  cos(prq) < -sqrt(1 - beta^2).
    Not a Delaunay subgraph for beta < 1, so test all pairs (nearest image, |pq| < W/2).
    At beta = 1 this is an independent Gabriel construction (used for verification)."""
    N = len(pts)
    tiled = tile_3x3(pts, W)
    cos_thr = -np.sqrt(max(0.0, 1 - beta**2))
    G = empty_torus_graph(pts)
    for i in range(N):
        for j in range(i + 1, N):
            vec = min_image(pts[j] - pts[i], W)
            d = np.linalg.norm(vec)
            if d >= W / 2:
                continue
            p, q = pts[i], pts[i] + vec
            # the region lies inside the disc on diameter pq: only test points in there
            near = np.linalg.norm(tiled - (p + q) / 2, axis=1) < d / 2
            r = tiled[near]
            a, b = p - r, q - r
            na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            ok = (na > 1e-12) & (nb > 1e-12)        # exclude p and q themselves
            cosang = np.einsum("kd,kd->k", a[ok], b[ok]) / (na[ok] * nb[ok])
            if not (cosang < cos_thr).any():
                add_torus_edge(G, i, j, vec)
    return G


def torus_beta_skeleton(pts, beta, W, kind="lune"):
    """Lune- (or circle-) based beta-skeleton on the torus of side W."""
    if beta >= 1:
        return torus_skeleton_delaunay(pts, beta, W, kind)
    return torus_skeleton_bruteforce(pts, beta, W)


# %% Verification
def edge_set(G):
    return {tuple(sorted(e)) for e in G.edges}


def torus_rng_bruteforce(pts, W):
    """Periodic relative neighbourhood graph: no r with max(|rp|, |rq|) < |pq| (min-image)."""
    D = torus_distance_matrix(pts, W)
    N, E = len(pts), set()
    for i in range(N):
        for j in range(i + 1, N):
            dmax = np.maximum(D[i], D[j])
            dmax[[i, j]] = np.inf
            if not (dmax < D[i, j]).any():
                E.add((i, j))
    return E


def verify_torus_skeleton(N=120, W=1.0, seed=7):
    pts = random_nodes(N, side=W, rng=seed)

    # 1. beta = 1: Delaunay-based construction == brute-force angle construction (Gabriel)
    assert edge_set(torus_skeleton_delaunay(pts, 1.0, W)) == edge_set(torus_skeleton_bruteforce(pts, 1.0, W))
    # 2. beta = 2 (lune) == brute-force periodic RNG
    assert edge_set(torus_beta_skeleton(pts, 2.0, W)) == torus_rng_bruteforce(pts, W)
    # 3. translation invariance: shifting every node (mod W) cannot change a torus network
    shifted = (pts + np.array([0.37, 0.81]) * W) % W
    for b in (0.8, 1.0, 1.5, 2.5):
        assert edge_set(torus_beta_skeleton(pts, b, W)) == edge_set(torus_beta_skeleton(shifted, b, W))
    # 4. nesting: larger beta => subset of edges
    sets = [edge_set(torus_beta_skeleton(pts, b, W)) for b in (0.7, 0.9, 1.0, 1.3, 2.0, 3.0)]
    assert all(s2 <= s1 for s1, s2 in zip(sets, sets[1:]))
    print(f"verification passed (N = {N}): Gabriel {len(sets[2])} edges, "
          f"RNG {len(sets[4])} edges, 3N = {3 * N}")


# %% Metrics for one torus network
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * x) / (n * x.sum())


def graph_distance_matrix(G):
    """All-pairs shortest-path lengths via scipy (C implementation - much faster than networkx)."""
    A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), weight="weight")
    return shortest_path(A, directed=False)


def torus_metrics(G, W, L_mst):
    N = G.number_of_nodes()
    pts = np.array([G.nodes[k]["pos"] for k in range(N)])
    euclid = torus_distance_matrix(pts, W)
    dG = graph_distance_matrix(G)
    off = ~np.eye(N, dtype=bool)
    flows = edge_flows(G, uniform_demand(N))
    f = np.array(list(flows.values()))
    # sanity: flow-weighted street length == total of all trip lengths
    assert np.isclose(sum(fl * G.edges[e]["weight"] for e, fl in flows.items()), dG[off].sum())
    length = G.size(weight="weight")
    return dict(
        mean_degree=2 * G.number_of_edges() / N,
        length=length,
        length_rel=length / L_mst,                  # street length relative to the MST
        mean_d=dG[off].mean(),                      # mean trip length
        mean_detour=(dG[off] / euclid[off]).mean(), # mean route factor
        f_max_frac=f.max() / (N * (N - 1)),         # share of ALL trips using the busiest street
        gini=gini(f),
    )


# %% Sweeps
def run_torus_sweep(point_sets, betas, W):
    """Every layout x every beta. Connectivity is recorded for all; the other
    metrics only for connected networks (shortest paths undefined otherwise)."""
    rows = []
    for seed, pts in enumerate(point_sets):
        N = len(pts)
        L_mst = nx.minimum_spanning_tree(torus_beta_skeleton(pts, 1.0, W)).size(weight="weight")
        for b in betas:
            G = torus_beta_skeleton(pts, b, W)
            giant = max(len(c) for c in nx.connected_components(G)) / N
            row = dict(beta=b, seed=seed, connected=giant == 1.0, giant_frac=giant,
                       mean_degree=2 * G.number_of_edges() / N)
            if row["connected"]:
                row.update(torus_metrics(G, W, L_mst))
            rows.append(row)
        print(f"  layout {seed + 1}/{len(point_sets)} done")
    return rows


def sweep_torus_random_nodes(betas, N=196, W=1.4, n_seeds=5):
    """Option (i): N uniformly random nodes on a W x W torus."""
    point_sets = [random_nodes(N, side=W, rng=seed) for seed in range(n_seeds)]
    return run_torus_sweep(point_sets, betas, W)


def sweep_torus_jittered_grid(betas, n=14, L=0.1, sigma=0.2, n_seeds=5):
    """Option (ii): n x n grid of pitch L, jittered, wrapped onto a torus of side W = nL
    (the grid tiles the torus exactly, so spacing is uniform across the seam)."""
    W = n * L
    point_sets = [jittered_grid(n, L=L, sigma=sigma, rng=seed) % W for seed in range(n_seeds)]
    return run_torus_sweep(point_sets, betas, W)


# %% Summaries and plots
def summarise(rows, betas, key, connected_only=True):
    """Mean and std over layouts at each beta (NaN where no layout qualifies)."""
    mean, std = [], []
    for b in betas:
        v = [r[key] for r in rows if r["beta"] == b and (r["connected"] or not connected_only)]
        mean.append(np.mean(v) if v else np.nan)
        std.append(np.std(v) if v else np.nan)
    return np.array(mean), np.array(std)


def plot_torus_metrics_vs_beta(results, betas):
    panels = [("mean_degree", "mean degree", False),
              ("length_rel", r"street length  $\Lambda/\Lambda_{MST}$", True),
              ("mean_detour", r"mean detour  $\langle d_G/d_E\rangle$", True),
              ("f_max_frac", "busiest street: share of all trips", True),
              ("gini", "Gini of edge flows", True),
              ("giant_frac", "largest component / N", False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, label, conn) in zip(axes.flat, panels):
        for name, rows in results.items():
            mean, sd = summarise(rows, betas, key, connected_only=conn)
            ax.plot(betas, mean, "o-", ms=3, label=name)
            ax.fill_between(betas, mean - sd, mean + sd, alpha=0.25)
        for b0 in (1.0, 2.0):                    # Gabriel and RNG landmarks
            ax.axvline(b0, color="grey", lw=0.7, ls=":")
        ax.set(xlabel=r"$\beta$", ylabel=label)
        if key in ("mean_degree", "length_rel"):
            ax.set_yscale("log")                 # beta < 1 networks are much denser
            ax.set_yticks([1.5, 2, 3, 5, 10, 15])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend()
    fig.suptitle("Q4: lune beta-skeletons on a torus, uniform all-to-all demand "
                 "(mean $\\pm$ 1 SD; flow/length metrics over connected layouts only)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Q4_metrics_vs_beta.png"), dpi=110, bbox_inches="tight")


def plot_torus_tradeoff(results, betas):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k, (name, rows) in enumerate(results.items()):
        x, _ = summarise(rows, betas, "length_rel")
        for ax, key in zip(axes, ["mean_detour", "f_max_frac"]):
            y, _ = summarise(rows, betas, key)
            ax.plot(x, y, "-", color=f"C{k}", lw=0.8)
            sc = ax.scatter(x, y, c=betas, cmap="viridis", marker="o^"[k], s=40,
                            label=name, zorder=3, vmin=betas.min(), vmax=betas.max())
    for ax, lab in zip(axes, ["mean detour", "busiest street: share of all trips"]):
        ax.set(xlabel=r"street length $\Lambda/\Lambda_{MST}$ (cost, log scale)", ylabel=lab)
        ax.set_xscale("log")
        ax.set_xticks([1.5, 2, 3, 5, 10, 15])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.colorbar(sc, ax=axes, label=r"$\beta$")
    fig.suptitle("Q4 trade-off on the torus: cost vs detour, cost vs congestion")
    fig.savefig(os.path.join(FIG_DIR, "Q4_tradeoff.png"), dpi=110, bbox_inches="tight")


def draw_torus_flows(ax, G, flows, W, vmax, title):
    """Hotness map on the periodic tile; each street drawn from both ends, clipped to the tile."""
    pos = nx.get_node_attributes(G, "pos")
    segs, f = [], []
    for (u, v), fl in flows.items():
        vec = G.edges[u, v]["vec"]
        pu, pv = np.array(pos[u]), np.array(pos[v])
        segs += [(pu, pu + vec), (pv, pv - vec)]
        f += [fl, fl]
    f = np.array(f)
    lc = LineCollection(segs, array=f, cmap="hot_r", norm=plt.Normalize(0, vmax),
                        linewidths=0.4 + 4 * f / vmax)
    ax.add_collection(lc)
    ax.set_xlim(0, W); ax.set_ylim(0, W); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return lc


def plot_torus_flow_maps(pts, W, betas_to_show, label):
    """Flow maps for one layout at several beta, on ONE shared colour scale."""
    Gs = [torus_beta_skeleton(pts, b, W) for b in betas_to_show]
    Fs = [edge_flows(G, uniform_demand(len(pts))) for G in Gs]
    vmax = max(max(F.values()) for F in Fs)
    fig, axes = plt.subplots(1, len(Gs), figsize=(4.2 * len(Gs), 4.6))
    for ax, b, G, F in zip(axes, betas_to_show, Gs, Fs):
        lc = draw_torus_flows(ax, G, F, W, vmax,
                              f"beta = {b}: {G.number_of_edges()} streets, max {max(F.values()):.0f}")
    fig.colorbar(lc, ax=axes, shrink=0.8, label="edge flow (trips)")
    fig.suptitle(f"Q4 flow maps on the torus ({label}), shared colour scale")
    fig.savefig(os.path.join(FIG_DIR, f"Q4_flow_maps_{label.replace(' ', '_')}.png"),
                dpi=110, bbox_inches="tight")


def print_torus_table(name, rows, betas):
    print(f"\n{name}")
    print(f"{'beta':>5} {'conn':>5} {'giant':>6} {'deg':>5} {'L/Lmst':>7} {'detour':>7} "
          f"{'fmax%':>6} {'gini':>6}")
    for b in betas:
        conn = np.mean([r["connected"] for r in rows if r["beta"] == b])
        vals = [summarise(rows, [b], k, c)[0][0] for k, c in
                [("giant_frac", False), ("mean_degree", False), ("length_rel", True),
                 ("mean_detour", True), ("f_max_frac", True), ("gini", True)]]
        print(f"{b:5.2f} {conn:5.0%} {vals[0]:6.3f} {vals[1]:5.2f} {vals[2]:7.3f} "
              f"{vals[3]:7.3f} {100 * vals[4]:6.2f} {vals[5]:6.3f}")


# %% Run
if __name__ == "__main__":
    verify_torus_skeleton()

    betas = np.round(np.concatenate([np.arange(0.7, 2.0, 0.1), np.arange(2.0, 3.01, 0.2)]), 2)

    print("\nTorus sweep (i): uniformly random nodes")
    random_rows = sweep_torus_random_nodes(betas, N=196, W=1.4, n_seeds=5)

    print("\nTorus sweep (ii): jittered grid")
    grid_rows = sweep_torus_jittered_grid(betas, n=14, L=0.1, sigma=0.2, n_seeds=5)

    print_torus_table("Torus (i): uniformly random nodes", random_rows, betas)
    print_torus_table("Torus (ii): jittered grid, sigma = 0.2 L", grid_rows, betas)

    results = {"random nodes": random_rows, "jittered grid": grid_rows}
    plot_torus_metrics_vs_beta(results, betas)
    plot_torus_tradeoff(results, betas)
    plot_torus_flow_maps(random_nodes(196, side=1.4, rng=0), 1.4, [0.8, 1.0, 1.5, 2.0], "random nodes")
    plot_torus_flow_maps(jittered_grid(14, sigma=0.2, rng=0) % 1.4, 1.4, [0.8, 1.0, 1.5, 2.0], "jittered grid")
    print(f"\nfigures saved to {FIG_DIR}")
    plt.show()
