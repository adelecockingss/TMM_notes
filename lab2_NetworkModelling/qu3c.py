# TMM Lab 02 - Q3(c): how street length trades off against path length and congestion
# as beta varies, for two ways of placing the nodes:
#   (i)  uniformly random nodes          -> sweep_random_nodes
#   (ii) perturbations of a uniform grid -> sweep_jittered_grid
# Lune-based skeletons on 1 <= beta <= 2 (guaranteed connected), uniform all-to-all demand.

# %% Imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from qu1 import edge_flows, uniform_demand
from qu3 import random_nodes, jittered_grid, beta_skeleton, delaunay_edges


# %% Metrics for one network
def gini(x):
    """Gini coefficient of non-negative values: 0 = all equal, -> 1 = all on one item."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * x) / (n * x.sum())


def shortest_path_matrix(G, weight="weight"):
    """N x N matrix of shortest-path lengths (nodes must be 0..N-1)."""
    N = G.number_of_nodes()
    dG = np.full((N, N), np.inf)
    for s, lengths in nx.all_pairs_dijkstra_path_length(G, weight=weight):
        for t, d in lengths.items():
            dG[s, t] = d
    return dG


def network_metrics(G, D):
    """Cost (length), performance (path length, detour) and congestion (flow) metrics."""
    N = G.number_of_nodes()
    pts = np.array([G.nodes[k]["pos"] for k in range(N)])
    euclid = squareform(pdist(pts))
    dG = shortest_path_matrix(G)
    off = ~np.eye(N, dtype=bool)                     # drop s == t pairs
    flows = np.array(list(edge_flows(G, D).values()))
    return dict(
        edges=G.number_of_edges(),
        length=G.size(weight="weight"),             # total street length
        mean_d=dG[off].mean(),                      # mean shortest-path length
        mean_detour=(dG[off] / euclid[off]).mean(), # mean route factor d_G / |x_s - x_t|
        f_max=flows.max(),
        f_mean=flows.mean(),
        gini=gini(flows),
    )


# %% Reference networks: Delaunay (densest planar) and MST (cheapest connected)
def delaunay_graph(pts):
    G = nx.Graph()
    for k, xy in enumerate(pts):
        G.add_node(k, pos=tuple(xy))
    for i, j in delaunay_edges(pts):
        G.add_edge(int(i), int(j), weight=float(np.linalg.norm(pts[i] - pts[j])))
    return G


def mst_graph(pts):
    return nx.minimum_spanning_tree(delaunay_graph(pts), weight="weight")


# %% Core sweep (shared by both node-placement methods)
def run_sweep(point_sets, betas):
    """For each node layout in point_sets, build the lune skeleton at every beta and
    record metrics. Also records Delaunay and MST for each layout as references.
    Returns (rows, refs): lists of dicts."""
    rows, refs = [], []
    for seed, pts in enumerate(point_sets):
        N = len(pts)
        D = uniform_demand(N)
        L_mst = mst_graph(pts).size(weight="weight")   # normaliser for street length

        for name, G in [("Delaunay", delaunay_graph(pts)), ("MST", mst_graph(pts))]:
            m = network_metrics(G, D)
            m.update(network=name, seed=seed, length_rel=m["length"] / L_mst)
            refs.append(m)

        for b in betas:
            G = beta_skeleton(pts, b, "lune")
            if not nx.is_connected(G):                 # metrics meaningless if split
                print(f"  seed {seed}, beta {b}: disconnected - skipped")
                continue
            m = network_metrics(G, D)
            m.update(beta=b, seed=seed, length_rel=m["length"] / L_mst)
            rows.append(m)
        print(f"  layout {seed + 1}/{len(point_sets)} done")
    return rows, refs


# %% Method (i): uniformly random nodes
def sweep_random_nodes(betas, N=196, side=1.4, n_seeds=5):
    """Q3 option (i): N nodes uniform in a side x side square, n_seeds layouts."""
    point_sets = [random_nodes(N, side=side, rng=seed) for seed in range(n_seeds)]
    return run_sweep(point_sets, betas)


# %% Method (ii): perturbed uniform grid
def sweep_jittered_grid(betas, n=14, L=0.1, sigma=0.2, n_seeds=5):
    """Q3 option (ii): n x n grid of pitch L, each node jittered by N(0, (sigma*L)^2)."""
    point_sets = [jittered_grid(n, L=L, sigma=sigma, rng=seed) for seed in range(n_seeds)]
    return run_sweep(point_sets, betas)


# %% Summaries and plots
def summarise(rows, betas, key):
    """Mean and std over seeds of metric `key` at each beta."""
    mean = np.array([np.mean([r[key] for r in rows if r["beta"] == b]) for b in betas])
    std = np.array([np.std([r[key] for r in rows if r["beta"] == b]) for b in betas])
    return mean, std


def ref_mean(refs, network, key):
    return np.mean([r[key] for r in refs if r["network"] == network])


def plot_metrics_vs_beta(results, betas):
    """2x2 panel of metrics against beta; results = {label: (rows, refs)}."""
    panels = [("length_rel", r"street length  $\Lambda/\Lambda_{MST}$"),
              ("mean_detour", r"mean detour  $\langle d_G / d_E \rangle$"),
              ("f_max", r"max edge flow  $f_{max}$"),
              ("gini", "Gini of edge flows")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (key, label) in zip(axes.flat, panels):
        for name, (rows, _) in results.items():
            mean, std = summarise(rows, betas, key)
            ax.plot(betas, mean, "o-", ms=3, label=name)
            ax.fill_between(betas, mean - std, mean + std, alpha=0.25)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Lune beta-skeletons, uniform all-to-all demand (mean $\\pm$ 1 sd over layouts)")
    fig.tight_layout()


def plot_tradeoff(results, betas, ykey="mean_detour", ylabel="mean detour"):
    """Cost vs performance: x = street length / MST length, y = chosen metric, colour = beta."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    markers = ["o", "^"]
    for (name, (rows, refs)), mk in zip(results.items(), markers):
        x, _ = summarise(rows, betas, "length_rel")
        y, _ = summarise(rows, betas, ykey)
        ax.plot(x, y, "-", color="grey", lw=0.8)
        sc = ax.scatter(x, y, c=betas, cmap="viridis", marker=mk, s=40, label=name, zorder=3)
        for ref, rmk in [("Delaunay", "s"), ("MST", "X")]:
            ax.scatter(ref_mean(refs, ref, "length_rel"), ref_mean(refs, ref, ykey),
                       marker=rmk, s=90, facecolor="none", edgecolor="k", zorder=4)
            ax.annotate(f"{ref} ({name.split()[0]})",
                        (ref_mean(refs, ref, "length_rel"), ref_mean(refs, ref, ykey)),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.colorbar(sc, ax=ax, label=r"$\beta$")
    ax.set_xlabel(r"street length  $\Lambda/\Lambda_{MST}$  (cost)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"Trade-off: street length vs {ylabel}")
    fig.tight_layout()


def print_table(name, rows, refs, betas):
    print(f"\n{name}")
    print(f"{'net':>9} {'edges':>6} {'L/L_mst':>8} {'<d>':>7} {'detour':>7} {'f_max':>7} {'gini':>6}")
    keys = ["edges", "length_rel", "mean_d", "mean_detour", "f_max", "gini"]
    fmt = "{:>9} {:6.0f} {:8.3f} {:7.3f} {:7.3f} {:7.0f} {:6.3f}"
    print(fmt.format("Delaunay", *[ref_mean(refs, "Delaunay", k) for k in keys]))
    for b in betas:
        print(fmt.format(f"b={b:.2f}", *[summarise(rows, [b], k)[0][0] for k in keys]))
    print(fmt.format("MST", *[ref_mean(refs, "MST", k) for k in keys]))


# %% Run both experiments
if __name__ == "__main__":
    betas = np.round(np.arange(1.0, 2.0 + 1e-9, 0.1), 2)

    # same number of nodes (196) and same node density in both experiments, so they compare fairly
    print("Method (i): uniformly random nodes")
    random_rows, random_refs = sweep_random_nodes(betas, N=196, side=1.4, n_seeds=5)

    print("Method (ii): jittered grid")
    grid_rows, grid_refs = sweep_jittered_grid(betas, n=14, L=0.1, sigma=0.2, n_seeds=5)

    print_table("Method (i): uniformly random nodes", random_rows, random_refs, betas)
    print_table("Method (ii): jittered grid (sigma = 0.2 L)", grid_rows, grid_refs, betas)

    results = {"random nodes": (random_rows, random_refs),
               "jittered grid": (grid_rows, grid_refs)}
    plot_metrics_vs_beta(results, betas)
    plot_tradeoff(results, betas, ykey="mean_detour", ylabel="mean detour")
    plot_tradeoff(results, betas, ykey="f_max", ylabel="max edge flow")
    plt.show()
