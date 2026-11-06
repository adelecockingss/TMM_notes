# TMM Lab 02 - Q3 comparison experiments
#   A: does running more layouts (5 vs 20) change the curves?  (SD band vs SE band)
#   B: how does grid jitter sigma change the trade-off, compared with random nodes?
#   C: square (with boundary) vs torus (doubly periodic, no boundary) - preview of Q4
# Figures are saved to ./figures and shown at the end.

# %% Imports
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from qu1 import edge_flows, uniform_demand, make_grid
from qu3 import random_nodes, blocked, delaunay_edges, beta_skeleton
from qu3c import (gini, shortest_path_matrix, network_metrics,
                  sweep_random_nodes, sweep_jittered_grid, summarise)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=110, bbox_inches="tight")


# %% Experiment A: 5 vs 20 layouts
def experiment_seed_count(betas, n_seeds=20):
    """Random-node sweep with many layouts; the first 5 are reused as the 'n = 5' case."""
    rows, _ = sweep_random_nodes(betas, N=196, side=1.4, n_seeds=n_seeds)
    return rows


def plot_seed_count(rows, betas, n_small=5):
    """Rows: f_max, Gini. Columns: 5 layouts, 20 layouts.
    Light band = +-1 SD (layout-to-layout spread); dark band = +-1 SE (uncertainty in the mean)."""
    n_big = len({r["seed"] for r in rows})
    subsets = {f"{n_small} layouts": [r for r in rows if r["seed"] < n_small],
               f"{n_big} layouts": rows}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey="row")
    for row, (key, label) in enumerate([("f_max", "max edge flow"), ("gini", "Gini of edge flows")]):
        for col, (title, sub) in enumerate(subsets.items()):
            n = len({r["seed"] for r in sub})
            mean, sd = summarise(sub, betas, key)
            se = sd / np.sqrt(n)
            ax = axes[row, col]
            ax.fill_between(betas, mean - sd, mean + sd, alpha=0.2, color="C0", label="$\\pm$1 SD")
            ax.fill_between(betas, mean - se, mean + se, alpha=0.5, color="C0", label="$\\pm$1 SE")
            ax.plot(betas, mean, "o-", ms=3, color="C0")
            ax.set_title(f"{label}: {title}")
            ax.set_xlabel(r"$\beta$")
            ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel(label)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Experiment A: more layouts shrink the SE band (mean uncertainty), not the SD band")
    fig.tight_layout()
    save(fig, "A_seed_count.png")


# %% Experiment B: grid jitter sigma vs random nodes
def experiment_grid_jitter(betas, sigmas, n_seeds=5):
    """One jittered-grid sweep per sigma. Returns {label: rows}."""
    out = {}
    for s in sigmas:
        print(f"sigma = {s}")
        rows, _ = sweep_jittered_grid(betas, n=14, L=0.1, sigma=s, n_seeds=n_seeds)
        out[f"grid, sigma = {s} L"] = rows
    return out


def plot_grid_jitter(results, betas):
    """Left: street length vs detour.  Middle: street length vs f_max.  Right: f_max vs beta."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for k, (name, rows) in enumerate(results.items()):
        x, _ = summarise(rows, betas, "length_rel")
        det, _ = summarise(rows, betas, "mean_detour")
        fm, fsd = summarise(rows, betas, "f_max")
        style = "k--" if "random" in name else "-"
        axes[0].plot(x, det, style, marker="o", ms=3, label=name)
        axes[1].plot(x, fm, style, marker="o", ms=3, label=name)
        axes[2].plot(betas, fm, style, marker="o", ms=3, label=name)
        # mark beta = 1 end of each curve
        axes[0].annotate(r"$\beta$=1", (x[0], det[0]), fontsize=7, xytext=(3, -8),
                         textcoords="offset points")
    axes[0].set(xlabel=r"street length $\Lambda/\Lambda_{MST}$", ylabel="mean detour",
                title="cost vs detour (each curve: beta 1 -> 2, right to left)")
    axes[1].set(xlabel=r"street length $\Lambda/\Lambda_{MST}$", ylabel="max edge flow",
                title="cost vs congestion")
    axes[2].set(xlabel=r"$\beta$", ylabel="max edge flow", title="congestion vs beta")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Experiment B: effect of grid jitter sigma (dashed = uniformly random nodes)")
    fig.tight_layout()
    save(fig, "B_grid_jitter.png")


# %% Experiment C helpers: networks on a torus (doubly periodic square of side W)
# Each edge stores 'vec' = displacement from its lower-numbered end u to the
# (nearest periodic image of the) higher-numbered end v, so wrapped edges can be drawn.
def torus_grid(n, L):
    """n x n grid wrapped onto a torus of side W = n L (every node has degree 4)."""
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            G.add_node(i * n + j, pos=(L * (i + 0.5), L * (j + 0.5)))
    for i in range(n):
        for j in range(n):
            for di, dj in [(1, 0), (0, 1)]:
                a, b = i * n + j, ((i + di) % n) * n + (j + dj) % n
                vec = np.array([di * L, dj * L])
                u, v, vec = (a, b, vec) if a < b else (b, a, -vec)
                G.add_edge(u, v, weight=L, vec=vec)
    return G


def torus_beta_skeleton(pts, beta, W):
    """Lune beta-skeleton on a torus: tile the points 3x3, take Delaunay candidates with an
    endpoint in the central tile, test the forbidden region against ALL tiled points,
    then fold indices back mod N. (This is the core of Q4.)"""
    N = len(pts)
    offsets = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]) * W
    tiled = np.vstack([pts + o for o in offsets])     # tile 4 is the original (0, 0) copy
    G = nx.Graph()
    for k, xy in enumerate(pts):
        G.add_node(k, pos=tuple(xy))
    for a, b in delaunay_edges(tiled):
        if not (4 * N <= a < 5 * N):
            a, b = b, a                               # make a the central-tile endpoint
            if not (4 * N <= a < 5 * N):
                continue                              # neither endpoint central: ignore
        i, j = a % N, b % N
        if i == j or G.has_edge(i, j):
            continue                                  # self-image, or already handled
        if blocked(tiled, a, b, beta, "lune"):
            continue
        vec = tiled[b] - tiled[a]
        u, v, vec = (i, j, vec) if i < j else (j, i, -vec)
        G.add_edge(u, v, weight=float(np.linalg.norm(vec)), vec=vec)
    return G


def torus_metrics(G, D, W):
    """Same metrics as network_metrics, but straight-line distance uses the periodic
    (minimum-image) distance, so the detour is measured against the torus geometry."""
    N = G.number_of_nodes()
    pts = np.array([G.nodes[k]["pos"] for k in range(N)])
    diff = np.abs(pts[:, None, :] - pts[None, :, :])
    diff = np.minimum(diff, W - diff)
    euclid = np.sqrt((diff**2).sum(axis=2))
    dG = shortest_path_matrix(G)
    off = ~np.eye(N, dtype=bool)
    flows = np.array(list(edge_flows(G, D).values()))
    return dict(edges=G.number_of_edges(), length=G.size(weight="weight"),
                mean_d=dG[off].mean(), mean_detour=(dG[off] / euclid[off]).mean(),
                f_max=flows.max(), f_mean=flows.mean(), gini=gini(flows))


def draw_flows(ax, G, flows, vmax, title, W=None):
    """Hotness map on a shared colour scale. If W is given, edges are drawn from both
    ends using 'vec' and clipped to the [0, W]^2 tile, so wrapped edges show at both sides."""
    pos = nx.get_node_attributes(G, "pos")
    segs, f = [], []
    for (u, v), fl in flows.items():
        if W is None:
            segs.append((pos[u], pos[v])); f.append(fl)
        else:
            vec = G.edges[u, v]["vec"]
            pu, pv = np.array(pos[u]), np.array(pos[v])
            segs += [(pu, pu + vec), (pv, pv - vec)]; f += [fl, fl]
    f = np.array(f)
    lc = LineCollection(segs, array=f, cmap="hot_r", norm=plt.Normalize(0, vmax),
                        linewidths=0.5 + 5 * f / vmax)
    ax.add_collection(lc)
    if W is not None:
        ax.set_xlim(0, W); ax.set_ylim(0, W)
        ax.plot([0, W, W, 0, 0], [0, 0, W, W, 0], "b--", lw=0.8)   # the periodic tile
    else:
        ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    return lc


# %% Experiment C: square vs torus
def plot_square_vs_torus_maps(beta=1.5, N=196, W=1.4, seed=0):
    """Flow maps: 10x10 grid open vs periodic; beta-skeleton square vs torus (same nodes).
    Each row shares one colour scale so hotness is directly comparable."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    n, L = 10, 0.1
    Gs, Gt = make_grid(n, L), torus_grid(n, L)
    fs, ft = edge_flows(Gs, uniform_demand(n * n)), edge_flows(Gt, uniform_demand(n * n))
    vmax = max(max(fs.values()), max(ft.values()))
    draw_flows(axes[0, 0], Gs, fs, vmax, f"10x10 grid, square: max {max(fs.values()):.0f}")
    lc = draw_flows(axes[0, 1], Gt, ft, vmax,
                    f"10x10 grid, torus: flows {min(ft.values()):.0f}-{max(ft.values()):.0f}",
                    W=n * L)
    fig.colorbar(lc, ax=axes[0, :], shrink=0.8, label="edge flow")
    print(f"torus grid flows: min {min(ft.values()):.0f}, max {max(ft.values()):.0f}, "
          f"mean {np.mean(list(ft.values())):.1f}")

    pts = random_nodes(N, side=W, rng=seed)
    Gs, Gt = beta_skeleton(pts, beta, "lune"), torus_beta_skeleton(pts, beta, W)
    fs, ft = edge_flows(Gs, uniform_demand(N)), edge_flows(Gt, uniform_demand(N))
    vmax = max(max(fs.values()), max(ft.values()))
    draw_flows(axes[1, 0], Gs, fs, vmax, f"beta = {beta} skeleton, square: max {max(fs.values()):.0f}")
    lc = draw_flows(axes[1, 1], Gt, ft, vmax,
                    f"beta = {beta} skeleton, torus: max {max(ft.values()):.0f}", W=W)
    fig.colorbar(lc, ax=axes[1, :], shrink=0.8, label="edge flow")
    fig.suptitle("Experiment C: removing the boundary (same colour scale within each row)")
    save(fig, "C_flow_maps.png")


def experiment_square_vs_torus(betas, N=196, W=1.4, n_seeds=5):
    """Torus sweep on the same random layouts (seeds 0..n_seeds-1) as sweep_random_nodes."""
    rows = []
    for seed in range(n_seeds):
        pts = random_nodes(N, side=W, rng=seed)
        D = uniform_demand(N)
        for b in betas:
            G = torus_beta_skeleton(pts, b, W)
            if not nx.is_connected(G):
                print(f"  torus seed {seed}, beta {b}: disconnected - skipped")
                continue
            m = torus_metrics(G, D, W)
            m.update(beta=b, seed=seed)
            rows.append(m)
        print(f"  torus layout {seed + 1}/{n_seeds} done")
    return rows


def plot_square_vs_torus_metrics(square_rows, torus_rows, betas):
    panels = [("length", "total street length"), ("mean_detour", "mean detour"),
              ("f_max", "max edge flow"), ("gini", "Gini of edge flows")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (key, label) in zip(axes.flat, panels):
        for name, rows in [("square", square_rows), ("torus", torus_rows)]:
            mean, sd = summarise(rows, betas, key)
            ax.plot(betas, mean, "o-", ms=3, label=name)
            ax.fill_between(betas, mean - sd, mean + sd, alpha=0.25)
        ax.set(xlabel=r"$\beta$", ylabel=label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Experiment C: square vs torus, same node layouts (mean $\\pm$ 1 SD, 5 layouts)")
    fig.tight_layout()
    save(fig, "C_square_vs_torus.png")


# %% Run each experiment
if __name__ == "__main__":
    betas = np.round(np.arange(1.0, 2.0 + 1e-9, 0.1), 2)

    print("Experiment A: 20 random layouts")
    random_rows = experiment_seed_count(betas, n_seeds=20)
    plot_seed_count(random_rows, betas)

    print("Experiment B: grid jitter")
    jitter_results = experiment_grid_jitter(betas, sigmas=[0.05, 0.2, 0.5])
    jitter_results["random nodes"] = [r for r in random_rows if r["seed"] < 5]
    plot_grid_jitter(jitter_results, betas)

    print("Experiment C: square vs torus")
    plot_square_vs_torus_maps(beta=1.5)
    torus_rows = experiment_square_vs_torus(betas)
    square_rows = [r for r in random_rows if r["seed"] < 5]   # same 5 layouts, square version
    plot_square_vs_torus_metrics(square_rows, torus_rows, betas)

    print(f"figures saved to {FIG_DIR}")
    plt.show()
