# TMM Lab 02 - Q3 & Q4: synthetic cities from beta-skeletons
#   Q3: square domain, random or jittered-grid nodes, vary beta
#   Q4: the same on a torus (doubly periodic tile, Espinosa, Connors & Wilson)
# Run cell-by-cell or: python q3_q4_beta_skeletons.py  (~1 min)
# MATLAB versions: q3_beta_skeletons.m, q4_torus.m

# %% Imports
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from net_utils import (beta_skeleton, random_nodes, jittered_grid, edge_lengths,
                       network_metrics, edge_flows, gini)

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

# %% Q3: what the skeletons look like as beta varies
xy = random_nodes(150, side=1.0, rng=3)
betas_show = [0.8, 1.0, 1.5, 2.0, 2.5]
fig, axs = plt.subplots(2, len(betas_show), figsize=(18, 7.5))
for col, b in enumerate(betas_show):
    for row, kind in enumerate(["lune", "circle"]):
        E = beta_skeleton(xy, b, kind)
        ax = axs[row, col]
        ax.add_collection(LineCollection(xy[E], colors="k", linewidths=0.7))
        ax.plot(*xy.T, "r.", ms=3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{kind}, beta = {b}: {len(E)} edges")
fig.savefig(FIG / "q3_skeleton_gallery.png", dpi=130, bbox_inches="tight")
# beta = 1 is the Gabriel graph (same for both kinds); lune beta = 2 is the
# Relative Neighbourhood Graph. Bigger beta = bigger forbidden region = fewer
# edges. Lune skeletons can disconnect for beta > 2; circle ones stay
# connected for much longer (circle region grows more slowly).


# %% Q3: trade-off - total street length vs path length and congestion
def mst_length(xy):
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    return minimum_spanning_tree(sp.csr_matrix(np.triu(d))).sum()


def sweep(make_points, betas, kind="lune", seeds=range(4)):
    rows = []
    for seed in seeds:
        xy = make_points(seed)
        Lmst = mst_length(xy)
        for b in betas:
            E = beta_skeleton(xy, b, kind)
            m = network_metrics(xy, E, edge_lengths(xy, E))
            m.update(beta=b, seed=seed, length_rel=m["length"] / Lmst)
            rows.append(m)
    return rows


def summary(rows, betas, key):
    vals = np.array([[r[key] for r in rows if r["beta"] == b and r["connected"]] for b in betas],
                    dtype=object)
    return np.array([np.mean(v) if len(v) else np.nan for v in vals])


betas = np.round(np.arange(0.8, 2.01, 0.1), 2)
experiments = {
    "random nodes": lambda s: random_nodes(144, 1.2, rng=s),
    "jittered grid": lambda s: jittered_grid(12, 0.1, frac=0.6, rng=s),
}
results = {name: sweep(f, betas) for name, f in experiments.items()}

fig, axs = plt.subplots(1, 4, figsize=(18, 4))
for name, rows in results.items():
    for ax, key, lab in zip(axs, ["length_rel", "mean_detour", "max_flow", "gini"],
                            ["total length / MST length", "mean detour (route factor)",
                             "max edge flow", "Gini of edge flows"]):
        ax.plot(betas, summary(rows, betas, key), "o-", ms=3, label=name)
        ax.set_xlabel("beta"); ax.set_title(lab)
axs[0].legend()
fig.savefig(FIG / "q3_metrics_vs_beta.png", dpi=130, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 5))
for name, rows in results.items():
    x, y = summary(rows, betas, "length_rel"), summary(rows, betas, "mean_detour")
    ax.plot(x, y, "o-", label=name)
    for b, xx, yy in zip(betas[::3], x[::3], y[::3]):
        ax.annotate(f"{b}", (xx, yy), fontsize=8)
ax.set_xlabel("cost: total length / MST length"); ax.set_ylabel("performance: mean detour")
ax.legend(); ax.set_title("Q3 trade-off (labels = beta)")
fig.savefig(FIG / "q3_tradeoff.png", dpi=130, bbox_inches="tight")
for name, rows in results.items():
    print(name)
    for b in betas[::2]:
        print(f"  beta {b:.1f}: length/MST {summary(rows, [b], 'length_rel')[0]:.2f}, "
              f"detour {summary(rows, [b], 'mean_detour')[0]:.3f}, "
              f"max flow {summary(rows, [b], 'max_flow')[0]:.0f}")


# %% Q4: beta-skeleton on a torus (no boundary)
def torus_skeleton(xy, W, beta, kind="lune"):
    """Tile the N points 3 x 3, build the skeleton of the 9N points, keep the
    edges with one end in the central tile, and map both ends back mod N.
    Returns edges (m, 2) and lengths (m,). Each edge appears once."""
    N = len(xy)
    shifts = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]) * W
    big = np.vstack([xy + s for s in shifts])          # copy 4 (shift 0,0) = original
    E = beta_skeleton(big, beta, kind)
    centre = (E // N == 4)
    E = E[centre.any(1)]
    length = np.hypot(*(big[E[:, 0]] - big[E[:, 1]]).T)
    a, b = E[:, 0] % N, E[:, 1] % N
    key = np.column_stack([np.minimum(a, b), np.maximum(a, b), np.round(length, 9)])
    _, first = np.unique(key, axis=0, return_index=True)   # edge seen from both ends
    return np.column_stack([a, b])[first], length[first]


def torus_metrics(xy, W, beta):
    N = len(xy)
    E, w = torus_skeleton(xy, W, beta)
    flow, dist = edge_flows(N, E, w, return_dist=True)
    dd = xy[:, None] - xy[None]
    dd -= W * np.round(dd / W)                          # minimum-image displacement
    euclid = np.hypot(dd[..., 0], dd[..., 1])
    off = ~np.eye(N, dtype=bool)
    ok = np.isfinite(dist[off]).all()
    return dict(length=w.sum(), mean_degree=2 * len(E) / N,
                mean_detour=(dist[off] / euclid[off]).mean() if ok else np.nan,
                max_flow=flow.max(), mean_flow=flow.mean(), gini=gini(flow)), (E, w, flow)


W = 1.2
xy_t = random_nodes(144, W, rng=0)
E, w, flow = torus_metrics(xy_t, W, 1.5)[1]
fig, ax = plt.subplots(figsize=(6, 6))
for (a, b), f in zip(E, flow):          # draw each street via its nearest image
    d = xy_t[b] - xy_t[a]; d -= W * np.round(d / W)
    ax.plot([xy_t[a, 0], xy_t[a, 0] + d[0]], [xy_t[a, 1], xy_t[a, 1] + d[1]],
            color=plt.cm.hot_r(0.2 + 0.8 * f / flow.max()), lw=0.5 + 4 * f / flow.max())
ax.add_patch(plt.Rectangle((0, 0), W, W, fill=False, ls="--"))
ax.set_aspect("equal"); ax.set_title("Q4 torus skeleton, beta = 1.5 (flows)")
fig.savefig(FIG / "q4_torus_flows.png", dpi=130, bbox_inches="tight")

# beta >= 1 only: for beta < 1 every pair of the 9N tiled points is a candidate
# edge (840k pairs here), which is very slow in a Python loop.
betas_t = betas[betas >= 1]
rows = {b: [torus_metrics(random_nodes(144, W, rng=s), W, b)[0] for s in range(4)]
        for b in betas_t}
print("\nQ4 torus (random nodes, all-to-all):")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, key in zip(axs, ["length", "mean_detour", "max_flow"]):
    ax.plot(betas_t, [np.nanmean([r[key] for r in rows[b]]) for b in betas_t], "o-")
    ax.set_xlabel("beta"); ax.set_title(key)
fig.savefig(FIG / "q4_torus_metrics.png", dpi=130, bbox_inches="tight")
for b in betas_t[::2]:
    r = rows[b]
    print(f"  beta {b:.1f}: length {np.mean([x['length'] for x in r]):.1f}, "
          f"detour {np.nanmean([x['mean_detour'] for x in r]):.3f}, "
          f"max/mean flow {np.mean([x['max_flow'] / x['mean_flow'] for x in r]):.2f}")
# Without a boundary every location is statistically the same, so the flow
# is spread far more evenly (max/mean is small) than in the square, where
# the centre is a bottleneck. Higher beta -> shorter network but longer
# detours and higher, more concentrated flows.

plt.show()
