# TMM Lab 02 - Q2: perturbing the square grid so that shortest paths are unique
# Run cell-by-cell or: python q2_perturbed_grid.py.   MATLAB: q2_perturbed_grid.m

# %% Imports
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt

from net_utils import (grid_network, edge_lengths, edge_flows, plot_flows,
                       is_strongly_connected, all_pairs_distances)

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(1)
n, L = 10, 0.1
xy, edges, w = grid_network(n, L)
N = n * n

# %% Q2(a) random edge weights U(0, 2L)
w_rand = rng.uniform(0, 2 * L, len(w))
f_a = edge_flows(N, edges, w_rand)
# Ties are now (almost surely) impossible -> every shortest path is unique and
# the result no longer depends on how the algorithm breaks ties.

# %% Q2(b) perturbed node positions: x, y + U(-L/2, L/2), weights = new lengths
xy_b = xy + rng.uniform(-L / 2, L / 2, xy.shape)
w_b = edge_lengths(xy_b, edges)          # the "new input matrix A" in MATLAB terms
f_b = edge_flows(N, edges, w_b)

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
plot_flows(axs[0], xy, edges, f_a, title="Q2(a) random weights U(0, 2L)")
plot_flows(axs[1], xy_b, edges, f_b, title="Q2(b) jittered nodes")
fig.savefig(FIG / "q2ab_flows.png", dpi=150, bbox_inches="tight")


# %% Q2(c) closing streets while keeping everything reachable
# A connected graph on N nodes needs at least N - 1 edges (a spanning tree).
# Protect the edges of a minimum spanning tree, randomly delete the others.
W = sp.csr_matrix((w_b, (edges[:, 0], edges[:, 1])), shape=(N, N))
mst = minimum_spanning_tree(W).tocoo()
mst_set = {tuple(sorted(e)) for e in zip(mst.row, mst.col)}
in_mst = np.array([tuple(sorted(e)) in mst_set for e in edges])
print(f"{len(edges)} streets, MST has {in_mst.sum()} (= N - 1): up to "
      f"{len(edges) - in_mst.sum()} can be closed with everything still reachable")

# Cost of closures: mean shortest-path length as more streets close
order = rng.permutation(np.flatnonzero(~in_mst))
closed_counts, mean_d = [], []
for k in range(0, len(order) + 1, 5):
    keep = np.ones(len(edges), bool); keep[order[:k]] = False
    D = all_pairs_distances(N, edges[keep], w_b[keep])
    closed_counts.append(k); mean_d.append(D[~np.eye(N, dtype=bool)].mean())

# One-way streets: make streets one-way (random direction) one at a time and
# keep the change only if every node can still reach every other node
# (= the directed graph is STRONGLY connected).
dir_edges = [tuple(e) for e in edges]         # each two-way street = 2 arcs
is_oneway = np.zeros(len(edges), bool)
for k in rng.permutation(len(edges)):
    u, v = edges[k]
    trial_oneway = is_oneway.copy(); trial_oneway[k] = True
    flip = rng.random() < 0.5
    arcs, wts = [], []
    for kk, (a, b) in enumerate(edges):
        if trial_oneway[kk]:
            arcs.append((b, a) if (kk == k and flip) or (kk != k and dir_edges[kk] == (b, a))
                        else (a, b))
            wts.append(w_b[kk])
        else:
            arcs += [(a, b), (b, a)]; wts += [w_b[kk]] * 2
    if is_strongly_connected(N, np.array(arcs), np.array(wts), directed=True):
        is_oneway = trial_oneway
        dir_edges[k] = (v, u) if flip else (u, v)
print(f"one-way streets possible (greedy, random order): {is_oneway.sum()} of {len(edges)}")
# Robbins' theorem: ALL streets can be made one-way and keep strong connectivity
# iff the graph has no bridge (every edge lies on a cycle). The greedy random
# order usually gets stuck well before that.

# %% Q2(d) plot one- vs two-way streets
arcs = np.array([dir_edges[k] for k in range(len(edges))])
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(closed_counts, mean_d, "o-")
axs[0].set_xlabel("streets closed (MST protected)"); axs[0].set_ylabel("mean shortest path (km)")
axs[0].set_title("Q2(c) cost of closing streets")
two = ~is_oneway
for (a, b) in edges[two]:
    axs[1].plot(*xy_b[[a, b]].T, color="0.5", lw=2.5)
seg = xy_b[arcs[is_oneway]]
d = seg[:, 1] - seg[:, 0]
axs[1].quiver(seg[:, 0, 0], seg[:, 0, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy",
              scale=1, color="tab:red", width=0.005)
axs[1].set_aspect("equal"); axs[1].set_title("Q2(d) grey = two-way, red arrows = one-way")
fig.savefig(FIG / "q2cd_closures_oneway.png", dpi=150, bbox_inches="tight")

# %% Q2(e) unconstrained gravity-model demand instead of all-to-all
P = rng.integers(50, 500, N)                              # random 'population' per node
Dst = np.hypot(*(xy_b[:, None, :] - xy_b[None, :, :]).transpose(2, 0, 1))
np.fill_diagonal(Dst, np.inf)
T = np.outer(P, P) / Dst**2
T *= (N * (N - 1)) / T.sum()                              # same total as all-to-all
f_e = edge_flows(N, edges, w_b, T)
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
plot_flows(axs[0], xy_b, edges, f_b, title="uniform all-to-all")
plot_flows(axs[1], xy_b, edges, f_e, title="gravity demand P_i P_j / d^2")
axs[1].scatter(*xy_b.T, s=P / 10, c="tab:blue", alpha=0.4, zorder=3)
fig.savefig(FIG / "q2e_gravity_flows.png", dpi=150, bbox_inches="tight")
# Gravity demand is dominated by SHORT trips between big neighbours, so flow
# concentrates near populous nodes instead of through the centre.

plt.show()
