# TMM Lab 02 - Q1: working with graphs on a square-grid ("Manhattan") network
# Run cell-by-cell or: python q1_grid.py.   MATLAB version: q1_grid.m
# Python has no single built-in like MATLAB's graph object; networkx is the
# closest (used in (a)-(b)), scipy.sparse.csgraph is the fast option (c)-(e).

# %% Imports and the network
from pathlib import Path
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from net_utils import grid_network, edge_flows, plot_flows, all_pairs_distances

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

n, L = 10, 0.1                       # 10 x 10 nodes, 100 m pitch
xy, edges, w = grid_network(n, L)
N = n * n
print(f"{N} nodes, {len(edges)} edges (expected 2 n (n-1) = {2 * n * (n - 1)})")

# networkx version of the same graph (like MATLAB's graph(A)).
# NB Python counts from 0: MATLAB node 1 = Python node 0, node 100 = 99.
G = nx.Graph()
G.add_nodes_from(range(N))
G.add_weighted_edges_from([(u, v, wt) for (u, v), wt in zip(edges, w)])
pos = {k: xy[k] for k in range(N)}

# %% Q1(a) neighbours and degree
for k, kind in [(0, "corner"), (1, "edge"), (55, "interior")]:
    print(f"neighbours of node {k} ({kind}):", sorted(int(v) for v in G.neighbors(k)))
deg = np.array([d for _, d in G.degree()])
print("degree counts:", {int(k): int((deg == k).sum()) for k in np.unique(deg)})
# expected: 4 corners of degree 2, 4(n-2) = 32 boundary nodes of degree 3,
# (n-2)^2 = 64 interior nodes of degree 4

# %% Q1(b) shortest path from node 1 to node 100 (MATLAB numbering) = 0 -> 99
path = [int(v) for v in nx.shortest_path(G, 0, N - 1, weight="weight")]
length = nx.shortest_path_length(G, 0, N - 1, weight="weight")
edgepath = [tuple(sorted(e)) for e in zip(path[:-1], path[1:])]
print(f"path: {path}\nlength: {length:.1f} km (= 18 blocks x 0.1 km)")
print(f"number of equally short paths: {sum(1 for _ in nx.all_shortest_paths(G, 0, N - 1))}")
# = C(18, 9) = 48620: choose which 9 of the 18 steps go 'right'

fig, ax = plt.subplots(figsize=(5, 5))
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.7")
nx.draw_networkx_edges(G, pos, edgelist=edgepath, ax=ax, edge_color="r", width=3)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color="k")
ax.set_aspect("equal"); ax.set_title("Q1(b) one shortest path 1 -> 100 (highlighted)")
fig.savefig(FIG / "q1b_shortest_path.png", dpi=150, bbox_inches="tight")

# %% Q1(c) all 10,000 shortest paths, and how often each edge is used
# Version 1 - the obvious loop over all OD pairs (like a MATLAB double loop)
eid = {tuple(sorted(e)): k for k, e in enumerate(map(tuple, edges))}
t0 = time.perf_counter()
count_loop = np.zeros(len(edges))
for s in range(N):
    paths = nx.single_source_dijkstra_path(G, s, weight="weight")   # one call per origin
    for t, p in paths.items():
        if t != s:
            for e in zip(p[:-1], p[1:]):
                count_loop[eid[tuple(sorted(e))]] += 1
print(f"loop over paths:  {time.perf_counter() - t0:.2f} s")

# Version 2 - shortest-path trees + accumulation (see net_utils.edge_flows)
t0 = time.perf_counter()
flow = edge_flows(N, edges, w)
print(f"tree accumulation: {time.perf_counter() - t0:.2f} s")
print(f"flows: min {flow.min():.0f}, max {flow.max():.0f}, total trips {N * (N - 1)}")

# Check: sum over edges of flow x length = sum over pairs of path length
D = all_pairs_distances(N, edges, w)
print(f"sum f_e l_e = {np.sum(flow * w):.1f} km;  sum_st d(s,t) = {D.sum():.1f} km")
# NB the two versions give different edge counts (but the same totals!) because
# Dijkstra breaks the many ties differently - shortest paths are NOT unique here.
print("same flows in both versions?", np.allclose(count_loop, flow))

# %% Q1(d) 'hotness' plot of the flows
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
plot_flows(axs[0], xy, edges, flow, title="Q1(d) all-to-all flows (one path per pair)")
# Making paths unique by tie-splitting: a tiny random perturbation of the
# weights picks a random shortest path for each pair - average many of them.
rng = np.random.default_rng(0)
flow_avg = np.mean([edge_flows(N, edges, w * (1 + 1e-6 * rng.random(len(w))))
                    for _ in range(20)], axis=0)
plot_flows(axs[1], xy, edges, flow_avg, title="average over 20 random tie-breaks")
fig.savefig(FIG / "q1d_hotness.png", dpi=150, bbox_inches="tight")
# Centre streets are busiest (most routes pass through the middle) - this is
# edge betweenness centrality: nx.edge_betweenness_centrality(G, weight='weight')

# %% Q1(e) Low Traffic Neighbourhood (LTN) idea
# An LTN puts "modal filters" (bollards) on residential streets so that cars
# can still reach every house but cannot drive THROUGH the area.
# Demo: main roads form the square x = 2..7, y = 2..7 (units of 100 m); the
# 4 x 4 block of nodes inside (x, y in 3..6) is the LTN.
# Rule for zero through traffic: every piece of the LTN must be entered from
# ONE side only. Filters split it into two cul-de-sac halves: the west half
# (x = 3, 4) is entered only from the south road, the east half (x = 5, 6)
# only from the north road.
ltn = {k for k in range(N) if 3 <= k // n <= 6 and 3 <= k % n <= 6}
def filtered(u, v):
    (xu, yu), (xv, yv) = divmod(int(u), n), divmod(int(v), n)
    if u not in ltn and v not in ltn:
        return False                                  # main roads / outside: open
    xs, ys = {xu, xv}, {yu, yv}
    if xs == {4, 5}:                                  # between the two halves
        return True
    if xs in ({2, 3}, {6, 7}):                        # west / east access
        return True
    if ys == {6, 7} and xs <= {3, 4}:                 # west half: no north access
        return True
    if ys == {2, 3} and xs <= {5, 6}:                 # east half: no south access
        return True
    return False
keep = np.array([not filtered(u, v) for u, v in edges])
edges_ltn, w_ltn = edges[keep], w[keep]
print(f"LTN: {np.sum(~keep)} streets filtered")

# Through traffic = trips with BOTH ends outside the LTN
outside = np.array([k not in ltn for k in range(N)])
demand_through = np.outer(outside, outside).astype(float); np.fill_diagonal(demand_through, 0)
inside_edge = lambda E: np.array([u in ltn and v in ltn for u, v in E])
f_before = edge_flows(N, edges, w, demand_through)
f_after = edge_flows(N, edges_ltn, w_ltn, demand_through)
print(f"through-traffic on LTN streets: before {f_before[inside_edge(edges)].sum():.0f}, "
      f"after {f_after[inside_edge(edges_ltn)].sum():.0f}")
Db, Da = all_pairs_distances(N, edges, w), all_pairs_distances(N, edges_ltn, w_ltn)
print(f"total travel distance: +{100 * (Da.sum() / Db.sum() - 1):.1f}% "
      "(the cost of the LTN: longer trips for residents)")

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
vmax = max(f_before.max(), f_after.max())
plot_flows(axs[0], xy, edges, f_before, title="through traffic, no LTN", vmax=vmax)
plot_flows(axs[1], xy, edges_ltn, f_after, title="through traffic, with LTN filters", vmax=vmax)
for ax in axs:
    ax.add_patch(plt.Rectangle((0.25, 0.25), 0.4, 0.4, fill=False, ec="g", lw=2, ls="--"))
fig.savefig(FIG / "q1e_ltn.png", dpi=150, bbox_inches="tight")

plt.show()
