# TMM Lab 02 - Q3(a): reusable edge-flow (all-or-nothing assignment) + hotness plot
# Test case: 10x10 square grid, pitch L = 0.1 km, uniform all-to-all demand.
# Check: sum_e f_e * l_e == sum_{s,t} d(s,t) == 6600 km.

# %% Imports
import heapq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# %% Graph construction
def make_grid(n=10, L=0.1):
    """n x n square grid with pitch L. Nodes are integers 0..n^2-1 (so they index
    rows/cols of a demand matrix); each node has 'pos' = (x, y), each edge 'weight' = L."""
    G = nx.grid_2d_graph(n, n)
    # relabel (i, j) tuples -> 0..N-1; keep the original tuple in node attribute 'ij'
    G = nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="ij")
    for v, data in G.nodes(data=True):
        i, j = data["ij"]
        data["pos"] = (L * i, L * j)
    nx.set_edge_attributes(G, L, "weight")
    return G


def uniform_demand(N):
    """All-to-all demand: one trip between every ordered pair s != t."""
    return np.ones((N, N)) - np.eye(N)


# %% Edge flows
def edge_flows(G, D, weight="weight", split_ties=True):
    """All-or-nothing assignment of demand D (N x N, indexed by node label) onto
    shortest paths. Returns dict {(u, v) with u < v: flow} (undirected, two-way streets).
      split_ties=True : each OD demand is shared EQUALLY over all tied shortest paths
      split_ties=False: ONE shortest path per OD pair (networkx's tie-breaking decides)"""
    if split_ties:
        return edge_flows_split_ties(G, D, weight)
    return edge_flows_single_path(G, D, weight)


def shortest_path_dag(G, s, weight="weight", rtol=1e-9):
    """Dijkstra from s that keeps ALL tied predecessors.
    Returns (order, pred, sigma): nodes in order of increasing distance, pred[w] = list
    of predecessors of w on shortest paths, sigma[w] = number of shortest s-w paths.
    Distances within rtol (relative) count as tied - needed because e.g. on the grid
    0.1 + 0.2 != 0.3 in floating point, so equal-length routes differ in the last bit."""
    dist, sigma, pred = {s: 0.0}, {s: 1.0}, {s: []}
    order, done = [], set()
    heap = [(0.0, s)]
    while heap:
        d, v = heapq.heappop(heap)
        if v in done:
            continue                                  # stale heap entry
        done.add(v)
        order.append(v)
        for w, attr in G[v].items():
            nd = d + attr[weight]
            tol = rtol * max(1.0, nd)
            if w not in dist or nd < dist[w] - tol:   # strictly shorter route found
                dist[w], sigma[w], pred[w] = nd, sigma[v], [v]
                heapq.heappush(heap, (nd, w))
            elif abs(nd - dist[w]) <= tol and w not in done:   # tied route
                sigma[w] += sigma[v]
                pred[w].append(v)
    return order, pred, sigma


def edge_flows_split_ties(G, D, weight="weight"):
    """Brandes-style accumulation: demand to each destination is passed back towards the
    origin, split between predecessors in proportion to their number of shortest paths
    (sigma_v / sigma_w). Net effect: every tied shortest path carries an equal share."""
    flows = {tuple(sorted(e)): 0.0 for e in G.edges}
    for s in G.nodes:
        order, pred, sigma = shortest_path_dag(G, s, weight)
        reached = set(order)
        unreachable = [t for t in G.nodes if t not in reached and D[s, t] > 0]
        if unreachable:
            raise ValueError(f"graph disconnected: node {s} cannot reach {len(unreachable)} "
                             f"destination(s) with positive demand")
        # delta[w] = total demand from s that must pass through w (to w or beyond)
        delta = dict.fromkeys(order, 0.0)
        for w in reversed(order):                     # farthest nodes first
            if w != s:
                delta[w] += D[s, w]                   # trips ending at w
            for v in pred[w]:
                share = sigma[v] / sigma[w] * delta[w]
                flows[(min(v, w), max(v, w))] += share
                delta[v] += share
    return flows


def edge_flows_single_path(G, D, weight="weight"):
    """One shortest path per OD pair (original Q3(a) version, tie-breaking dependent)."""
    flows = {tuple(sorted(e)): 0.0 for e in G.edges}
    for s in G.nodes:
        # one Dijkstra per origin -> shortest paths from s to every reachable t
        paths = nx.single_source_dijkstra_path(G, s, weight=weight)
        # demand to an unreachable node would otherwise be silently dropped
        unreachable = [t for t in G.nodes if t not in paths and D[s, t] > 0]
        if unreachable:
            raise ValueError(f"graph disconnected: node {s} cannot reach {len(unreachable)} "
                             f"destination(s) with positive demand")
        for t, path in paths.items():
            if t == s or D[s, t] == 0:
                continue
            # add this OD pair's demand to every edge along its path
            for u, v in zip(path[:-1], path[1:]):
                flows[(min(u, v), max(u, v))] += D[s, t]
    return flows


def total_travel_distance(G, D, weight="weight"):
    """sum_{s,t} D_st * d(s,t) computed directly from shortest-path lengths (for checking)."""
    total = 0.0
    for s, lengths in nx.all_pairs_dijkstra_path_length(G, weight=weight):
        for t, d in lengths.items():
            total += D[s, t] * d
    return total


# %% Hotness plot
def plot_flows(G, flows, ax=None, max_width=6.0, cmap="hot_r", title=None):
    """Draw edges coloured and thickened by flow. Returns the axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    pos = nx.get_node_attributes(G, "pos")
    edges = list(flows.keys())
    f = np.array([flows[e] for e in edges])
    segments = [(pos[u], pos[v]) for u, v in edges]

    norm = plt.Normalize(vmin=f.min(), vmax=f.max())
    widths = 0.5 + max_width * f / f.max()
    lc = LineCollection(segments, array=f, cmap=cmap, norm=norm, linewidths=widths)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    plt.colorbar(lc, ax=ax, label="edge flow (trips)")
    if title:
        ax.set_title(title)
    return ax


# %% Test on the 10x10 grid
if __name__ == "__main__":
    n, L = 10, 0.1
    G = make_grid(n, L)
    N = G.number_of_nodes()
    D = uniform_demand(N)

    direct_km = total_travel_distance(G, D)
    print(f"sum_(s,t) d(s,t)     = {direct_km:.4f} km")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, split in zip(axes, [False, True]):
        flows = edge_flows(G, D, split_ties=split)
        label = "split over tied paths" if split else "one path per OD pair"

        # check 1: flow-weighted edge length == total of all shortest-path distances
        flow_km = sum(f * G.edges[e]["weight"] for e, f in flows.items())
        assert np.isclose(flow_km, direct_km) and np.isclose(flow_km, 6600.0)

        # some summary statistics of congestion
        f = np.array(list(flows.values()))
        print(f"{label:>22}: sum f_e*l_e = {flow_km:.4f} km, max flow = {f.max():.1f}, "
              f"min flow = {f.min():.1f}, mean flow = {f.mean():.1f}")
        plot_flows(G, flows, ax=ax, title=f"{n}x{n} grid, {label}")

    # check 2: split-ties flows on a unit-weight grid = 2 x networkx edge betweenness
    # (networkx counts each unordered pair once; our uniform demand counts both directions)
    G1 = make_grid(n, 1.0)
    ours = edge_flows(G1, uniform_demand(N))
    nx_eb = nx.edge_betweenness_centrality(G1, normalized=False)
    assert all(np.isclose(ours[tuple(sorted(e))], 2 * b) for e, b in nx_eb.items())
    print("split-ties flows match 2 x networkx edge betweenness")
    plt.show()
