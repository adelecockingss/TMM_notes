"""
TMM Lab 02 - shared network helpers (numpy/scipy only, mirrors the MATLAB files).

Graph representation used throughout (same as MATLAB's G.Edges table):
    xy    : (n, 2) node coordinates
    edges : (m, 2) int array of node pairs (u, v)
    w     : (m,)  edge weights (lengths)
    directed=False -> two-way streets, True -> each row is a one-way street u -> v

Functions
    grid_network(n, L)            Manhattan grid (Q1)
    to_csgraph(...)               sparse weighted adjacency for scipy's dijkstra
    edge_flows(...)               all-or-nothing assignment of an OD matrix (Q1c)
    plot_flows(...)               'hotness' plot (Q1d)
    beta_skeleton(xy, beta, kind) lune / circle based beta-skeletons (Q3)
    network_metrics(...)          length / path / congestion summary (Q3, Q4)

MATLAB equivalents: grid_network.m, edge_flows.m, plot_flows.m, beta_skeleton.m
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Building networks
# ---------------------------------------------------------------------------
def grid_network(n=10, L=0.1):
    """n x n Manhattan grid with pitch L. Node k is at (L*floor(k/n), L*(k mod n)),
    exactly like the lab's  h.XData = L*floor((0:99)/10); h.YData = L*rem(0:99,10)."""
    k = np.arange(n * n)
    xy = np.column_stack([L * (k // n), L * (k % n)]).astype(float)
    edges = []
    for a in range(n * n):
        if a % n < n - 1:
            edges.append((a, a + 1))         # up (same column)
        if a + n < n * n:
            edges.append((a, a + n))         # right (next column)
    edges = np.array(edges)
    w = np.full(len(edges), float(L))
    return xy, edges, w


def edge_lengths(xy, edges):
    return np.hypot(*(xy[edges[:, 0]] - xy[edges[:, 1]]).T)


def to_csgraph(n, edges, w, directed=False):
    """Sparse adjacency matrix of weights, plus a matching matrix of edge ids
    (so that after a shortest-path search we can find which edge u->v is)."""
    u, v = edges[:, 0], edges[:, 1]
    if directed:
        rows, cols, data, ids = u, v, w, np.arange(len(w))
    else:
        rows = np.concatenate([u, v]); cols = np.concatenate([v, u])
        data = np.concatenate([w, w]); ids = np.concatenate([np.arange(len(w))] * 2)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # store id+1 so that id 0 is not confused with "no entry"
    EID = sp.csr_matrix((ids + 1, (rows, cols)), shape=(n, n))
    return W, EID


# ---------------------------------------------------------------------------
# Q1(c) flows: all-or-nothing assignment on shortest paths
# ---------------------------------------------------------------------------
def edge_flows(n, edges, w, demand=None, directed=False, return_dist=False):
    """Load the OD matrix `demand` (n x n, default all-to-all = 1 per ordered
    pair) onto shortest paths; return the flow on every edge.

    Trick (no need to list paths!): Dijkstra from origin s gives a shortest-path
    TREE: pred[v] = previous node on the route s -> v. Visit nodes from the
    furthest to the nearest; each node passes everything destined for itself
    and its descendants up to its parent:
        load[v] = demand[s, v];  for v far->near:  flow(pred[v], v) += load[v]
                                                   load[pred[v]]   += load[v]
    Total cost: one Dijkstra per origin. Ties between equally short paths are
    broken arbitrarily by Dijkstra (one path per OD pair)."""
    if demand is None:
        demand = np.ones((n, n)) - np.eye(n)
    W, EID = to_csgraph(n, edges, w, directed)
    origins = np.flatnonzero(demand.sum(1) > 0)
    dist, pred = dijkstra(W, directed=True, indices=origins, return_predecessors=True)
    flow = np.zeros(len(w))
    EIDd = EID.toarray() if n <= 3000 else EID.tolil()
    for r, s in enumerate(origins):
        load = demand[s].astype(float).copy()
        order = np.argsort(-dist[r])                   # furthest first
        for v in order:
            p = pred[r, v]
            if p < 0 or load[v] == 0:                  # origin itself, or unreachable
                continue
            flow[int(EIDd[p, v]) - 1] += load[v]
            load[p] += load[v]
    if return_dist:
        return flow, dist
    return flow


def all_pairs_distances(n, edges, w, directed=False):
    W, _ = to_csgraph(n, edges, w, directed)
    return dijkstra(W, directed=True)


def is_strongly_connected(n, edges, w, directed=False):
    W, _ = to_csgraph(n, edges, w, directed)
    k, _ = connected_components(W, directed=True, connection="strong")
    return k == 1


# ---------------------------------------------------------------------------
# Q1(d) 'hotness' plot
# ---------------------------------------------------------------------------
def plot_flows(ax, xy, edges, flow, cmap="hot_r", max_width=6, directed=False,
               title=None, vmax=None):
    """Colour AND width proportional to flow. One-way streets get arrows."""
    vmax = vmax or flow.max()
    segs = xy[edges]                                       # (m, 2, 2)
    lc = LineCollection(segs, cmap=cmap, linewidths=0.5 + max_width * flow / vmax,
                        array=flow, clim=(0, vmax))
    ax.add_collection(lc)
    if directed:
        mid = segs.mean(1); dvec = segs[:, 1] - segs[:, 0]
        ax.quiver(mid[:, 0], mid[:, 1], dvec[:, 0], dvec[:, 1], angles="xy",
                  scale_units="xy", scale=4, width=0.004, color="tab:blue", zorder=3)
    ax.plot(xy[:, 0], xy[:, 1], "k.", ms=2)
    ax.set_aspect("equal"); ax.autoscale()
    plt.colorbar(lc, ax=ax, label="flow")
    if title:
        ax.set_title(title)
    return lc


# ---------------------------------------------------------------------------
# Q3 beta-skeletons
# ---------------------------------------------------------------------------
def _blocked_mask(xy, i, j, beta, kind):
    """For candidate edges (i[k], j[k]), True if some OTHER point lies strictly
    inside the forbidden region. Vectorised over points; loop over edges."""
    out = np.zeros(len(i), bool)
    for k, (a, b) in enumerate(zip(i, j)):
        p, q = xy[a], xy[b]
        d = np.linalg.norm(q - p)
        if beta >= 1:
            R = beta * d / 2
            if kind == "lune":       # intersection of 2 discs, centres ON line pq
                c1 = (1 - beta / 2) * p + (beta / 2) * q
                c2 = (beta / 2) * p + (1 - beta / 2) * q
                inside = (np.linalg.norm(xy - c1, axis=1) < R - 1e-12) & \
                         (np.linalg.norm(xy - c2, axis=1) < R - 1e-12)
            else:                    # circle: union of 2 discs THROUGH p and q
                m = (p + q) / 2
                nrm = np.array([-(q - p)[1], (q - p)[0]]) / d
                h = d / 2 * np.sqrt(beta**2 - 1)
                inside = (np.linalg.norm(xy - (m + h * nrm), axis=1) < R - 1e-12) | \
                         (np.linalg.norm(xy - (m - h * nrm), axis=1) < R - 1e-12)
        else:                        # beta < 1 (both kinds): lens = intersection
            R = d / (2 * beta)       # of 2 discs of radius d/(2 beta) through p, q
            m = (p + q) / 2
            nrm = np.array([-(q - p)[1], (q - p)[0]]) / d
            h = np.sqrt(max(R**2 - (d / 2) ** 2, 0))
            inside = (np.linalg.norm(xy - (m + h * nrm), axis=1) < R - 1e-12) & \
                     (np.linalg.norm(xy - (m - h * nrm), axis=1) < R - 1e-12)
        inside[[a, b]] = False
        out[k] = inside.any()
    return out


def beta_skeleton(xy, beta, kind="lune"):
    """Edges (m, 2) of the beta-skeleton of the points xy.

    Edge pq is present iff no other point lies in the 'forbidden region' R(p,q,beta):
      beta = 1           : disc with diameter pq          -> Gabriel graph
      beta = 2, lune     : lune of two discs radius |pq|  -> Relative Neighbourhood Graph
      beta > 1           : region grows -> fewer edges (lune: may disconnect for beta > 2)
      beta < 1           : thin lens -> more edges, -> complete graph as beta -> 0
    For beta >= 1 the skeleton is a subgraph of the Gabriel graph, which is a
    subgraph of the Delaunay triangulation, so only Delaunay edges need testing.
    NB the File Exchange 'ProximityGraphs' toolbox uses a different parametrisation."""
    n = len(xy)
    if beta >= 1:
        s = Delaunay(xy).simplices
        cand = np.unique(np.sort(np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]]), 1), axis=0)
    else:
        i, j = np.triu_indices(n, 1)
        cand = np.column_stack([i, j])
    keep = ~_blocked_mask(xy, cand[:, 0], cand[:, 1], beta, kind)
    return cand[keep]


def random_nodes(N, side=1.0, rng=None):
    return np.random.default_rng(rng).uniform(0, side, (N, 2))


def jittered_grid(n, L=0.1, frac=0.5, rng=None):
    """n x n grid of pitch L, each coordinate moved by U(-frac*L/2, frac*L/2)."""
    rng = np.random.default_rng(rng)
    xy, _, _ = grid_network(n, L)
    return xy + rng.uniform(-frac * L / 2, frac * L / 2, xy.shape)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum())


def network_metrics(xy, edges, w, demand=None):
    """Cost: total length. Performance: mean shortest path, mean detour
    (route factor = network distance / straight line). Congestion: flows."""
    n = len(xy)
    flow, dist = edge_flows(n, edges, w, demand, return_dist=True)
    off = ~np.eye(n, dtype=bool)
    euclid = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    connected = np.isfinite(dist[off]).all()
    return dict(n_edges=len(edges), length=w.sum(), connected=connected,
                mean_path=dist[off].mean() if connected else np.inf,
                mean_detour=(dist[off] / euclid[off]).mean() if connected else np.inf,
                max_flow=flow.max(), mean_flow=flow.mean(), gini=gini(flow),
                mean_degree=2 * len(edges) / n)
