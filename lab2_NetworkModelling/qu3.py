# TMM Lab 02 - Q3(b): beta-skeletons (lune- and circle-based, beta >= 1)
# Build the skeleton from Delaunay candidate edges, verify it against brute-force
# Gabriel (beta = 1) and RNG (beta = 2) graphs, then look at how it changes with beta.

# %% Imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform

from qu1 import edge_flows, plot_flows, uniform_demand   # reuse the Q3(a) machinery


# %% Node placement
def random_nodes(N, side=1.0, rng=None):
    """Option (i): N nodes uniformly at random in a side x side square."""
    rng = np.random.default_rng(rng)
    return rng.uniform(0, side, size=(N, 2))


def jittered_grid(n, L=0.1, sigma=0.2, rng=None):
    """Option (ii): n x n grid of pitch L, each node moved by N(0, (sigma*L)^2) per axis.
    Some jitter (sigma > 0) is needed: exact grid squares are cocircular (degenerate)."""
    rng = np.random.default_rng(rng)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pts = L * np.column_stack([i.ravel(), j.ravel()]).astype(float)
    return pts + rng.normal(0.0, sigma * L, size=pts.shape)


# %% Candidate edges and the forbidden-region test
def delaunay_edges(pts):
    """Unique undirected Delaunay edges as an (E, 2) int array with i < j."""
    s = Delaunay(pts).simplices                     # (M, 3) triangles
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e.sort(axis=1)                                  # (j, i) -> (i, j)
    return np.unique(e, axis=0)                     # each shared side appears twice


def blocked(pts, i, j, beta, kind="lune"):
    """True if any point other than i, j lies strictly inside the forbidden region of (i, j)."""
    p, q = pts[i], pts[j]
    d = np.linalg.norm(q - p)
    R = beta * d / 2                                # radius of both discs
    if kind == "lune":
        # intersection of two discs centred ON the line pq
        c1 = (1 - beta / 2) * p + (beta / 2) * q
        c2 = (beta / 2) * p + (1 - beta / 2) * q
        inside = (np.linalg.norm(pts - c1, axis=1) < R) & (np.linalg.norm(pts - c2, axis=1) < R)
    elif kind == "circle":
        # union of two discs through p and q, centred on the perpendicular bisector
        m = (p + q) / 2
        n_hat = np.array([-(q - p)[1], (q - p)[0]]) / d
        h = (d / 2) * np.sqrt(beta**2 - 1)
        inside = (np.linalg.norm(pts - (m + h * n_hat), axis=1) < R) | \
                 (np.linalg.norm(pts - (m - h * n_hat), axis=1) < R)
    else:
        raise ValueError("kind must be 'lune' or 'circle'")
    inside[[i, j]] = False                          # p, q sit on the boundary; never self-block
    return inside.any()


def beta_skeleton(pts, beta, kind="lune"):
    """beta >= 1 skeleton as nx.Graph: integer nodes with 'pos', edges with 'weight' = length."""
    if beta < 1:
        raise ValueError("beta < 1 skeletons are not subgraphs of Delaunay - need all pairs")
    G = nx.Graph()
    for k, xy in enumerate(pts):
        G.add_node(k, pos=tuple(xy))
    for i, j in delaunay_edges(pts):
        if not blocked(pts, i, j, beta, kind):
            G.add_edge(int(i), int(j), weight=float(np.linalg.norm(pts[i] - pts[j])))
    return G


# %% Independent brute-force checks (all pairs, no Delaunay)
def edge_set(G):
    return {tuple(sorted(e)) for e in G.edges}


def gabriel_bruteforce(pts):
    """Gabriel: keep (i, j) iff no r has angle irj > 90 deg, i.e. (p - r).(q - r) < 0."""
    N, E = len(pts), set()
    for i in range(N):
        for j in range(i + 1, N):
            dots = np.einsum("kd,kd->k", pts[i] - pts, pts[j] - pts)
            dots[[i, j]] = np.inf
            if not (dots < 0).any():
                E.add((i, j))
    return E


def rng_bruteforce(pts):
    """Relative neighbourhood graph: keep (i, j) iff no r with max(|ri|, |rj|) < |ij|."""
    D = squareform(pdist(pts))
    N, E = len(pts), set()
    for i in range(N):
        for j in range(i + 1, N):
            dmax = np.maximum(D[i], D[j])
            dmax[[i, j]] = np.inf
            if not (dmax < D[i, j]).any():
                E.add((i, j))
    return E


# %% Run: verification + pictures
if __name__ == "__main__":
    N = 200
    pts = random_nodes(N, side=1.0, rng=1)
    n_dt = len(delaunay_edges(pts))
    print(f"N = {N}, Delaunay edges = {n_dt}  (3N - 6 = {3 * N - 6} is the upper bound)")

    # check 1 & 2: landmark graphs match their brute-force definitions
    assert edge_set(beta_skeleton(pts, 1.0, "lune")) == gabriel_bruteforce(pts)
    assert edge_set(beta_skeleton(pts, 1.0, "circle")) == gabriel_bruteforce(pts)
    assert edge_set(beta_skeleton(pts, 2.0, "lune")) == rng_bruteforce(pts)
    print("beta = 1 matches Gabriel, beta = 2 (lune) matches RNG")

    # checks 3 & 4: nesting in beta, and circle-based within lune-based
    betas = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    prev = {"lune": None, "circle": None}
    print(f"\n{'beta':>5} {'lune E':>7} {'conn':>5} {'circle E':>9} {'conn':>5} {'lune length':>12}")
    for b in betas:
        Gs = {k: beta_skeleton(pts, b, k) for k in ("lune", "circle")}
        for k, G in Gs.items():
            if prev[k] is not None:
                assert edge_set(G) <= prev[k], f"{k} not nested at beta={b}"
            prev[k] = edge_set(G)
        assert edge_set(Gs["circle"]) <= edge_set(Gs["lune"])
        length = Gs["lune"].size(weight="weight")
        print(f"{b:5.2f} {Gs['lune'].number_of_edges():7d} {str(nx.is_connected(Gs['lune'])):>5} "
              f"{Gs['circle'].number_of_edges():9d} {str(nx.is_connected(Gs['circle'])):>5} "
              f"{length:12.3f}")
    print("nesting checks passed")

    # pictures: lune skeleton at several beta (disconnected ones still drawn)
    show = [1.0, 1.5, 2.0, 3.0]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 4))
    for ax, b in zip(axes, show):
        G = beta_skeleton(pts, b, "lune")
        nx.draw_networkx_edges(G, nx.get_node_attributes(G, "pos"), ax=ax, width=0.8)
        ax.plot(pts[:, 0], pts[:, 1], "k.", ms=3)
        ax.set_title(f"lune beta = {b}: {G.number_of_edges()} edges")
        ax.set_aspect("equal")

    # preview of Q3(c): the same edge_flows works unchanged on a skeleton
    G = beta_skeleton(pts, 1.5, "lune")
    flows = edge_flows(G, uniform_demand(N))
    plot_flows(G, flows, title="lune beta = 1.5, uniform demand")
    plt.show()
