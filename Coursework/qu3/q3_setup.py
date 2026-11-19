# TMM Coursework Q3: basic set-up
# N nodes uniformly random in (0,1) x (0,1), plus corner nodes (0,0) = origin and (1,1) = destination,
# joined by a lune-based beta-skeleton of two-way streets (beta = 1 Gabriel ... beta = 2 RNG).
# Run cell-by-cell or: python q3_setup.py
# The skeleton code is the Lab 2 Q3 method: Delaunay candidate edges, then the forbidden-lune test.

# %% Imports and helpers
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)


# %% Node placement
def make_nodes(N, rng=None):
    """N uniform random nodes in the open unit square, then O = (0,0) and D = (1,1).
    Returns (pts, O, D): pts is (N+2, 2); O = N and D = N+1 are the node indices."""
    rng = np.random.default_rng(rng)
    interior = rng.uniform(0, 1, size=(N, 2))      # uniform on [0,1); hitting 0 exactly has prob ~0
    pts = np.vstack([interior, [0.0, 0.0], [1.0, 1.0]])
    return pts, N, N + 1


# %% Beta-skeleton (lune-based, 1 <= beta <= 2)
def delaunay_edges(pts):
    """Unique undirected Delaunay edges as an (E, 2) int array with i < j.
    For beta >= 1 every skeleton edge is a Delaunay edge, so these are the only candidates."""
    s = Delaunay(pts).simplices
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def lune_blocked(pts, i, j, beta):
    """True if a node other than i, j lies strictly inside the lune of (i, j):
    the intersection of two discs of radius beta*|pq|/2 centred on the line pq."""
    p, q = pts[i], pts[j]
    R = beta * np.linalg.norm(q - p) / 2
    c1 = (1 - beta / 2) * p + (beta / 2) * q
    c2 = (beta / 2) * p + (1 - beta / 2) * q
    inside = (np.linalg.norm(pts - c1, axis=1) < R) & (np.linalg.norm(pts - c2, axis=1) < R)
    inside[[i, j]] = False
    return inside.any()


def beta_skeleton(pts, beta):
    """Undirected street network: nodes carry 'pos', edges carry 'length' (Euclidean)."""
    if not 1 <= beta <= 2:
        raise ValueError("use 1 <= beta <= 2 (Gabriel to RNG); outside this the graph may disconnect")
    G = nx.Graph()
    for k, xy in enumerate(pts):
        G.add_node(k, pos=tuple(xy))
    for i, j in delaunay_edges(pts):
        if not lune_blocked(pts, i, j, beta):
            G.add_edge(int(i), int(j), length=float(np.linalg.norm(pts[i] - pts[j])))
    return G


def two_way(G):
    """Each undirected street (u, v) -> two arcs u->v and v->u with the same length.
    Needed for assignment: with flows x >= 0, a single arc only carries traffic one way."""
    return G.to_directed()


def build_network(N, beta, rng=None):
    """The full Q3 set-up: returns (G, O, D) with G the undirected skeleton."""
    pts, O, D = make_nodes(N, rng)
    G = beta_skeleton(pts, beta)
    # RNG (beta = 2) contains the minimum spanning tree, so every 1 <= beta <= 2 skeleton is connected
    assert nx.is_connected(G), "skeleton should be connected for beta <= 2"
    return G, O, D


# %% Plotting
def plot_network(G, O, D, ax=None, title=None):
    ax = ax or plt.gca()
    pos = nx.get_node_attributes(G, "pos")
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.7, edge_color="0.35")
    xy = np.array([pos[k] for k in G.nodes if k not in (O, D)])
    ax.plot(xy[:, 0], xy[:, 1], ".", color="0.2", ms=3)
    ax.plot(*pos[O], "o", color="tab:green", ms=8, label="origin (0,0)")
    ax.plot(*pos[D], "s", color="tab:red", ms=8, label="destination (1,1)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)   # networkx hides these
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title or f"{G.number_of_nodes()} nodes, {G.number_of_edges()} streets")
    return ax


# %% Run: build one network and check it
if __name__ == "__main__":
    N, beta, seed = 100, 1.5, 1
    G, O, D = build_network(N, beta, rng=seed)
    A = two_way(G)
    pos = nx.get_node_attributes(G, "pos")

    print(f"N = {N} random + 2 corners = {G.number_of_nodes()} nodes, beta = {beta}")
    print(f"{G.number_of_edges()} two-way streets -> {A.number_of_edges()} directed arcs")
    print(f"total street length = {G.size(weight='length'):.3f}")
    print(f"O = node {O} at {pos[O]}, degree {G.degree[O]};  D = node {D} at {pos[D]}, degree {G.degree[D]}")
    L = nx.shortest_path_length(G, O, D, weight="length")
    print(f"shortest O->D route length = {L:.4f}  (straight line sqrt(2) = {np.sqrt(2):.4f}, ratio {L / np.sqrt(2):.3f})")

    plot_network(G, O, D, title=f"Q3 set-up: N = {N}, lune beta = {beta}, {G.number_of_edges()} streets")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "q3_setup.png", dpi=150)
    plt.show()
