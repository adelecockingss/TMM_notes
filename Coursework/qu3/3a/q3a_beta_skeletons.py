# TMM Coursework Q3(a): beta-skeleton networks of two-way streets
# Run cell-by-cell or: python q3a_beta_skeletons.py
# Builds on ../q3_setup.py (node placement, lune beta-skeleton, two-way arcs).
#   1. checks the skeleton code against the textbook definitions of the Gabriel graph and RNG
#   2. exemplar plots varying N and beta (Gabriel beta = 1 -> RNG beta = 2)
#   3. network statistics vs beta, averaged over random seeds
#   4. fixes the single network used in parts (b) and (c) and saves it to data/

# %% Imports and helpers
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # so we can import ../q3_setup.py
from q3_setup import make_nodes, beta_skeleton, two_way, build_network, plot_network

FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)


def edge_set(G):
    return {tuple(sorted(e)) for e in G.edges}


# %% 1. Verification against brute-force definitions (all pairs, no Delaunay shortcut)
def gabriel_bruteforce(pts):
    """Gabriel graph: keep (p, q) iff no other point r lies strictly inside the circle with
    diameter pq, i.e. the angle prq is never > 90 degrees, i.e. (p - r).(q - r) < 0 for no r."""
    E = set()
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dots = np.einsum("kd,kd->k", pts[i] - pts, pts[j] - pts)
            dots[[i, j]] = np.inf
            if not (dots < 0).any():
                E.add((i, j))
    return E


def rng_bruteforce(pts):
    """Relative neighbourhood graph: keep (p, q) iff no r is closer to both p and q than they
    are to each other, i.e. max(|rp|, |rq|) < |pq| for no r."""
    D = squareform(pdist(pts))
    E = set()
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dmax = np.maximum(D[i], D[j])
            dmax[[i, j]] = np.inf
            if not (dmax < D[i, j]).any():
                E.add((i, j))
    return E


pts, O, D = make_nodes(300, rng=0)
assert edge_set(beta_skeleton(pts, 1.0)) == gabriel_bruteforce(pts)
assert edge_set(beta_skeleton(pts, 2.0)) == rng_bruteforce(pts)
print("check 1: beta = 1 skeleton == brute-force Gabriel graph, beta = 2 == brute-force RNG")

prev = None
for b in np.linspace(1, 2, 11):
    E = edge_set(beta_skeleton(pts, b))
    assert prev is None or E <= prev               # bigger beta -> bigger lune -> fewer streets
    prev = E
print("check 2: streets only ever disappear as beta increases (nested family)")

G = beta_skeleton(pts, 1.5)
A = two_way(G)
assert A.number_of_edges() == 2 * G.number_of_edges()
assert all(A.has_edge(v, u) and A[u][v]["length"] == A[v][u]["length"] for u, v in A.edges)
print("check 3: every street u-v becomes two arcs u->v and v->u of equal length")


# %% 2. Exemplar plots: rows vary N, columns vary beta (same points within each row)
Ns, betas = [50, 200, 1000], [1.0, 1.25, 1.5, 1.75, 2.0]
names = {1.0: "Gabriel", 2.0: "RNG"}
fig, axes = plt.subplots(len(Ns), len(betas), figsize=(3.2 * len(betas), 3.4 * len(Ns)))
for r, N in enumerate(Ns):
    for c, b in enumerate(betas):
        Gx, Ox, Dx = build_network(N, b, rng=r)
        ax = axes[r, c]
        plot_network(Gx, Ox, Dx, ax=ax,
                     title=f"N={N}, beta={b}" + (f" ({names[b]})" if b in names else "")
                           + f"\n{Gx.number_of_edges()} streets")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle("Q3(a) lune beta-skeletons on N random nodes + corners O=(0,0) (green), D=(1,1) (red)")
fig.tight_layout()
fig.savefig(FIG / "q3a_grid_N_beta.png", dpi=130)


# %% 3. Statistics vs beta: N = 200, averaged over 20 seeds
def stats(G, O, D):
    return {
        "streets": G.number_of_edges(),
        "mean degree": 2 * G.number_of_edges() / G.number_of_nodes(),
        "total length": G.size(weight="length"),
        "O-D detour": nx.shortest_path_length(G, O, D, weight="length") / np.sqrt(2),
        "deg O": G.degree[O], "deg D": G.degree[D],
    }

rows = []
for b in np.linspace(1, 2, 11):
    for seed in range(20):
        Gx, Ox, Dx = build_network(200, b, rng=seed)
        rows.append({"beta": b, "seed": seed, **stats(Gx, Ox, Dx)})
df = pd.DataFrame(rows)
summary = df.groupby("beta")[["streets", "mean degree", "total length", "O-D detour"]].mean()
print("\nN = 200, mean over 20 seeds:")
print(summary.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, col, lab in zip(axes, ["mean degree", "total length", "O-D detour"],
                        ["mean degree (streets per junction x 2 / nodes)", "total street length",
                         "shortest O-D route / sqrt(2)"]):
    g = df.groupby("beta")[col]
    ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), fmt="o-", capsize=3)
    ax.set_xlabel("beta"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
fig.suptitle("Q3(a) N = 200: fewer, shorter streets as beta grows, but longer O-D routes (mean +/- sd, 20 seeds)")
fig.tight_layout()
fig.savefig(FIG / "q3a_stats_vs_beta.png", dpi=130)


# %% 4. The network for parts (b) and (c): N = 200, beta = 1.5
# beta = 1.5 is mid-way between Gabriel and RNG and was the best fit to Bristol's streets in Lab 2 Q11.
# Pick the first seed where O and D each have >= 2 streets, so no single street is a forced bottleneck.
N_CHOSEN, BETA_CHOSEN = 200, 1.5
seed = next(s for s in range(100)
            if min(build_network(N_CHOSEN, BETA_CHOSEN, rng=s)[0].degree[n] for n in (N_CHOSEN, N_CHOSEN + 1)) >= 2)
G, O, D = build_network(N_CHOSEN, BETA_CHOSEN, rng=seed)
A = two_way(G)
s = stats(G, O, D)
print(f"\nchosen network: N = {N_CHOSEN}, beta = {BETA_CHOSEN}, seed = {seed}")
print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} streets -> {A.number_of_edges()} arcs, "
      f"total length {s['total length']:.3f}")
print(f"  deg(O) = {s['deg O']}, deg(D) = {s['deg D']}, shortest O-D = {s['O-D detour'] * np.sqrt(2):.4f}")

pos = nx.get_node_attributes(G, "pos")
pd.DataFrame([(k, *pos[k]) for k in G.nodes], columns=["node", "x", "y"]).to_csv(DATA / "nodes.csv", index=False)
pd.DataFrame([(u, v, A[u][v]["length"]) for u, v in A.edges], columns=["from", "to", "length"]) \
  .to_csv(DATA / "arcs.csv", index=False)
print(f"  saved data/nodes.csv and data/arcs.csv (O = node {O}, D = node {D})")

fig, ax = plt.subplots(figsize=(6.5, 6.8))
plot_network(G, O, D, ax=ax,
             title=f"Q3 network for (b), (c): N = {N_CHOSEN}, beta = {BETA_CHOSEN}, "
                   f"{G.number_of_edges()} two-way streets")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "q3a_chosen_network.png", dpi=150, bbox_inches="tight")
plt.show()
