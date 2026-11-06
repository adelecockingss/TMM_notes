# TMM Lab 03 - Q10-Q14: assignment on big networks (link variables)
# quadprog is not available in Python (or in your MATLAB), so we use:
#  - gradient projection (path-based, exact) for single-OD problems, Q10-Q12
#  - Frank-Wolfe (link-based, the classic method) for many OD pairs, Q13-Q14
# Run cell-by-cell or: python q10_q14_big_networks.py  (~2 min)
# MATLAB version: q10_q14_big_networks.m

# %% Imports and data
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse.csgraph import dijkstra
import scipy.sparse as sp

from assign_utils import frank_wolfe, aon, two_way, gradient_projection

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)


def plot_arc_flows(ax, xy, arcs, x, title, vmax=None):
    """Two-way streets: add both directions' flow for the colour/width."""
    m = len(arcs) // 2
    f = x[:m] + x[m:] if len(x) == 2 * m and np.all(arcs[:m] == arcs[m:, ::-1]) else x
    E = arcs[:m] if len(f) == m else arcs
    vmax = vmax or max(f.max(), 1e-12)
    ax.add_collection(LineCollection(xy[E], colors="0.85", linewidths=0.5))
    used = f > 1e-6 * vmax
    lc = LineCollection(xy[E[used]], array=f[used], cmap="hot_r", clim=(0, vmax),
                        linewidths=0.5 + 4 * f[used] / vmax)
    ax.add_collection(lc); ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title)
    return lc


# %% Check Frank-Wolfe against the exact Q9 answer (two OD pairs, 1->4, 2->4)
arcs9 = np.array([[0, 2], [1, 2], [2, 3], [0, 3], [1, 3]])     # links 1..5
dem9 = np.zeros((4, 4)); dem9[0, 3] = dem9[1, 3] = 1
x9, gap, it = frank_wolfe(4, arcs9, [2, 2, 1, 4, 4], np.ones(5), dem9, "UE", tol=1e-10)
print(f"FW on Q9 network: {x9.round(4)} (exact 0.5 0.5 1 0.5 0.5), gap {gap:.1e}, {it} iters")
x9s, *_ = frank_wolfe(4, arcs9, [2, 2, 1, 4, 4], np.ones(5), dem9, "SO", tol=1e-10)
print(f"FW SO: {x9s.round(4)} (exact 0.375 0.375 0.75 0.625 0.625)")

# %% Q10 the beta-skeleton network from BigNetworkExperiments.mlx
E = pd.read_csv(HERE / "data" / "bignet_edges.csv")          # exported from the .mat
xy = np.loadtxt(HERE / "data" / "bignet_nodes.csv", delimiter=",")
edges = E[["s", "t"]].to_numpy() - 1                            # 0-based
n = len(xy)
o, dn = 0, n - 1                                                # node 1 -> node 100
# (a)/(b): free-flow cost a = link length, congestibility b = 1 for all.
# Every street is TWO-WAY, so it becomes two arcs (u->v and v->u), each with
# its own flow >= 0. (The .mlx builds B with ONE orientation per street and
# LB = 0, which silently makes every street one-way, low -> high node number.)
arcs = two_way(edges)
a = np.concatenate([E["length"], E["length"]]); b = np.ones(len(arcs))


def solve(d, mode, arcs=arcs, a=a, b=b):
    """Exact single-OD assignment: link flows, used routes, route flows."""
    return gradient_projection(n, arcs, a, b, o, dn, d, mode)


cost = lambda x, a=a, b=b: float(x @ (a + b * x))
for d in [0.001, 1.0]:
    xu, Ru, yu = solve(d, "UE"); xs, Rs, ys = solve(d, "SO")
    print(f"Q10 d = {d}: UE uses {len(Ru)} routes / {np.sum(xu > 1e-12)} arcs, system cost "
          f"{cost(xu):.5g}; SO cost {cost(xs):.5g}; PoA {cost(xu) / cost(xs):.4f}")

# Frank-Wolfe on the same problem: the classic link-based method, but slow
dem1 = np.zeros((n, n)); dem1[o, dn] = 1.0
xf, gf, itf = frank_wolfe(n, arcs, a, b, dem1, "UE", tol=1e-5, max_iter=2000)
print(f"   Frank-Wolfe after {itf} iterations: gap {gf:.0e}, max link-flow error "
      f"{np.abs(xf - solve(1.0, 'UE')[0]).max():.1e}")

# Compare with the one-way-only version implied by the .mlx's B matrix
xo, *_ = solve(1.0, "UE", arcs=edges, a=E["length"].to_numpy(), b=np.ones(len(edges)))
print(f"   (one orientation per street, as in the .mlx: system cost "
      f"{cost(xo, E['length'].to_numpy(), np.ones(len(edges))):.5g})")

# %% Q10(c) how the network fills up as demand grows
demands = [0.01, 0.3, 1, 3, 10]
fig, axs = plt.subplots(2, len(demands), figsize=(20, 8))
for k, d in enumerate(demands):
    for row, mode in enumerate(["UE", "SO"]):
        x = solve(d, mode)[0]
        lc = plot_arc_flows(axs[row, k], xy, arcs, x, f"{mode}, d = {d}: "
                            f"{np.sum(x[:len(edges)] + x[len(edges):] > 1e-6 * d)} streets used")
        axs[row, k].plot(*xy[[o, dn]].T, "bo")
fig.savefig(FIG / "q10c_fill_up.png", dpi=110, bbox_inches="tight")

# %% Q10(d) link costs at UE -> every used route is a shortest route
d = 3.0
xu, routes, yr = solve(d, "UE")
c = a + b * xu
W = sp.csr_matrix((c, (arcs[:, 0], arcs[:, 1])), shape=(n, n))
pi = dijkstra(W, indices=o)                     # shortest cost from O to every node
slack = pi[arcs[:, 0]] + c - pi[arcs[:, 1]]     # >= 0 for every arc
used = xu > 1e-6 * d
print(f"Q10(d) d = {d}: shortest O-D cost with UE weights = {pi[dn]:.5f}")
print(f"   used arcs: max slack {slack[used].max():.1e} (= 0: every used arc lies on a "
      f"shortest path); {len(routes)} used routes, costs from "
      f"{min(c[r].sum() for r in routes):.6f} to {max(c[r].sum() for r in routes):.6f}")
# So with weights a + b x(UE), ALL used routes are shortest paths and have the
# same cost - Wardrop's principle, read off from link variables alone.

# %% Q11 grid network (Lab 2's 10 x 10, L = 0.1), corner to corner
ng, L = 10, 0.1
kk = np.arange(ng * ng)
gxy = np.column_stack([L * (kk // ng), L * (kk % ng)])
gedges = np.array([(p, p + 1) for p in kk if p % ng < ng - 1] +
                  [(p, p + ng) for p in kk if p + ng < ng * ng])
garcs = two_way(gedges); ga = np.full(len(garcs), L); gb = np.ones(len(garcs))
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for ax, dd in zip(axs[:2], [0.05, 1.0]):
    x, Rg, yg = gradient_projection(ng * ng, garcs, ga, gb, 0, ng * ng - 1, dd, "UE")
    plot_arc_flows(ax, gxy, garcs, x, f"Q11 grid UE, d = {dd}: {len(Rg)} routes")
# On a grid the AON (d -> 0) answer is not unique (48,620 shortest paths);
# with congestion (b > 0) the UE LINK flows are unique and spread out
# symmetrically - most heavily near O and D, the bottleneck of Q12.

# %% Q12 spreading the origin/destination over an area (centroid connectors)
# Add a dummy 'zone' node joined by connector arcs to every grid node within
# 0.25 km of each corner. Connectors cost their length but never congest
# (b = 0), so traffic enters the street network at many points.
N2 = ng * ng + 2
zO, zD = ng * ng, ng * ng + 1
near_o = np.flatnonzero(np.hypot(*gxy.T) <= 0.25)
near_d = np.flatnonzero(np.hypot(*(gxy - gxy[-1]).T) <= 0.25)
con = np.array([(zO, v) for v in near_o] + [(v, zD) for v in near_d])
arcs12 = np.vstack([garcs, con])
len_o = np.hypot(*(gxy[near_o] - gxy[0]).T)       # connector length = distance
len_d = np.hypot(*(gxy[near_d] - gxy[-1]).T)      # from the corner (zone centre)
a12 = np.concatenate([ga, len_o, len_d])
b12 = np.concatenate([gb, np.zeros(len(con))])
x12, *_ = gradient_projection(N2, arcs12, a12, b12, zO, zD, 1.0, "UE")
plot_arc_flows(axs[2], gxy, garcs, x12[:len(garcs)], "Q12 with zone connectors, d = 1")
x_plain, *_ = gradient_projection(ng * ng, garcs, ga, gb, 0, ng * ng - 1, 1.0, "UE")
print(f"Q12: max street flow {x_plain.max():.3f} (single node) -> "
      f"{x12[:len(garcs)].max():.3f} (spread over {len(near_o)} nodes)")
fig.savefig(FIG / "q11_q12_grid.png", dpi=130, bbox_inches="tight")

# %% Q13 multiple OD pairs: all-to-all demand on a 6 x 6 grid
# In a QUADPROG formulation each destination j needs its own flow vector
# x^(j) with B x^(j) = s^(j) ("colouring"), total flow x = sum_j x^(j):
#   variables  = n_dest * n_arcs,   equality constraints = n_dest * n_nodes,
# with the constraint matrix block-diagonal: kron(I, B).
# Frank-Wolfe needs none of this - its AON step assigns whole paths.
n6 = 6; k6 = np.arange(n6 * n6)
xy6 = np.column_stack([k6 // n6, k6 % n6]) * 0.1
e6 = np.array([(p, p + 1) for p in k6 if p % n6 < n6 - 1] +
              [(p, p + n6) for p in k6 if p + n6 < n6 * n6])
arcs6 = two_way(e6); a6 = np.full(len(arcs6), 0.1); b6 = np.full(len(arcs6), 0.01)
dem6 = np.ones((n6 * n6, n6 * n6)) - np.eye(n6 * n6)
print(f"Q13: a stacked QP would have {n6 * n6 * len(arcs6)} variables and "
      f"{n6 * n6 * n6 * n6} equality constraints")
x6u, gu, iu = frank_wolfe(n6 * n6, arcs6, a6, b6, dem6, "UE", tol=1e-5, max_iter=5000)
x6s, gs, is_ = frank_wolfe(n6 * n6, arcs6, a6, b6, dem6, "SO", tol=1e-5, max_iter=5000)
c6 = lambda x: x @ (a6 + b6 * x)
print(f"   UE cost {c6(x6u):.2f}, SO cost {c6(x6s):.2f}, PoA {c6(x6u) / c6(x6s):.4f} "
      f"(gaps {gu:.0e}, {gs:.0e})")
# PoA = 1: all arcs have the same free-flow cost and only Manhattan-shortest
# routes are used, so every used route has the same sum(a). Then 'equal
# sum(b x)' (UE) and 'equal sum(2 b x)' (SO) are the same condition.

# %% Q14 a real map: Bristol junction graph from Lab 2 (if available)
lab2 = HERE.parents[1] / "lab2_NetworkModelling" / "answers" / "data"
if (lab2 / "bristol_edges.csv").exists():
    J = pd.read_csv(lab2 / "bristol_junctions.csv")
    Eb = pd.read_csv(lab2 / "bristol_edges.csv", keep_default_na=False)
    fwd = Eb["oneway"] != "-1"; bwd = ~Eb["oneway"].isin(["yes", "true", "1"])
    arcsb = np.vstack([Eb[["u", "v"]].to_numpy()[fwd], Eb[["v", "u"]].to_numpy()[bwd]])
    lenb = np.concatenate([Eb["length_km"][fwd], Eb["length_km"][bwd]])
    hw = np.concatenate([Eb["highway"][fwd], Eb["highway"][bwd]])
    speed = np.where(np.isin(hw, ["trunk", "primary", "trunk_link", "primary_link"]), 50, 30)
    lanes = np.where(np.isin(hw, ["trunk", "primary", "secondary"]), 2, 1)
    ab = lenb / speed * 60                         # free-flow minutes
    bb = ab / (600 * lanes)                        # doubles cost at 600 veh/h per lane
    xyb = J[["x_km", "y_km"]].to_numpy()
    s = int(np.argmin(np.hypot(*(xyb - [-1.2, -1.2]).T)))
    t = int(np.argmin(np.hypot(*(xyb - [1.2, 1.2]).T)))
    demb = np.zeros((len(J), len(J))); demb[s, t] = 3000      # veh/h
    xb, gap, it = frank_wolfe(len(J), arcsb, ab, bb, demb, "UE", tol=1e-4, max_iter=3000)
    x0, _ = aon(len(J), arcsb, ab, demb)
    print(f"Q14 Bristol: AON uses {np.sum(x0 > 0)} arcs; UE uses {np.sum(xb > 1)} arcs "
          f"(gap {gap:.0e} after {it} iters)")
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, xx, ttl in [(axs[0], x0, "AON (free flow)"), (axs[1], xb, "UE, 3000 veh/h")]:
        ax.add_collection(LineCollection(xyb[arcsb], colors="0.85", linewidths=0.4))
        u = xx > 1
        ax.add_collection(LineCollection(xyb[arcsb[u]], array=xx[u], cmap="hot_r",
                                         linewidths=0.5 + 4 * xx[u] / xx.max()))
        ax.plot(*xyb[[s, t]].T, "bo"); ax.autoscale(); ax.set_aspect("equal"); ax.set_title(ttl)
    fig.savefig(FIG / "q14_bristol_ue.png", dpi=110, bbox_inches="tight")
else:
    print("Q14: run lab2 answers/q5_q11_osm.py first to create the Bristol network")

plt.show()
