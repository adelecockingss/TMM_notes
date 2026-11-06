"""
TMM Lab 03 - traffic assignment solvers (numpy/scipy only).

    parallel_affine(a, b, d, mode)   exact UE/SO on parallel links, costs a + b x
    parallel_general(costs, d, mode) UE/SO on parallel links, any increasing costs
    qp_small(H, f, Aeq, beq)         min 1/2 x'Hx + f'x, Aeq x = beq, x >= 0, by
                                     enumerating supports (= complementarity method)
    aon(n, arcs, cost, demand)       all-or-nothing assignment on shortest paths
    frank_wolfe(n, arcs, a, b, demand, mode)   UE/SO on a whole network, any OD matrix
    gradient_projection(n, arcs, a, b, o, d_node, d, mode)   exact single-OD UE/SO

MATLAB equivalents: parallel_affine.m, qp_small.m, aon_flows.m, frank_wolfe.m,
gradient_projection.m
(the MATLAB scripts use quadprog instead of qp_small if you have the
Optimization Toolbox).

Cost conventions: link cost c_i(x_i) = a_i + b_i x_i.
  System cost      f(x)  = sum x_i c_i(x_i)            -> SO minimises this
  Beckmann         f^(x) = sum int_0^{x_i} c_i(s) ds   -> UE minimises this
For affine costs both are quadratics:  a'x + theta * x' diag(b) x  with
theta = 1 (SO) or 1/2 (UE). In quadprog form 1/2 x'Hx + f'x: H = 2 theta diag(b).
"""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


# ---------------------------------------------------------------------------
# Parallel links
# ---------------------------------------------------------------------------
def parallel_affine(a, b, d, mode="UE"):
    """Exact UE or SO flows on n parallel links with costs a_i + b_i x_i.

    UE (Wardrop): every used link has the same cost C, unused links cost >= C.
    With cost a_i + b_i x_i, link i carries x_i(C) = max(0, (C - a_i)/b_i);
    find the level C with sum_i x_i(C) = d ('water-filling').
    SO: the same with MARGINAL cost  d/dx [x c(x)] = a_i + 2 b_i x_i,
    i.e. just replace b by 2b.
    Links with b_i = 0 (constant cost) can absorb any amount at cost a_i."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if mode == "SO":
        b = 2 * b
    n = len(a)
    x = np.zeros(n)
    const = b == 0
    var = ~const

    def S(C):  # total flow the congestible links take at cost level C
        return np.sum(np.maximum(0, (C - a[var]) / b[var])) if var.any() else 0.0

    C_const = a[const].min() if const.any() else np.inf
    if S(C_const) >= d:
        # all flow on congestible links: S is piecewise linear increasing in C.
        # Sort breakpoints a_i and solve exactly on the right piece.
        order = np.argsort(a[var])
        av, bv = a[var][order], b[var][order]
        for k in range(1, len(av) + 1):
            # use the k cheapest links: sum (C - a_i)/b_i = d
            C = (d + np.sum(av[:k] / bv[:k])) / np.sum(1 / bv[:k])
            if k == len(av) or C <= av[k]:
                break
        x[var] = np.maximum(0, (C - a[var]) / b[var])
    else:
        C = C_const
        x[var] = np.maximum(0, (C - a[var]) / b[var])
        x[np.flatnonzero(const & (a == C_const))[0]] = d - x[var].sum()
    return x


def parallel_general(cost, dcost, d, mode="UE", tol=1e-12):
    """UE/SO on parallel links with any strictly increasing costs (Q5).
    cost, dcost: lists of functions c_i(x) and c_i'(x).
    UE equalises c_i, SO equalises marginal cost m_i(x) = c_i(x) + x c_i'(x).
    Nested bisection: inner finds x_i(C) = largest x in [0, d] with m_i(x) <= C,
    outer finds the level C with sum x_i(C) = d."""
    m = cost if mode == "UE" else [lambda x, c=c, dc=dc: c(x) + x * dc(x)
                                   for c, dc in zip(cost, dcost)]

    def x_of_C(mi, C):
        if mi(0) >= C:
            return 0.0
        if mi(d) <= C:
            return d
        lo, hi = 0.0, d
        while hi - lo > tol * max(d, 1):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if mi(mid) <= C else (lo, mid)
        return 0.5 * (lo + hi)

    lo = min(mi(0) for mi in m); hi = max(mi(d) for mi in m)
    for _ in range(200):
        C = 0.5 * (lo + hi)
        if sum(x_of_C(mi, C) for mi in m) < d:
            lo = C
        else:
            hi = C
    # Flow just below and just above the level C. Strictly increasing links
    # agree; a constant-cost link jumps from 0 to d at C. Give the remaining
    # demand to the links that jump.
    xl = np.array([x_of_C(mi, lo) for mi in m])
    xh = np.array([x_of_C(mi, hi) for mi in m])
    jump = xh - xl
    rem = d - xl.sum()
    return xl + rem * jump / jump.sum() if jump.sum() > 0 else xl


def system_cost(x, a, b):
    return float(np.sum(x * (a + b * x)))


# ---------------------------------------------------------------------------
# Small quadratic programmes by the complementarity method
# ---------------------------------------------------------------------------
def qp_small(H, f, Aeq, beq, tol=1e-9):
    """min 1/2 x'Hx + f'x  s.t.  Aeq x = beq, x >= 0   (H positive semi-definite)

    KKT conditions: Hx + f = Aeq' mu + nu,  nu >= 0,  x >= 0,  nu_i x_i = 0.
    Guess the support S = {i : x_i > 0} (the 'used' links/routes). Then nu_S = 0
    and [H_SS  -Aeq_S'; Aeq_S  0] [x_S; mu] = [-f_S; beq] is a LINEAR system.
    Accept the guess if x_S >= 0 and nu_rest = (Hx + f - Aeq'mu)_rest >= 0.
    This is exactly the lecture's complementarity method, automated. It tries
    2^n supports, so only use it for small n (<= ~15)."""
    H = np.atleast_2d(np.asarray(H, float)); f = np.asarray(f, float).ravel()
    Aeq = np.atleast_2d(np.asarray(Aeq, float)); beq = np.asarray(beq, float).ravel()
    n, m = len(f), len(beq)
    best = None
    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            S = list(S)
            K = np.block([[H[np.ix_(S, S)], -Aeq[:, S].T],
                          [Aeq[:, S], np.zeros((m, m))]])
            rhs = np.concatenate([-f[S], beq])
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)     # min-norm if singular
            if np.linalg.norm(K @ sol - rhs) > 1e-7 * max(1, np.linalg.norm(rhs)):
                continue                                       # inconsistent
            x = np.zeros(n); x[S] = sol[:k]; mu = sol[k:]
            nu = H @ x + f - Aeq.T @ mu
            if x.min() >= -tol and nu.min() >= -1e-7:
                val = 0.5 * x @ H @ x + f @ x
                if best is None or val < best[1] - 1e-12:
                    best = (np.maximum(x, 0), val)
        if best is not None:
            return best          # smallest support that satisfies KKT
    raise RuntimeError("no KKT point found")


# ---------------------------------------------------------------------------
# Network assignment: all-or-nothing and Frank-Wolfe
# ---------------------------------------------------------------------------
def aon(n, arcs, cost, demand):
    """All-or-nothing: put every OD demand on its current shortest path.
    arcs: (m, 2) DIRECTED arcs u -> v (a two-way street = two arcs).
    demand: (n, n) OD matrix. Returns arc flows (m,) and the shortest-path
    cost matrix rows for the origins used (for the gap computation)."""
    m = len(arcs)
    W = sp.csr_matrix((cost + 1e-15, (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    EID = sp.csr_matrix((np.arange(1, m + 1), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    origins = np.flatnonzero(demand.sum(1) > 0)
    dist, pred = dijkstra(W, directed=True, indices=origins, return_predecessors=True)
    y = np.zeros(m)
    sptt = 0.0                      # shortest-path travel time, sum_od d_od * dist
    for r, s in enumerate(origins):
        load = demand[s].astype(float).copy()
        fin = np.isfinite(dist[r])                 # skip unreachable nodes
        sptt += np.sum(load[fin] * dist[r][fin])
        # id of the tree arc pred[v] -> v for every reachable v (one sparse lookup)
        v_all = np.flatnonzero(pred[r] >= 0)
        tree_arc = np.zeros(n, int)
        tree_arc[v_all] = np.asarray(EID[pred[r, v_all], v_all]).ravel() - 1
        for v in np.argsort(-dist[r]):             # furthest first
            p = pred[r, v]
            if p < 0 or load[v] == 0:
                continue
            y[tree_arc[v]] += load[v]
            load[p] += load[v]
    return y, sptt


def frank_wolfe(n, arcs, a, b, demand, mode="UE", tol=1e-6, max_iter=5000, verbose=False):
    """Frank-Wolfe (convex combinations) algorithm for affine link costs.

    Repeat:
      1. costs at current flows: c = a + b x (UE) or a + 2 b x (SO, marginal)
      2. all-or-nothing load y onto shortest paths for those costs
      3. move x <- x + lam (y - x), lam in [0, 1] minimising the objective
         (exact for quadratics: lam = -grad.(y - x) / (theta-weighted curvature))
    Stop when the relative gap (c.x - c.y)/(c.x) < tol: at a UE every used
    path is a shortest path, so c.x = c.y exactly.
    Handles any number of OD pairs - no 'colouring' needed, because the
    AON step works with whole paths."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    k = 1.0 if mode == "UE" else 2.0
    x, _ = aon(n, arcs, a, demand)
    gap = np.inf
    for it in range(1, max_iter + 1):
        c = a + k * b * x
        y, sptt = aon(n, arcs, c, demand)
        gap = (c @ x - sptt) / max(c @ x, 1e-15)
        if gap < tol:
            break
        dvec = y - x
        curv = k * np.sum(b * dvec**2)
        lam = 1.0 if curv <= 0 else np.clip(-(c @ dvec) / curv, 0, 1)
        x = x + lam * dvec
        if verbose and it % 200 == 0:
            print(f"  FW iter {it}: gap {gap:.2e}")
    return x, gap, it


def gradient_projection(n, arcs, a, b, origin, dest, d, mode="UE", tol=1e-12, max_iter=10000):
    """Exact single-OD UE/SO by path-based gradient projection (affine costs).

    Keep a set of routes P with flows y (route-link incidence columns).
    Repeat:
      1. link costs c = a + k b x  (k = 1 UE, 2 SO: marginal cost)
      2. add the current shortest route to P if it is new
      3. move flow from each more expensive used route p to the cheapest
         route q. For affine costs the exact step that equalises them is
            delta = (c_p - c_q) / (k * sum of b over links in p but not q, or q not p)
         (capped at y_p)
    until all used routes cost the same as the shortest route (Wardrop).
    Returns link flows x, route list (as arc-index arrays) and route flows y."""
    a = np.asarray(a, float); b = np.asarray(b, float); m = len(arcs)
    k = 1.0 if mode == "UE" else 2.0

    def shortest_route(c):
        W = sp.csr_matrix((c + 1e-15, (arcs[:, 0], arcs[:, 1])), shape=(n, n))
        eid = sp.csr_matrix((np.arange(1, m + 1), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
        dist, pred = dijkstra(W, indices=origin, return_predecessors=True)
        route, v = [], dest
        while v != origin:
            u = pred[v]; route.append(int(eid[u, v]) - 1); v = u
        return np.zeros(m, bool) | np.isin(np.arange(m), route), dist[dest]

    R, _ = shortest_route(a)
    routes, y = [R], [float(d)]
    x = d * R.astype(float)                  # link flows, updated incrementally
    for it in range(max_iter):
        c = a + k * b * x
        R, cmin = shortest_route(c)
        if not any(np.array_equal(R, r) for r in routes):
            routes.append(R); y.append(0.0)
        cr = np.array([c[r].sum() for r in routes])
        used = np.array(y) > 0
        if cr[used].max() - cmin <= tol * max(cmin, 1):
            break
        q = int(np.argmin(cr))
        for p in np.flatnonzero(used):
            if p == q:
                continue
            c = a + k * b * x
            gap = c[routes[p]].sum() - c[routes[q]].sum()
            sym = routes[p] ^ routes[q]                  # links in one route only
            curv = k * b[sym].sum()
            delta = y[p] if curv <= 0 else min(y[p], gap / curv)
            if delta > 0:
                y[p] -= delta; y[q] += delta
                x += delta * (routes[q].astype(float) - routes[p])
    keep = np.array(y) > 1e-14 * d
    return x, [np.flatnonzero(r) for r, kp in zip(routes, keep) if kp], np.array(y)[keep]


def two_way(edges):
    """Undirected street list (m, 2) -> directed arcs (2m, 2): u->v then v->u."""
    edges = np.asarray(edges)
    return np.vstack([edges, edges[:, ::-1]])
