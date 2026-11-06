"""
TMM Lab 01 - shared trip-distribution helpers for Q4-Q8.

    load_level(level)            -> T, O, D, dist, labels for 'bristol', 'utla' or 'all'
    fit_c(T, X, method)          -> scale constant c for T ~ c X
    furness(O, D, F)             -> doubly constrained balancing factors
    cpc(Tm, Te)                  -> Sorensen index / common part of commuters
    opportunities_matrix(D, d)   -> s_ij for intervening-opportunities & radiation

Run q3_od_matrix.py first (it creates data/od_msoa*.npz and data/od_utla.npz).
MATLAB equivalents: load_od_level.m, furness.m, cpc.m, opportunities_matrix.m
"""
from pathlib import Path
import numpy as np
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent / "data"
BRISTOL_EN = (358_800, 173_000)          # Bristol city centre, OS grid metres


def intrazonal_distance(dist):
    """Distance to use for trips that start and end in the same zone.
    The centroid-to-centroid distance is 0, which would make 1/d^2 infinite.
    A standard fix: half the distance to the nearest neighbouring centroid
    (roughly the average trip length inside the zone)."""
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    out = dist.copy()
    np.fill_diagonal(out, 0.5 * nearest)
    return out


def load_level(level="bristol", radius_km=50, dtype=np.float64):
    """Return a dict with dense T (n x n), O, D, dist (km, diagonal filled), labels.

    'bristol': MSOAs within radius_km of Bristol, keeping only flows INSIDE the
               region (so O and D are recomputed from the internal flows - the
               region is treated as a closed system).
    'utla'   : 174 Upper Tier Local Authorities, all of England & Wales.
    'all'    : all 7264 MSOAs - needs ~3 GB RAM (use dtype=np.float32).
    """
    if level == "utla":
        z = np.load(DATA / "od_utla.npz", allow_pickle=True)
        T = z["T"].astype(dtype)
        dist = intrazonal_distance(z["dist"].astype(dtype))
        return dict(T=T, O=T.sum(1), D=T.sum(0), dist=dist, labels=z["labels"],
                    E=z["E"], N=z["N"])

    z = np.load(DATA / "od_msoa.npz", allow_pickle=True)
    Tsp = sp.load_npz(DATA / "od_msoa_T.npz")
    E, N, labels = z["E"], z["N"], z["labels"]
    if level == "bristol":
        keep = np.hypot(E - BRISTOL_EN[0], N - BRISTOL_EN[1]) <= radius_km * 1000
        idx = np.flatnonzero(keep)
        Tsp = Tsp[idx][:, idx]
        E, N, labels = E[idx], N[idx], labels[idx]
    elif level != "all":
        raise ValueError(level)
    T = Tsp.toarray().astype(dtype)
    Ek, Nk = E.astype(dtype) / 1000, N.astype(dtype) / 1000
    dist = np.hypot(Ek[:, None] - Ek[None, :], Nk[:, None] - Nk[None, :])
    dist = intrazonal_distance(dist)
    return dict(T=T, O=T.sum(1), D=T.sum(0), dist=dist, labels=labels, E=E, N=N)


# ---------------------------------------------------------------------------
# Fitting the constant c in  T_ij ~ c * X_ij   (X_ij = O_i D_j f(d_ij))
# ---------------------------------------------------------------------------
def fit_c(T, X, method="poisson"):
    """Three ways to choose c - they give very different answers because
    T has a huge range (lots of zeros and small numbers, a few huge ones).

    'ols'    : least squares through the origin, min sum (T - cX)^2
               => c = sum(X T) / sum(X^2).  Dominated by the few biggest X.
    'log'    : least squares on logs of the NON-ZERO entries,
               min sum (log T - log c - log X)^2 => log c = mean(log T - log X).
               Every non-zero pair counts equally; zeros are ignored.
    'poisson': maximum likelihood assuming T_ij ~ Poisson(c X_ij)
               => c = sum(T) / sum(X).  Uses the zeros, makes the model total
               equal the observed total; the standard choice for count data.
    """
    if method == "ols":
        return (X * T).sum() / (X * X).sum()
    if method == "log":
        m = T > 0
        return np.exp(np.mean(np.log(T[m]) - np.log(X[m])))
    if method == "poisson":
        return T.sum() / X.sum()
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------
def cpc(Tm, Te):
    """Sorensen index = common part of commuters:
        CPC = 2 sum min(Tm, Te) / (sum Tm + sum Te)
    0 = no overlap, 1 = perfect. Insensitive to the scale of a few big flows."""
    return 2 * np.minimum(Tm, Te).sum() / (Tm.sum() + Te.sum())


def fit_report(name, Tm, Te):
    r = np.corrcoef(Tm.ravel(), Te.ravel())[0, 1]
    rmse = np.sqrt(np.mean((Tm - Te) ** 2))
    print(f"{name:32s} CPC = {cpc(Tm, Te):.3f}   r = {r:.3f}   RMSE = {rmse:.1f}")


# ---------------------------------------------------------------------------
# Doubly constrained model: Furness / iterative proportional fitting
# ---------------------------------------------------------------------------
def furness(O, D, F, tol=1e-8, max_iter=1000, verbose=False):
    """Find a_i, b_j such that T_ij = a_i O_i b_j D_j F_ij has row sums O and
    column sums D (F_ij = f(d_ij), the deterrence matrix).

    The 2n equations
        sum_j a_i O_i b_j D_j F_ij = O_i   ->  a_i = 1 / sum_j b_j D_j F_ij
        sum_i a_i O_i b_j D_j F_ij = D_j   ->  b_j = 1 / sum_i a_i O_i F_ij
    are nonlinear together but each is trivial given the other, so alternate
    (this is also called RAS, IPF or RIM weighting in survey statistics).
    Note: (a, b) -> (k a, b / k) gives the same T, so the solution is only
    unique up to that scaling."""
    b = np.ones_like(D, dtype=float)
    for it in range(max_iter):
        a = 1.0 / np.maximum(F @ (b * D), 1e-300)
        b = 1.0 / np.maximum(F.T @ (a * O), 1e-300)
        T = (a * O)[:, None] * F * (b * D)[None, :]
        err = np.abs(T.sum(1) - O).max() / O.max()
        if verbose and it % 10 == 0:
            print(f"  iter {it:4d}: max row-sum error {err:.2e}")
        if err < tol:
            break
    return a, b, T, it + 1


# ---------------------------------------------------------------------------
# Intervening opportunities / radiation
# ---------------------------------------------------------------------------
def opportunities_matrix(D, dist):
    """s_ij = total opportunities (here: jobs D_k) in zones k strictly closer
    to i than j is, excluding i and j themselves.
    For each origin i: sort zones by distance from i, take a cumulative sum.
    Zone i is always first (its diagonal distance is the smallest)."""
    n = len(D)
    s = np.empty((n, n), dtype=dist.dtype)
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")
        Ds = D[order]
        before = np.cumsum(Ds) - Ds              # sum of everything earlier in the order
        s_row = np.empty(n, dtype=dist.dtype)
        s_row[order] = before - D[i]             # exclude the origin zone itself
        s_row[i] = 0
        s[i] = s_row
    return s
