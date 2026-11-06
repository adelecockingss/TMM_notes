# TMM Lab 01 - Q8: intervening opportunities and radiation models
# Run cell-by-cell or: python q8_other_models.py  (run q3_od_matrix.py first)
# MATLAB version: q8_other_models.m
#
# Both models say distance matters only through the number of OPPORTUNITIES
# (here: jobs, D_k) a commuter passes on the way. s_ij = jobs in zones closer
# to i than j is (excluding i and j). Both are origin-constrained by design:
# row i shares out O_i.

# %% Imports and data
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from gravity_utils import load_level, cpc, fit_report, furness, opportunities_matrix

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

LEVEL = "bristol"
z = load_level(LEVEL)
T, O, D, dist = z["T"], z["O"], z["D"], z["dist"]
n = len(O)
s = opportunities_matrix(D, dist)
offdiag = ~np.eye(n, dtype=bool)


# %% Intervening opportunities (Stouffer / Schneider)
# Each opportunity passed is accepted with probability L. The probability of
# NOT stopping before the opportunities V is exp(-L V), so the chance of
# stopping in zone j (which has D_j opportunities after V_ij intervening ones):
#     P_ij = exp(-L V_ij) - exp(-L (V_ij + D_j)),   V_ij = D_i + s_ij  (j != i)
#     P_ii = 1 - exp(-L D_i)                        (stop in your own zone)
# Normalise each row so that sum_j T_ij = O_i.  One free parameter: L.
# (Anders's slide writes the same thing with (1 - alpha)^V instead of exp(-L V).)
def intervening_opportunities(L):
    V = s + D[:, None]                    # opportunities passed before reaching j
    np.fill_diagonal(V, 0)
    P = np.exp(-L * V) - np.exp(-L * (V + D[None, :]))
    P /= P.sum(1, keepdims=True)
    return O[:, None] * P


res = minimize_scalar(lambda logL: -cpc(intervening_opportunities(10**logL), T),
                      bounds=(-8, -2), method="bounded")
L_best = 10 ** res.x
T_io = intervening_opportunities(L_best)
print(f"Intervening opportunities: best L = {L_best:.2e} per job")


# %% Radiation model (Simini et al. 2012) - parameter-free
#     T_ij = O_i * m_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij)) / (1 - m_i / M)
# m_i = "population" at origin, n_j at destination. Here: m = O (workers),
# n = D (jobs). It predicts no intra-zonal trips, so we compare off-diagonal.
m_, n_ = O, D
Trad = O[:, None] * (m_[:, None] * n_[None, :]) / (
    (m_[:, None] + s) * (m_[:, None] + n_[None, :] + s))
np.fill_diagonal(Trad, 0)
# Instead of the (1 - m_i/M) finite-size correction, simply rescale each row
# to the observed number of trips LEAVING zone i (the radiation model has no
# intra-zonal trips), so the comparison with the other models is fair.
T_out = T * offdiag
Trad *= (T_out.sum(1) / np.maximum(Trad.sum(1), 1e-12))[:, None]


# %% Compare all models on the off-diagonal (between-zone) trips
def offd(M):
    return M * offdiag


_, _, T_dbl, _ = furness(O, D, dist**-1.4)              # best doubly constr. from Q7
_, _, T_dbl2, _ = furness(O, D, dist**-2.0)
print("\nOff-diagonal comparison (trips between different zones):")
for name, Tm in [("doubly constrained d^-2", T_dbl2), ("doubly constrained d^-1.4", T_dbl),
                 ("intervening opportunities", T_io), ("radiation (no parameters)", Trad)]:
    fit_report(name, offd(Tm), T_out)

# %% Plots: model vs observed, and trip length distributions
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
mask = T_out > 0
for ax, (name, Tm) in zip(axs[:2], [("intervening opp.", T_io), ("radiation", Trad)]):
    ax.loglog(T_out[mask], Tm[mask], ".", ms=1, alpha=0.3)
    ax.plot([1, T_out.max()], [1, T_out.max()], "k--")
    ax.set_xlabel("observed"); ax.set_ylabel("model"); ax.set_title(name)
bins = np.arange(0, 60, 2)
for Tm, lab in [(T_out, "observed"), (offd(T_dbl), "gravity d^-1.4"),
                (offd(T_io), "intervening opp."), (Trad, "radiation")]:
    axs[2].hist(dist.ravel(), bins=bins, weights=Tm.ravel(), histtype="step", lw=1.5,
                label=lab)
axs[2].set_xlabel("distance (km)"); axs[2].set_ylabel("trips"); axs[2].legend()
fig.savefig(FIG / f"q8_other_models_{LEVEL}.png", dpi=150, bbox_inches="tight")

plt.show()
