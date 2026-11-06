# TMM Lab 01 - Q6 & Q7: singly and doubly constrained gravity models
# Run cell-by-cell or: python q6_q7_constrained.py  (run q3_od_matrix.py first)
# MATLAB version: q6_q7_constrained.m

# %% Imports and data
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from gravity_utils import load_level, fit_c, cpc, fit_report, furness

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

LEVEL = "bristol"   # 'bristol', 'utla' or 'all'
z = load_level(LEVEL)
T, O, D, dist = z["T"], z["O"], z["D"], z["dist"]
F = 1.0 / dist**2                          # classic deterrence f(d) = 1/d^2

# %% Q6(a) unconstrained model: row/column sums are wrong
c = fit_c(T, np.outer(O, D) * F, "poisson")
T_un = c * np.outer(O, D) * F
row_err = T_un.sum(1) / O - 1
col_err = T_un.sum(0) / D - 1
print(f"Unconstrained: row sums off by {np.abs(row_err).mean():.0%} on average "
      f"(range {row_err.min():+.0%} to {row_err.max():+.0%})")
print(f"               col sums off by {np.abs(col_err).mean():.0%} on average")

# %% Q6(b) singly (origin) constrained: scale each row by A_i
# A_i = O_i / sum_j (c O_i D_j f_ij)  so that  sum_j A_i T_un_ij = O_i exactly.
A = O / T_un.sum(1)
T_oc = A[:, None] * T_un
print("Origin-constrained: max row error =", np.abs(T_oc.sum(1) - O).max().round(6),
      f"| column sums off by {np.abs(T_oc.sum(0) / D - 1).mean():.0%} on average")
# (c cancels: T_oc_ij = O_i D_j f_ij / sum_k D_k f_ik  - the usual form)

# Destination-constrained, the other way round
B = D / T_un.sum(0)
T_dc = T_un * B[None, :]
print("Destination-constrained: max col error =", np.abs(T_dc.sum(0) - D).max().round(6),
      f"| row sums off by {np.abs(T_dc.sum(1) / O - 1).mean():.0%} on average")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].loglog(O, T_un.sum(1), ".", label="unconstrained")
axs[0].loglog(O, T_oc.sum(1), ".", label="origin-constrained")
axs[0].plot([O.min(), O.max()], [O.min(), O.max()], "k--")
axs[0].set_xlabel("observed O_i"); axs[0].set_ylabel("model row sum"); axs[0].legend()
axs[1].loglog(D, T_un.sum(0), ".", label="unconstrained")
axs[1].loglog(D, T_oc.sum(0), ".", label="origin-constrained")
axs[1].plot([D.min(), D.max()], [D.min(), D.max()], "k--")
axs[1].set_xlabel("observed D_j"); axs[1].set_ylabel("model column sum"); axs[1].legend()
fig.savefig(FIG / f"q6_row_col_sums_{LEVEL}.png", dpi=150, bbox_inches="tight")

# %% Q7(a) doubly constrained: solve for a_i, b_j by Furness iteration
a, b, T_dbl, iters = furness(O, D, F, verbose=True)
print(f"Furness converged in {iters} iterations")

# %% Q7(b) verify the row and column sums
print("max |row sum - O| =", np.abs(T_dbl.sum(1) - O).max())
print("max |col sum - D| =", np.abs(T_dbl.sum(0) - D).max())

# %% Q7(a) alternative: treat it as a nonlinear system and use a solver
# Unknowns x = log(a), log(b) (logs keep them positive). Residuals:
#   r_i = sum_j T_ij - O_i ,  r_j = sum_i T_ij - D_j   (2n equations)
# One equation is redundant (sum O = sum D) and (a, b) -> (ka, b/k) is a
# symmetry, so use least squares rather than a square solver.
from scipy.optimize import least_squares
n = len(O)
def residuals(x):
    Tm = (np.exp(x[:n]) * O)[:, None] * F * (np.exp(x[n:]) * D)[None, :]
    return np.concatenate([(Tm.sum(1) - O) / O.max(), (Tm.sum(0) - D) / D.max()])
x0 = np.concatenate([np.log(a * 1.5), np.log(b)])   # deliberately perturbed start
sol = least_squares(residuals, x0, method="lm")
print(f"least_squares: max residual {np.abs(sol.fun).max():.1e} "
      f"after {sol.nfev} evaluations")

# %% Q7(c) how well do the models fit T?
fit_report("unconstrained 1/d^2", T_un, T)
fit_report("origin-constrained 1/d^2", T_oc, T)
fit_report("doubly constrained 1/d^2", T_dbl, T)

# Calibrate the power for the doubly constrained model. Two common criteria:
#  - maximise CPC
#  - match the observed mean trip length (Hyman's method) - the standard one
#    in practice, because it needs only O, D and the mean trip length
mean_obs = (T * dist).sum() / T.sum()
gammas = np.arange(0.5, 4.01, 0.1)
cp, mtl = [], []
for g in gammas:
    _, _, Tg, _ = furness(O, D, dist**-g, tol=1e-6)
    cp.append(cpc(Tg, T)); mtl.append((Tg * dist).sum() / Tg.sum())
cp, mtl = np.array(cp), np.array(mtl)
g_cpc = gammas[cp.argmax()]
g_mtl = gammas[np.argmin(np.abs(mtl - mean_obs))]
print(f"Doubly constrained: best gamma by CPC = {g_cpc:.1f} (CPC {cp.max():.3f}); "
      f"mean trip length {mean_obs:.1f} km matched at gamma = {g_mtl:.1f}")
_, _, T_best, _ = furness(O, D, dist**-g_cpc)
fit_report(f"doubly constrained d^-{g_cpc:.1f}", T_best, T)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(gammas, cp, "o-", ms=3); axs[0].set_xlabel("gamma"); axs[0].set_ylabel("CPC")
axs[0].set_title("Doubly constrained: CPC vs gamma")
axs[1].plot(gammas, mtl, "o-", ms=3); axs[1].axhline(mean_obs, color="r", ls="--",
                                                    label="observed")
axs[1].set_xlabel("gamma"); axs[1].set_ylabel("mean trip length (km)"); axs[1].legend()
m = T > 0
axs[2].loglog(T[m], T_best[m], ".", ms=1, alpha=0.3)
axs[2].plot([1, T.max()], [1, T.max()], "k--")
axs[2].set_xlabel("observed T_ij"); axs[2].set_ylabel("model T_ij")
axs[2].set_title(f"Doubly constrained, gamma = {g_cpc:.1f}")
fig.savefig(FIG / f"q7_doubly_constrained_{LEVEL}.png", dpi=150, bbox_inches="tight")

# Trip length distributions: model vs data
bins = np.arange(0, 60, 2)
fig, ax = plt.subplots(figsize=(6, 4))
for Tm, lab in [(T, "observed"), (T_dbl, "doubly constr. 1/d^2"),
                (T_best, f"doubly constr. d^-{g_cpc:.1f}")]:
    ax.hist(dist.ravel(), bins=bins, weights=Tm.ravel(), histtype="step", lw=1.5, label=lab)
ax.set_xlabel("distance (km)"); ax.set_ylabel("trips"); ax.legend()
fig.savefig(FIG / f"q7_trip_length_{LEVEL}.png", dpi=150, bbox_inches="tight")

plt.show()
