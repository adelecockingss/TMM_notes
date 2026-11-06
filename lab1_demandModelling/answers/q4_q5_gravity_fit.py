# TMM Lab 01 - Q4 & Q5: fitting the unconstrained gravity model
#     T_ij ~ c O_i D_j f(d_ij)
# Run cell-by-cell or: python q4_q5_gravity_fit.py  (run q3_od_matrix.py first)
# MATLAB version: q4_q5_gravity_fit.m

# %% Imports and data
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from gravity_utils import load_level, fit_c, cpc, fit_report

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

LEVEL = "bristol"   # 'bristol' (MSOAs within 50 km), 'utla' (174 zones) or 'all'
z = load_level(LEVEL)
T, O, D, dist = z["T"], z["O"], z["D"], z["dist"]
n = len(O)
print(f"{LEVEL}: {n} zones, {T.sum():,.0f} trips, {(T > 0).mean():.1%} of pairs non-zero")

# %% Q4: f(d) = 1/d^2. Unravel into an n^2 x 2 matrix [X_ij, T_ij]
OD = np.outer(O, D)                       # O_i D_j for every pair
X = OD / dist**2
pairs = np.column_stack([X.ravel(), T.ravel()])   # the "n^2 x 2 matrix"
print(pairs[:5])

# Three choices of c (see gravity_utils.fit_c for the maths)
cs = {m: fit_c(T, X, m) for m in ["ols", "log", "poisson"]}
for m, c in cs.items():
    fit_report(f"1/d^2, c by {m} (c={c:.2e})", c * X, T)

# %% Q4: scatter plot (log-log, non-zero pairs only) with the three lines
nz = T.ravel() > 0
fig, ax = plt.subplots(figsize=(6, 5))
hb = ax.hexbin(np.log10(pairs[nz, 0]), np.log10(pairs[nz, 1]), gridsize=60,
               bins="log", cmap="Greys")
fig.colorbar(hb, label="number of OD pairs")
xx = np.linspace(np.log10(pairs[nz, 0]).min(), np.log10(pairs[nz, 0]).max(), 2)
for (m, c), col in zip(cs.items(), ["tab:red", "tab:green", "tab:blue"]):
    ax.plot(xx, xx + np.log10(c), color=col, label=f"{m}: c = {c:.1e}")
# Binned mean of T in each X bin (including zeros!) - what the model should track
edges = np.linspace(xx[0], xx[1], 25)
k = np.digitize(np.log10(np.maximum(pairs[:, 0], 1e-300)), edges)
means = [pairs[k == b, 1].mean() if np.any(k == b) else np.nan for b in range(1, len(edges))]
ax.plot(0.5 * (edges[1:] + edges[:-1]), np.log10(means), "ko", ms=3, label="binned mean of T")
ax.set_xlabel(r"$\log_{10}\, O_i D_j / d_{ij}^2$"); ax.set_ylabel(r"$\log_{10}\, T_{ij}$")
ax.legend(fontsize=8); ax.set_title(f"Q4 gravity fit ({LEVEL})")
fig.savefig(FIG / f"q4_gravity_scatter_{LEVEL}.png", dpi=150, bbox_inches="tight")
# Why weighting matters: plain OLS on T (not log T) is dominated by the few
# largest flows (often intra-zonal), and the log fit ignores the zeros. The
# Poisson fit treats every pair as a count with variance ~ mean, which
# down-weights the big flows relative to OLS but still uses the zeros.

# %% Q5: other inverse powers  f(d) = d^-gamma  (and exponential exp(-beta d))
gammas = np.arange(0.5, 4.01, 0.1)
res = []
for g in gammas:
    Xg = OD / dist**g
    c = fit_c(T, Xg, "poisson")
    Tm = c * Xg
    # Poisson deviance: the quantity the Poisson MLE minimises (lower = better)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(T > 0, T * np.log(T / Tm), 0) - (T - Tm))
    res.append((g, c, cpc(Tm, T), dev))
res = np.array(res)
g_best = res[np.argmax(res[:, 2]), 0]
print(f"Best power by CPC: gamma = {g_best:.1f}; by deviance: "
      f"gamma = {res[np.argmin(res[:, 3]), 0]:.1f}")

betas = np.linspace(0.01, 1.0, 100)                        # per km
cpc_exp = [cpc(fit_c(T, OD * np.exp(-b * dist), "poisson") * OD * np.exp(-b * dist), T)
           for b in betas]
b_best = betas[int(np.argmax(cpc_exp))]
print(f"Best exponential: beta = {b_best:.3f} /km  (CPC {max(cpc_exp):.3f})")

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(res[:, 0], res[:, 2], "o-", ms=3)
axs[0].axvline(2, color="k", ls=":", label="classic gamma = 2")
axs[0].set_xlabel(r"$\gamma$ in $f(d) = d^{-\gamma}$"); axs[0].set_ylabel("CPC")
axs[0].legend(); axs[0].set_title("Power-law deterrence")
axs[1].plot(betas, cpc_exp, "o-", ms=3, color="tab:orange")
axs[1].set_xlabel(r"$\beta$ (1/km) in $f(d) = e^{-\beta d}$"); axs[1].set_ylabel("CPC")
axs[1].set_title("Exponential deterrence")
fig.savefig(FIG / f"q5_gamma_scan_{LEVEL}.png", dpi=150, bbox_inches="tight")

# %% Q5: estimate f(d) directly - "f(d) ~ T_ij / (O_i D_j)" (Anders's slide)
# Bin pairs by distance and compare sum(T) with sum(O_i D_j) in each bin.
dbins = np.logspace(np.log10(dist.min()), np.log10(dist.max()), 30)
kb = np.digitize(dist.ravel(), dbins)
f_emp = np.array([T.ravel()[kb == b].sum() / OD.ravel()[kb == b].sum()
                  if np.any(kb == b) else np.nan for b in range(1, len(dbins))])
dmid = np.sqrt(dbins[1:] * dbins[:-1])
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(dmid, f_emp, "o", label="empirical  sum T / sum OD")
ok = np.isfinite(f_emp) & (f_emp > 0)
s, i0 = np.polyfit(np.log(dmid[ok]), np.log(f_emp[ok]), 1)
ax.loglog(dmid, np.exp(i0) * dmid**s, "r-", label=f"power law slope {s:.2f}")
ax.set_xlabel("distance (km)"); ax.set_ylabel("f(d)"); ax.legend()
ax.set_title(f"Empirical deterrence function ({LEVEL})")
fig.savefig(FIG / f"q5_empirical_f_{LEVEL}.png", dpi=150, bbox_inches="tight")

# %% Q5: generalised gravity model  T = k O^alpha D^beta d^-gamma  (Anders)
# Fit as a Poisson GLM with log link:
#     log E[T_ij] = log k + alpha log O_i + beta log D_j - gamma log d_ij
# (a regression on the logs of non-zero T is the quick-and-dirty alternative)
ok = (OD.ravel() > 0)
Xglm = np.column_stack([np.log(np.repeat(O, n)), np.log(np.tile(D, n)),
                        np.log(dist.ravel())])[ok]
glm = sm.GLM(T.ravel()[ok], sm.add_constant(Xglm), family=sm.families.Poisson()).fit()
k_, alpha, beta, mgam = glm.params
print(f"Poisson GLM: alpha = {alpha:.2f}, beta = {beta:.2f}, gamma = {-mgam:.2f}")
Tm = np.zeros_like(T); Tm.ravel()[ok] = glm.predict(sm.add_constant(Xglm))
fit_report("generalised gravity (GLM)", Tm, T)

plt.show()
