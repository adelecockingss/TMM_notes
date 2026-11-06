# TMM Lab 03 - Q4 & Q5: n parallel links, affine and nonlinear costs
# Run cell-by-cell or: python q4_q5_parallel_n.py.   MATLAB: q4_q5_parallel_n.m

# %% Imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from assign_utils import parallel_affine, parallel_general, qp_small, system_cost

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# %% Q4: n = 5 links with random affine costs
n = 5
a = rng.uniform(0, 2, n); b = rng.uniform(0.1, 2, n)
d = 2.0
x_ue = parallel_affine(a, b, d, "UE"); x_so = parallel_affine(a, b, d, "SO")
# check against the general QP solver (Beckmann: H = diag(b); SO: 2 diag(b))
x_ue_qp, _ = qp_small(np.diag(b), a, np.ones((1, n)), [d])
x_so_qp, _ = qp_small(2 * np.diag(b), a, np.ones((1, n)), [d])
print("a =", a.round(2), " b =", b.round(2))
print("UE:", x_ue.round(4), " costs", (a + b * x_ue).round(4), " QP agrees:", np.allclose(x_ue, x_ue_qp))
print("SO:", x_so.round(4), " QP agrees:", np.allclose(x_so, x_so_qp))
# UE: all USED links have the same cost; unused ones (if any) cost more at x = 0.

# PoA vs demand, and the worst case over many random networks
ds = np.linspace(0.01, 6, 300)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
worst = 1.0
for trial in range(200):
    a = rng.uniform(0, 2, n); b = rng.uniform(0, 2, n)        # b = 0 allowed (constant)
    poa = [system_cost(parallel_affine(a, b, d, "UE"), a, b) /
           system_cost(parallel_affine(a, b, d, "SO"), a, b) for d in ds]
    worst = max(worst, max(poa))
    if trial < 8:
        axs[0].plot(ds, poa, lw=1)
axs[0].set_xlabel("demand d"); axs[0].set_ylabel("PoA"); axs[0].set_title("Q4: random 5-link networks")
print(f"Q4 worst PoA over 200 random 5-link networks: {worst:.4f} (bound 4/3)")


# %% Q5: nonlinear costs - Pigou with c1 = x^p, c2 = 1 at d = 1
# Theory (Roughgarden): PoA = 1 / (1 - p (p+1)^(-(p+1)/p)), grows with p.
ps = np.arange(1, 9)
poa_num, poa_th = [], []
for p in ps:
    cost = [lambda x, p=p: x**p, lambda x: 1.0]
    dcost = [lambda x, p=p: p * x ** (p - 1), lambda x: 0.0]
    xu = parallel_general(cost, dcost, 1.0, "UE"); xs = parallel_general(cost, dcost, 1.0, "SO")
    f = lambda x: x[0] * x[0] ** p + x[1] * 1.0
    poa_num.append(f(xu) / f(xs))
    poa_th.append(1 / (1 - p * (p + 1) ** (-(p + 1) / p)))
for p, pn, pt in zip(ps, poa_num, poa_th):
    print(f"Q5 Pigou c1 = x^{p}: PoA numeric {pn:.4f}, theory {pt:.4f}")
axs[1].plot(ps, poa_num, "o-", label="numerical"); axs[1].plot(ps, poa_th, "k--", label="theory")
axs[1].set_xlabel("p in c1 = x^p"); axs[1].set_ylabel("PoA at d = 1"); axs[1].legend()
axs[1].set_title("Q5: nonlinear Pigou")
fig.savefig(FIG / "q4_q5_poa.png", dpi=130, bbox_inches="tight")

# BPR link performance function (used in practice): t = t0 (1 + 0.15 (x/cap)^4)
t0 = np.array([10.0, 15.0]); cap = np.array([1000.0, 2000.0])
cost = [lambda x, i=i: t0[i] * (1 + 0.15 * (x / cap[i]) ** 4) for i in range(2)]
dcost = [lambda x, i=i: t0[i] * 0.6 * x**3 / cap[i] ** 4 for i in range(2)]
for d in [500, 2000, 4000]:
    xu = parallel_general(cost, dcost, d, "UE"); xs = parallel_general(cost, dcost, d, "SO")
    f = lambda x: sum(xi * c(xi) for xi, c in zip(x, cost))
    print(f"BPR d = {d}: UE {xu.round(0)}, SO {xs.round(0)}, PoA {f(xu) / f(xs):.4f}")

plt.show()
