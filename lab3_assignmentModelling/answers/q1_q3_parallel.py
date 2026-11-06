# TMM Lab 03 - Q1-Q3: two parallel links, UE vs SO, Price of Anarchy
# Run cell-by-cell or: python q1_q3_parallel.py.   MATLAB: q1_q3_parallel.m
# The by-hand working for Q1 is in NOTES.md - this script checks it.

# %% Imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from assign_utils import parallel_affine, qp_small, system_cost

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

# %% Q1 (a)-(d): c1 = 1/2 + x1, c2 = 1 + x2/2, d = 1
a = np.array([0.5, 1.0]); b = np.array([1.0, 0.5]); d = 1.0
x_ue = parallel_affine(a, b, d, "UE")
x_so = parallel_affine(a, b, d, "SO")
f_ue, f_so = system_cost(x_ue, a, b), system_cost(x_so, a, b)
print(f"UE  x = {x_ue}  link costs {a + b * x_ue}  system cost {f_ue:.4f} (by hand 7/6)")
print(f"SO  x = {x_so}  link costs {a + b * x_so}  system cost {f_so:.4f} (by hand 9/8)")
print(f"PoA = {f_ue / f_so:.4f} (by hand 28/27 = {28 / 27:.4f})")

# Q1(d): the same UE from minimising the Beckmann functional as a QP
# Beckmann: sum (a_i x_i + b_i x_i^2 / 2) -> H = diag(b), f = a (quadprog form)
x_b, _ = qp_small(np.diag(b), a, [[1, 1]], [d])
x_s, _ = qp_small(2 * np.diag(b), a, [[1, 1]], [d])      # SO: H = 2 diag(b)
print("Beckmann QP:", x_b, "  SO QP:", x_s)


# %% Q1(e) general demand: PoA against d
def poa_curve(a, b, ds):
    out = []
    for d in ds:
        fu = system_cost(parallel_affine(a, b, d, "UE"), a, b)
        fs = system_cost(parallel_affine(a, b, d, "SO"), a, b)
        out.append(fu / fs)
    return np.array(out)


ds = np.linspace(1e-3, 4, 800)
poa = poa_curve(a, b, ds)
print(f"Q1(e) max PoA = {poa.max():.4f} at d = {ds[poa.argmax()]:.3f}")
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(ds, [parallel_affine(a, b, d, "UE")[0] for d in ds], label="UE x1")
axs[0].plot(ds, [parallel_affine(a, b, d, "UE")[1] for d in ds], label="UE x2")
axs[0].plot(ds, [parallel_affine(a, b, d, "SO")[0] for d in ds], "--", label="SO x1")
axs[0].plot(ds, [parallel_affine(a, b, d, "SO")[1] for d in ds], "--", label="SO x2")
axs[0].set_xlabel("demand d"); axs[0].set_ylabel("flow"); axs[0].legend()
axs[1].plot(ds, poa); axs[1].set_xlabel("demand d"); axs[1].set_ylabel("PoA")
axs[1].set_title("Q1(e): c1 = 1/2 + x, c2 = 1 + x/2")
fig.savefig(FIG / "q1e_poa.png", dpi=130, bbox_inches="tight")

# %% Q2(b) fixed a, b, d - as in AssignmentQ2.mlx
a2, b2, d2 = 0.5, 0.5, 1000
for mode, H in [("UE", np.diag([1, b2])), ("SO", 2 * np.diag([1, b2]))]:
    x, val = qp_small(H, [a2, 1], [[1, 1]], [d2])
    print(f"Q2(b) {mode}: x = {x.round(3)}, link costs {a2 + x[0]:.2f}, {1 + b2 * x[1]:.2f}; "
          f"objective {val:.1f}, system cost {system_cost(x, np.array([a2, 1]), np.array([1, b2])):.1f}")


# %% Q2(c)-(d) scan over demand for fixed a, b
# Interesting behaviour happens at demands below ~ the point where both links
# are used in BOTH assignments; beyond that PoA -> 1 (both tend to share flow
# in the ratio of 'capacities'). UE uses link 2 once a + d >= 1, i.e. d >= 1 - a;
# SO uses it once a + 2d >= 1, i.e. d >= (1 - a)/2. So scan d up to a few x (1 - a)
# and beyond 1/b, the scale on which the flows rebalance.
def max_poa(a, b):
    """Q2(e): maximise PoA over d. PoA(d) is smooth and single-peaked in the
    region where SO uses both links but UE uses only link 1, so a bounded
    1-D search on a sensible bracket finds it."""
    av, bv = np.array([a, 1.0]), np.array([1.0, b])
    neg = lambda d: -poa_curve(av, bv, [d])[0]
    dmax = 10 * (1 + 1 / max(b, 1e-3))
    grid = np.linspace(1e-4, dmax, 400)
    k = int(np.argmin([neg(d) for d in grid]))
    r = minimize_scalar(neg, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
                        method="bounded")
    return -r.fun, r.x


fig, ax = plt.subplots(figsize=(6, 4))
for (aa, bb) in [(0, 0), (0.5, 0.5), (0.2, 1), (1, 1)]:
    ds = np.linspace(1e-3, 4, 500)
    ax.plot(ds, poa_curve(np.array([aa, 1.0]), np.array([1.0, bb]), ds), label=f"a={aa}, b={bb}")
    p, dstar = max_poa(aa, bb)
    print(f"Q2(e) a={aa}, b={bb}: max PoA {p:.4f} at d = {dstar:.3f}")
ax.set_xlabel("demand d"); ax.set_ylabel("PoA"); ax.legend()
ax.set_title("Q2(d) PoA vs demand, c1 = a + x1, c2 = 1 + b x2")
fig.savefig(FIG / "q2d_poa_vs_d.png", dpi=130, bbox_inches="tight")

# %% Q3 maximum PoA over 0 <= a, b <= 1
A, B = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
P = np.vectorize(lambda a, b: max_poa(a, b)[0])(A, B)
i = np.unravel_index(P.argmax(), P.shape)
print(f"Q3 max PoA over the grid = {P.max():.4f} at a = {A[i]:.2f}, b = {B[i]:.2f} "
      f"(theory: 4/3 = {4 / 3:.4f}, Pigou a = b = 0)")
fig, ax = plt.subplots(figsize=(5.5, 4.5))
cs = ax.contourf(A, B, P, levels=20, cmap="viridis"); fig.colorbar(cs, label="max PoA")
ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_title("Q3 max over d of PoA")
fig.savefig(FIG / "q3_max_poa.png", dpi=130, bbox_inches="tight")

plt.show()
