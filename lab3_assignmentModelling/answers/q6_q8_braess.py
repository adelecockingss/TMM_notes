# TMM Lab 03 - Q6-Q8: the Braess network
# Run cell-by-cell or: python q6_q8_braess.py.   MATLAB: q6_q8_braess.m
# By-hand derivations are in NOTES.md; this script checks them numerically.
#
# Links: 1 O->N, 2 N->D, 3 O->S, 4 S->D, 5 N->S (midtown)
# Routes: North {1,2}, South {3,4}, Midtown {1,5,4}

# %% Imports and the route formulation
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from assign_utils import qp_small

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

A = np.array([[1, 0, 1],      # link-route incidence: A[i, r] = 1 if route r uses link i
              [1, 0, 0],
              [0, 1, 0],
              [0, 1, 1],
              [0, 0, 1]], float)


def braess(a, b, d, mode="UE", midtown=True):
    """Solve in route variables y: minimise g(Ay) subject to sum y = d, y >= 0.
    UE (Beckmann): H = A' diag(b) A,  f = A' a.   SO: H = 2 A' diag(b) A.
    Returns route flows y, link flows x, route costs and system cost."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    R = [0, 1, 2] if midtown else [0, 1]
    Ar = A[:, R]
    H = Ar.T @ np.diag(b) @ Ar * (2 if mode == "SO" else 1)
    y_r, _ = qp_small(H, Ar.T @ a, np.ones((1, len(R))), [d])
    y = np.zeros(3); y[R] = y_r
    x = A @ y
    c_link = a + b * x
    c_route = A.T @ c_link
    return y, x, c_route, float(x @ c_link)


def report(name, a, b, d):
    print(f"--- {name}, d = {d}")
    for mode, mid in [("UE", True), ("SO", True), ("UE", False)]:
        y, x, cr, f = braess(a, b, d, mode, mid)
        print(f"  {mode} {'with' if mid else 'without'} midtown: y(N,S,M) = {y.round(3)}, "
              f"route costs {cr.round(3)}, system cost {f:.3f}")


# %% Q6(a) lecture example: c1 = x1, c2 = 1, c3 = 1, c4 = x4, c5 = 0
a6, b6 = [0, 1, 1, 0, 0], [1, 0, 0, 1, 0]
report("Q6(a) lecture network", a6, b6, 1.0)
# UE uses only midtown (cost 2 each, system cost 2); without it, cost 3/2.


# %% Q6(b) does the paradox depend on demand?
def per_user_cost(a, b, d, midtown):
    return braess(a, b, d, "UE", midtown)[3] / d


ds = np.linspace(0.05, 3, 300)
with_m = np.array([per_user_cost(a6, b6, d, True) for d in ds])
without = np.array([per_user_cost(a6, b6, d, False) for d in ds])
par = ds[with_m > without + 1e-9]
print(f"Q6(b) paradox for {par.min():.2f} < d < {par.max():.2f}  (by hand: 2/3 < d < 2)")
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
axs[0].plot(ds, with_m, label="UE with midtown"); axs[0].plot(ds, without, label="UE without")
axs[0].set_xlabel("demand d"); axs[0].set_ylabel("cost per user"); axs[0].legend()
axs[0].set_title("Q6(b) lecture network")

# %% Q6(c) swap the costs: c1, c4 constant (= 1), c2, c3 = flow
a6c, b6c = [1, 0, 0, 1, 0], [0, 1, 1, 0, 0]
wc = [per_user_cost(a6c, b6c, d, True) for d in ds]
woc = [per_user_cost(a6c, b6c, d, False) for d in ds]
print(f"Q6(c) max (with - without) = {max(np.subtract(wc, woc)):.2e}  -> no paradox")
# Midtown costs 1 + 0 + 1 = 2 regardless of flow: it is only used once N and S
# already cost 2, so adding it can only help (it adds capacity).

# %% Q6(d) midtown link with constant cost a5
a5s = np.linspace(0, 1.2, 61); ds2 = np.linspace(0.05, 3, 60)
ratio = np.array([[per_user_cost([0, 1, 1, 0, a5], b6, d, True) /
                   per_user_cost([0, 1, 1, 0, a5], b6, d, False) for d in ds2] for a5 in a5s])
im = axs[1].contourf(ds2, a5s, ratio, levels=[0.99, 1.0001, 1.1, 1.2, 1.3, 1.4], cmap="Reds")
fig.colorbar(im, ax=axs[1], label="UE cost with / without midtown")
axs[1].set_xlabel("demand d"); axs[1].set_ylabel("midtown cost a5")
axs[1].set_title("Q6(d) paradox region (ratio > 1)")
print("Q6(d) paradox occurs only for a5 <", a5s[(ratio > 1 + 1e-9).any(1)].max().round(3))

# %% Q7 Sheffi's example
report("Q7 Sheffi p76", [0, 50, 50, 0, 10], [10, 1, 1, 10, 1], 6.0)


# %% Q8 c1 = c4 = a + x, c2 = c3 = 1 + b x, c5 = 0: when is there a paradox?
def paradox_analytic(a, b, d):
    """By-hand result (see NOTES.md). Returns UE cost per user with / without."""
    C0 = a + 1 + d * (1 + b) / 2                 # without midtown: y_N = y_S = d/2
    if a + d <= 1:                               # everyone takes midtown
        return 2 * (a + d), C0
    y = (a + d - 1) / (1 + b)                    # flow on N (= flow on S)
    if y < d / 2:                                # all three routes used
        return 2 * (a + d - y), C0
    return C0, C0                                # midtown unused


for d in [0.5, 1.0, 2.0]:
    errs = []
    for a in np.linspace(0, 1, 6):
        for b in np.linspace(0, 1, 6):
            num = per_user_cost([a, 1, 1, a, 0], [1, b, b, 1, 0], d, True)
            errs.append(abs(num - paradox_analytic(a, b, d)[0]))
    print(f"Q8 d = {d}: max |numeric - analytic| = {max(errs):.1e}")

ag, bg = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
for d, ls in [(0.5, ":"), (1.0, "-"), (2.0, "--")]:
    gain = np.vectorize(lambda a, b: np.subtract(*paradox_analytic(a, b, d)))(ag, bg)
    axs[2].contour(ag, bg, gain, levels=[1e-9], colors="k", linestyles=ls)
    axs[2].plot([], [], "k" + ls, label=f"d = {d}")
axs[2].set_xlabel("a"); axs[2].set_ylabel("b"); axs[2].legend()
axs[2].set_title("Q8 paradox boundaries (paradox below/left)")
fig.savefig(FIG / "q6_q8_braess.png", dpi=130, bbox_inches="tight")

plt.show()
