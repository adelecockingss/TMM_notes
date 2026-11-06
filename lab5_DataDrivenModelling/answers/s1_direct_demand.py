# TMM Lab 05 - Section 1: direct demand modelling by inverse distance weighting
# Worked example + exercises 1.0-1.3.   MATLAB version: s1_direct_demand.m
# Run: python s1_direct_demand.py  (uses synthetic data unless the Blackboard
# files are present - see melbourne_data.py)

# %% Imports and data
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize_scalar

from melbourne_data import load

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
D = load()
central, xs, ys, sensx, sensy = D["central"], D["xs"], D["ys"], D["sensx"], D["sensy"]
TAG = " (SYNTHETIC data)" if D["synthetic"] else ""
n = len(xs) // 2
segs = np.stack([np.column_stack([xs[0::2], ys[0::2]]), np.column_stack([xs[1::2], ys[1::2]])], 1)

# %% Step 1: street segments -> network (node = segment, edge = segments touch)
# Vectorised version of the sheet's double loop: segments i, j are adjacent
# if any end point of i coincides with any end point of j. Weight = half of
# each segment's length added together (distance between their midpoints
# along the street).
length = np.hypot(*(segs[:, 1] - segs[:, 0]).T)
ends = segs.reshape(-1, 2)                                 # 2n end points
same = (np.abs(ends[:, None, 0] - ends[None, :, 0]) + np.abs(ends[:, None, 1] - ends[None, :, 1])) == 0
touch = same.reshape(n, 2, n, 2).any(axis=(1, 3))
np.fill_diagonal(touch, False)
A = np.where(touch, (length[:, None] + length[None, :]) / 2, 0.0)
print(f"{n} street segments, {int(touch.sum() / 2)} adjacencies{TAG}")

# %% Step 2: assign sensors to the nearest segment; merge sensors on one segment
mid = segs.mean(1)
idsens = np.array([np.argmin(np.hypot(*(mid - [sx, sy]).T)) for sx, sy in zip(sensx, sensy)])
if not D["synthetic"]:
    # the sheet's manual correction for sensor 9 (index 8 in Python)
    idsens[8] = np.argsort(np.hypot(*(mid - [sensx[8], sensy[8]]).T))[1]
# Sum the two sensors on a shared segment (opposite pavements); double a lone
# sensor (it only sees one side). NB: done by grouping, in the sorted order of
# the segment ids - the sheet's while-loop assumes paired sensors are next to
# each other in the file and fills `signal` in file order, but then uses the
# SORTED unique(idsens) as `sensnodes`, which only matches if idsens is sorted.
sensnodes = np.unique(idsens)
signal = np.array([central[idsens == s].sum(0) * (2 if np.sum(idsens == s) == 1 else 1)
                   for s in sensnodes])
print(f"{len(sensx)} sensors -> {len(sensnodes)} sensor sites")

fig, ax = plt.subplots(figsize=(7, 6))
ax.add_collection(LineCollection(segs, colors="0.6", linewidths=3))
ax.add_collection(LineCollection(segs[sensnodes], colors="k", linewidths=3))
ax.scatter(sensx, sensy, s=80, c="g", zorder=3)
ax.autoscale(); ax.set_aspect("equal"); ax.set_title("streets and sensors" + TAG)
fig.savefig(FIG / "s1_network.png", dpi=130, bbox_inches="tight")

# %% Step 3: shortest network distances from each sensor site to every segment
G = sp.csr_matrix(A)
alldists = dijkstra(G, indices=sensnodes)          # (n_sensors, n_segments)


# %% Step 4: IDW prediction (IDWprediction.m)
def idw_predict(nodes, counts, alldists, p=2.0, extra=None):
    """counts: (n_sensors,) at one time. For each node, weighted average of
    the OTHER sensors' counts with weights 1/d^p (a sensor never predicts
    itself: that is what the sheet's 'shortpaths == 0' removal does).
    extra: optional (n_sensors,) multiplicative sensor weights (exercise 1.3)."""
    out = np.empty(len(nodes))
    for k, node in enumerate(nodes):
        d = alldists[:, node]
        use = d > 0
        w = 1 / d[use] ** p
        if extra is not None:
            w = w * extra[use]
        out[k] = np.sum(w * counts[use]) / np.sum(w)
    return out


print("prediction for sensor site 1 at hour 100:",
      idw_predict([sensnodes[0]], signal[:, 99], alldists)[0].round(1),
      " ground truth:", signal[0, 99])
focal = 2
pred = np.array([idw_predict([sensnodes[focal]], signal[:, t], alldists)[0] for t in range(300)])
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(signal[focal, :300], lw=2, label="ground truth")
ax.plot(pred, "b", lw=2, label="IDW prediction")
ax.set_xlabel("time (hours)"); ax.set_ylabel("pedestrian count"); ax.legend()
ax.set_title(f"sensor site {focal + 1}" + TAG)
fig.savefig(FIG / "s1_focal_prediction.png", dpi=130, bbox_inches="tight")

# %% Exercise 1.0 - parameters
# ONE parameter: the distance exponent p (= 2 here). (The network
# construction - weights, sensor merging - involves further modelling
# choices, but those are not fitted.)

# %% Exercise 1.1 - predictions on the whole network over time (animation)
all_nodes = np.arange(n)
frames = range(24 * 2, 24 * 4)                      # two days, hour by hour
P = np.array([idw_predict(all_nodes, signal[:, t], alldists) for t in frames])
# at a sensor site, show the measured count instead of the leave-self-out estimate
fig, ax = plt.subplots(figsize=(7, 6))
lc = LineCollection(segs, cmap="inferno", linewidths=4, array=P[0], clim=(0, np.percentile(P, 99)))
ax.add_collection(lc); ax.scatter(sensx, sensy, s=20, c="c", zorder=3)
ax.autoscale(); ax.set_aspect("equal"); fig.colorbar(lc, label="predicted count")
ttl = ax.set_title("")
def draw(k):
    lc.set_array(P[k]); ttl.set_text(f"hour {frames[k]} (hour of day {frames[k] % 24}){TAG}")
    return lc, ttl
anim = animation.FuncAnimation(fig, draw, frames=len(P), interval=150)
anim.save(FIG / "s1_idw_animation.gif", writer=animation.PillowWriter(fps=6))
draw(int(np.argmax(P.sum(1)))); fig.savefig(FIG / "s1_idw_peak_hour.png", dpi=130, bbox_inches="tight")
print("Exercise 1.1: animation saved to figures/s1_idw_animation.gif")


# %% Exercise 1.2 - rigorous accuracy: leave-one-sensor-out cross-validation
def loo_errors(p=2.0, extra=None, dists=alldists, T=range(signal.shape[1])):
    """Predict each sensor from the others at every hour; return RMSE and
    MAE per sensor. (IDW already excludes the sensor itself, so 'leave one
    out' just means predicting each sensor site in turn.)"""
    T = list(T)
    rmse, mae = [], []
    for k in range(len(sensnodes)):
        pr = np.array([idw_predict([sensnodes[k]], signal[:, t], dists, p, extra)[0] for t in T])
        err = pr - signal[k, T]
        rmse.append(np.sqrt(np.mean(err**2))); mae.append(np.mean(np.abs(err)))
    return np.array(rmse), np.array(mae)


sub = range(0, signal.shape[1], 3)                  # every 3rd hour keeps it quick
rmse2, mae2 = loo_errors(2.0, T=sub)
print("Exercise 1.2: leave-one-out RMSE per sensor site (p = 2):", rmse2.round(0))
print(f"   mean RMSE {rmse2.mean():.1f}, mean MAE {mae2.mean():.1f}; "
      f"baseline 'mean of other sensors' (p = 0): {loo_errors(0.0, T=sub)[0].mean():.1f}")
# Also: relative errors (counts differ hugely between sites), skill over time
# of day, and a separate test period (fit on June, test on July).

# %% Exercise 1.3 - extended IDW
# (i) a free exponent p fitted by cross-validation
res = minimize_scalar(lambda p: loo_errors(p, T=sub)[0].mean(), bounds=(0, 6), method="bounded")
print(f"Exercise 1.3(i): best exponent p = {res.x:.2f} (mean LOO RMSE {res.fun:.1f})")

# (ii) perceived distance using (artificial) pavement widths w_e:
#      d_perceived = sum over segments of length_e * (w_ref / w_e)^gamma
#      - wide pavements feel 'shorter' (more pleasant), gamma is a parameter.
rng = np.random.default_rng(1)
width = rng.uniform(1, 5, n)                         # metres, made up (as suggested)


def perceived_dists(gamma, w_ref=3.0):
    L_eff = length * (w_ref / width) ** gamma
    Ae = np.where(touch, (L_eff[:, None] + L_eff[None, :]) / 2, 0.0)
    return dijkstra(sp.csr_matrix(Ae), indices=sensnodes)


for gamma in [0.0, 0.5, 1.0]:
    print(f"Exercise 1.3(ii): gamma = {gamma}: mean LOO RMSE "
          f"{loo_errors(2.0, dists=perceived_dists(gamma), T=sub)[0].mean():.1f}")
# With MADE-UP widths this cannot help (they carry no information) - the
# point is the method: add covariates to the distance, fit their parameters
# by cross-validation, and keep them only if out-of-sample error drops.

# (iii) sensor weights: w_k multiplies sensor k's IDW weight, e.g. to trust
# reliable sensors more. Free per-sensor weights = 11 extra parameters fitted
# on 11 sensors -> overfitting risk; they also make the model location-
# specific (less transferable). Here: weight by each sensor's correlation
# with the others (a crude 'representativeness').
corr = np.corrcoef(signal)
rep = (corr.sum(1) - 1) / (len(signal) - 1)
print(f"Exercise 1.3(iii): representativeness-weighted IDW mean LOO RMSE "
      f"{loo_errors(2.0, extra=rep, T=sub)[0].mean():.1f}")

plt.show()
