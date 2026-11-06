# TMM Lab 05 - Section 3: machine-learning a microscopic pedestrian model
# Real data: b160_combined.txt (bottleneck experiment, columns id, frame,
# x (cm), y (cm)). Worked example + exercises 3.0-3.4.
# MATLAB version: s3_ml_microscopic.m.   Run: python s3_ml_microscopic.py (~1 min)

# %% Imports and data
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

data = np.loadtxt(HERE.parent / "b160_combined (1).txt")
data = data[data[:, 3] <= 0]                       # remove the bottleneck itself (y > 0)
ids = np.unique(data[:, 0]).astype(int)
times = np.unique(data[:, 1])

# Discretise onto 10 cm x 10 cm cells (same formula as the sheet, 1-based)
dn = data.copy()
dn[:, 2] = np.floor((data[:, 2] + 250) / 10) + 1
dn[:, 3] = np.floor(np.abs(data[:, 3]) / 10) + 1
# Downsample 10 Hz -> 2 Hz (every 10th frame)
dn = dn[np.isin(dn[:, 1], np.arange(times.min(), times.max() + 1, 10))].astype(int)
print(f"{len(ids)} pedestrians, {len(dn)} discretised positions at 2 Hz")

# Index: frame -> array of (id, x, y);  id -> its trajectory (frames sorted)
by_time = {t: dn[dn[:, 1] == t] for t in np.unique(dn[:, 1])}
traj = {i: dn[dn[:, 0] == i] for i in ids}


# %% dataparcel: input 9x9 occupancy around the person, output 7x7 next cell
def dataparcel(pid, k):
    """Python port of dataparcel.m for pedestrian pid at row k of its
    trajectory (the MATLAB version picks k at random). Grids are flattened
    COLUMN-major (order='F') so class numbers match MATLAB's vec2ind - 1.
    Returns input (81,), output class (0..48 or None if the step is > 3 cells),
    current position, and the step (dx, dy)."""
    T = traj[pid]
    t, x, y = T[k, 1], T[k, 2], T[k, 3]
    nx, ny = T[k + 1, 2], T[k + 1, 3]
    grid = np.zeros((9, 9)); grid[4, 4] = 1
    for _, ox, oy in by_time[t][:, [0, 2, 3]]:
        dx, dy = ox - x, oy - y
        if abs(dx) <= 4 and abs(dy) <= 4:
            grid[4 + dx, 4 + dy] = 1
    dx, dy = nx - x, ny - y
    cls = (dx + 3) + 7 * (dy + 3) if abs(dx) <= 3 and abs(dy) <= 3 else None
    return grid.ravel(order="F"), cls, np.array([x, y]), (dx, dy)


def extra_features(pid, k):
    """Exercise 3.3: what the sheet's model is missing.
    - the previous step (people keep moving roughly as before: 'momentum')
    - the direction to the exit (the bottleneck entrance at x ~ 0 cm, y = 0)"""
    T = traj[pid]
    prev = T[k, 2:4] - T[k - 1, 2:4] if k > 0 else np.zeros(2)
    exit_cell = np.array([26, 1])                   # x = 0 cm -> cell 26, y = 0 -> cell 1
    to_exit = exit_cell - T[k, 2:4]
    return np.r_[prev, to_exit / max(np.linalg.norm(to_exit), 1), np.linalg.norm(to_exit)]


def samples(pids, random_draws=None, extra=False):
    """random_draws=N: the sheet's scheme (random person, random time, with
    replacement). None: every (person, time) of the given people once."""
    X, y, P, step = [], [], [], []
    if random_draws:
        pairs = [(p, rng.integers(0, len(traj[p]) - 1)) for p in rng.choice(pids, random_draws)]
    else:
        pairs = [(p, k) for p in pids for k in range(len(traj[p]) - 1)]
    for p, k in pairs:
        inp, cls, pos, st = dataparcel(p, k)
        if cls is None:
            continue
        X.append(np.r_[inp, extra_features(p, k)] if extra else inp)
        y.append(cls); P.append(pos); step.append(extra_features(p, k)[:2])
    return np.array(X), np.array(y), np.array(P), np.array(step)


def evaluate(name, model, Xte, yte, prev_step=None):
    pred = model.predict(Xte) if model is not None else \
        (np.clip(prev_step[:, 0], -3, 3) + 3 + 7 * (np.clip(prev_step[:, 1], -3, 3) + 3)).astype(int)
    to_xy = lambda c: np.column_stack([c % 7 - 3, c // 7 - 3])
    err = np.linalg.norm(to_xy(pred) - to_xy(yte), axis=1)
    print(f"{name:45s} accuracy {np.mean(pred == yte):.3f} | within 1 cell "
          f"{np.mean(err <= 1.5):.3f} | mean error {10 * err.mean():.1f} cm")
    return pred


def nn():
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                                       max_iter=500, random_state=0))


# %% Worked example: random sampling, 4000 train / 1000 test, NN classifier
Xtr, ytr, _, _ = samples(ids, 4000)
Xte, yte, _, ste = samples(ids, 1000)
model = nn().fit(Xtr, ytr)
pred = evaluate("sheet: random samples, NN on 9x9 occupancy", model, Xte, yte)

# %% Exercise 3.1: confusion matrix
labels = np.unique(np.r_[yte, pred])
C = confusion_matrix(yte, pred, labels=labels)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
axs[0].imshow(np.log1p(C), cmap="Blues")
tick = [f"({l % 7 - 3},{l // 7 - 3})" for l in labels]
axs[0].set_xticks(range(len(labels)), tick, rotation=90, fontsize=6)
axs[0].set_yticks(range(len(labels)), tick, fontsize=6)
axs[0].set_xlabel("predicted step (dx, dy)"); axs[0].set_ylabel("true step (dx, dy)")
axs[0].set_title("3.1 confusion matrix (log counts)")
# Diagonal = correct; off-diagonal = wrong. Only the steps that occur appear
# (not all 49). The matrix treats a 1-cell miss the same as a 3-cell miss, so
# also report the distance error (done in evaluate()).

# %% Exercise 3.0: a better split - by pedestrian, every time point used once
perm = rng.permutation(ids)
train_ids, test_ids = perm[:int(0.8 * len(ids))], perm[int(0.8 * len(ids)):]
Xtr2, ytr2, _, _ = samples(train_ids)
Xte2, yte2, Pte2, ste2 = samples(test_ids)
print(f"split by pedestrian: {len(ytr2)} train / {len(yte2)} test samples")
m2 = nn().fit(Xtr2, ytr2)
evaluate("3.0: split by pedestrian, NN on 9x9 occupancy", m2, Xte2, yte2)
# Random sampling with replacement puts the SAME person (even the same
# moment) in both training and test sets, so the test score is optimistic
# and differs every run. Splitting by person tests generalisation to new
# people and is reproducible (fixed seed, every sample used once).

# %% Exercise 3.3: baselines and an improved model
evaluate("baseline: repeat previous step (no ML)", None, Xte2, yte2, prev_step=ste2)
Xtr3, ytr3, _, _ = samples(train_ids, extra=True)
Xte3, yte3, _, _ = samples(test_ids, extra=True)
m3 = nn().fit(Xtr3, ytr3)
evaluate("3.3: NN + previous step + direction to exit", m3, Xte3, yte3)

# %% Exercise 3.4: closed-loop prediction for one pedestrian
# Start from the true position, then use the MODEL's predicted positions as
# the person's position for the next step (others still follow the data).
pid = test_ids[0]
T = traj[pid]
pos = T[1, 2:4].astype(int).copy(); prev = T[1, 2:4] - T[0, 2:4]
path = [pos.copy()]
for k in range(1, len(T) - 1):
    t = T[k, 1]
    grid = np.zeros((9, 9)); grid[4, 4] = 1
    for _, ox, oy in by_time[t][:, [0, 2, 3]]:
        if _ == pid:
            continue
        dx, dy = ox - pos[0], oy - pos[1]
        if abs(dx) <= 4 and abs(dy) <= 4:
            grid[4 + dx, 4 + dy] = 1
    to_exit = np.array([26, 1]) - pos
    f = np.r_[grid.ravel(order="F"), prev, to_exit / max(np.linalg.norm(to_exit), 1),
              np.linalg.norm(to_exit)]
    c = m3.predict(f[None, :])[0]
    step = np.array([c % 7 - 3, c // 7 - 3])
    pos = pos + step; prev = step; path.append(pos.copy())
path = np.array(path)
axs[1].plot(dn[:, 2], dn[:, 3], ".", ms=1, color="0.8")
axs[1].plot(T[:, 2], T[:, 3], "r-", lw=2, label="observed")
axs[1].plot(path[:, 0], path[:, 1], "go-", ms=3, label="closed-loop prediction")
axs[1].set_xlabel("x (10 cm)"); axs[1].set_ylabel("y (10 cm)"); axs[1].legend()
axs[1].set_title(f"3.4 pedestrian {pid}")
fig.savefig(FIG / "s3_ml_model.png", dpi=130, bbox_inches="tight")
print(f"3.4: closed-loop end point {path[-1]}, observed end {T[-1, 2:4]} (cells of 10 cm)")

plt.show()
