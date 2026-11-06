# TMM Lab 04 - Problem 1: corridor models
#   (a) uni-directional periodic corridor: fundamental diagram, calibration
#   (b) 50 m corridor with a group of 4 that stays together
#   (c) bi-directional corridor: mean speed vs fraction alpha walking right
# Headless (fast). Set SHOW = True to watch the simulations.
# Run: python p1_corridor.py   (~3 min)

# %% Imports
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from ped_common import World, social_force, normalise_vector, run

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
SHOW = False
RENDER = 20 if SHOW else 0


def corridor(L, w, groups, interactions, directions, init=None):
    """Periodic corridor [0, L] x [0, w]. Destinations sit OUTSIDE the corridor,
    so nobody ever arrives - they just keep walking round (a 'ring')."""
    return World({
        "space": {
            "corridor": {"type": "rectangle", "coordinates": [0, 0, L, w], "colour": "gray",
                         "add_boundaries": False},
            "right": {"type": "rectangle", "coordinates": [L + 1, 0, L + 2, w], "colour": "green",
                      "add_boundaries": False},
            "left": {"type": "rectangle", "coordinates": [-2, 0, -1, w], "colour": "green",
                     "add_boundaries": False},
        },
        "pedestrians": groups,
        "boundaries": [[0, 0, L, 0], [0, w, L, w]],
        "periodic_boundaries": {"axis": "x", "pos1": 0, "pos2": L},
        "functions": {"update_directions": directions, "process_interactions": interactions,
                      "pedestrian_initialisation": init or (lambda *a: None)},
    }, use_graphics=SHOW)


def along_x(pedestrians, boundaries, polygons):
    """Desired direction: straight along the corridor towards the destination side."""
    for ped in pedestrians.values():
        dx = polygons[ped.destination]["nodes"].mean(0)[0] - ped.pos[0]
        ped.desired_direction = np.array([np.sign(dx), 0.0])


def mean_speed(world, T=30, warmup=10, dt=0.05):
    """Average speed in each pedestrian's own desired direction after a warm-up."""
    run(world, warmup, dt, RENDER)
    v = []
    for _ in range(int(T / dt)):
        world.update(dt)
        v.append(np.mean([p.vel @ p.desired_direction for p in world.pedestrians.values()]))
    return float(np.mean(v))


# %% (a)(i) fundamental diagram: speed and flow against density
def weidmann(rho):
    """Empirical fundamental diagram (Weidmann 1993): v = 1.34(1 - exp(-1.913(1/rho - 1/5.4)))."""
    return 1.34 * (1 - np.exp(-1.913 * (1 / rho - 1 / 5.4)))


L, w = 10, 3
densities = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
results = {}
rng = np.random.default_rng(1)


def narrow_speeds(ped, *args):
    """Desired speeds ~ N(1.34, 0.1^2) instead of the provided N(1.34, 0.37^2)."""
    ped.desired_walking_speed = rng.normal(1.34, 0.1)


for label, init in [("provided: v0 ~ N(1.34, 0.37)", None),
                    ("adjusted: v0 ~ N(1.34, 0.10)", narrow_speeds)]:
    rows = []
    for rho in densities:
        N = int(round(rho * L * w))
        world = corridor(L, w, {"g": {"source": "corridor", "destination": "right",
                                      "colour": "red", "birth_rate": 999, "max_count": N}},
                         lambda peds, b, p: social_force(peds, b), along_x, init)
        v = mean_speed(world)
        rows.append((rho, N, v, rho * v))
        print(f"{label}: rho = {rho:4.2f} ({N:3d} peds): speed {v:.2f} m/s, "
              f"flow {rho * v:.2f} ped/(m s)")
    results[label] = np.array(rows)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
r = np.linspace(0.05, 5.3, 200)
axs[0].plot(r, weidmann(r), "k--", label="Weidmann (empirical)")
axs[1].plot(r, r * weidmann(r), "k--", label="Weidmann (empirical)")
for label, rows in results.items():
    axs[0].plot(rows[:, 0], rows[:, 2], "o-", label=label)
    axs[1].plot(rows[:, 0], rows[:, 3], "o-", label=label)
axs[0].set_xlabel("density (ped/m$^2$)"); axs[0].set_ylabel("speed (m/s)"); axs[0].legend()
axs[1].set_xlabel("density (ped/m$^2$)"); axs[1].set_ylabel("specific flow (ped/(m s))")
fig.savefig(FIG / "p1a_fundamental_diagram.png", dpi=130, bbox_inches="tight")
# Expected (lecture): speed falls with density; flow rises to a maximum of
# ~1.2-1.3 ped/(m s) at ~1.5-2 ped/m^2, then falls (congested branch).
# The provided model matches Weidmann well at high density (0.35 vs 0.33 m/s
# at 3 ped/m^2) but is too slow at low density (1.0 vs 1.3 m/s at 0.5).
# Two reasons: (1) desired speeds ~ N(1.34, 0.37^2) are very spread out, and
# in a narrow corridor fast walkers get stuck behind slow ones (platoons);
# (2) the social force has a long range (B = 2 m), so people ahead push back.
# Narrowing the speed spread fixes most of (1). Shortening B instead lets
# people barely slow down even at 3 ped/m^2 (tested: flow ~3.3, unrealistic),
# so the lecture's force parameters are kept.


# %% (b) group of 4 that stays together (attractive force between members)
def with_groups(pedestrians, boundaries, polygons, L, A_group=1.0, d_group=1.2):
    """Social force + attraction to group members further away than d_group.
    Separation uses the periodic (shortest way round) distance along x."""
    social_force(pedestrians, boundaries)
    members = [p for p in pedestrians.values() if p.group_name == "group"]
    for pi in members:
        for pj in members:
            if pi is pj:
                continue
            dvec = pj.pos - pi.pos
            dvec[0] -= L * round(dvec[0] / L)
            if np.linalg.norm(dvec) > d_group:
                pi.acc = pi.acc + A_group * normalise_vector(dvec)


def group_spread(world, L):
    m = [p.pos for p in world.pedestrians.values() if p.group_name == "group"]
    best = 0.0
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            dvec = m[j] - m[i]; dvec[0] -= L * round(dvec[0] / L)
            best = max(best, np.linalg.norm(dvec))
    return best


fig, ax = plt.subplots(figsize=(6, 4))
for A_group in [0.0, 1.0, 3.0]:
    L2, w2 = 50, 4
    world = corridor(L2, w2, {
        "individuals": {"source": "corridor", "destination": "right", "colour": "red",
                        "birth_rate": 999, "max_count": 16},
        "group": {"source": "corridor", "destination": "right", "colour": "blue",
                  "birth_rate": 999, "max_count": 4}},
        partial(with_groups, L=L2, A_group=A_group), along_x)
    t, s = [], []
    for k in range(int(60 / 0.05)):
        world.update(0.05)
        t.append(world.time); s.append(group_spread(world, L2))
    ax.plot(t, s, label=f"A_group = {A_group}")
    print(f"(b) A_group = {A_group}: group spread after 60 s = {s[-1]:.1f} m")
ax.set_xlabel("time (s)"); ax.set_ylabel("largest distance between members (m)"); ax.legend()
ax.set_title("(b) group cohesion, L = 50 m, w = 4 m, N = 20")
fig.savefig(FIG / "p1b_group.png", dpi=130, bbox_inches="tight")
# Without attraction the 4 drift apart (different desired speeds); with it,
# they converge to a cluster ~1-2 m across and walk at a common speed
# (roughly the group's average desired speed: fast members are held back).


# %% (c) bi-directional corridor: mean speed vs alpha
alphas = [0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
speeds = []
for alpha in alphas:
    N = 40; nr = int(round(alpha * N))
    world = corridor(10, 2, {
        "l2r": {"source": "corridor", "destination": "right", "colour": "red",
                "birth_rate": 999, "max_count": nr},
        "r2l": {"source": "corridor", "destination": "left", "colour": "blue",
                "birth_rate": 999, "max_count": N - nr}},
        lambda peds, b, p: social_force(peds, b), along_x)
    speeds.append(mean_speed(world))
    print(f"(c) alpha = {alpha:.2f}: mean speed {speeds[-1]:.2f} m/s")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alphas, speeds, "o-"); ax.set_xlabel("alpha (fraction walking left to right)")
ax.set_ylabel("mean speed in desired direction (m/s)"); ax.set_ylim(bottom=0)
ax.set_title("(c) bi-directional corridor, L = 10 m, w = 2 m, N = 40")
fig.savefig(FIG / "p1c_bidirectional.png", dpi=130, bbox_inches="tight")
# Symmetric in alpha <-> 1 - alpha, slowest near alpha = 0.5 (most head-on
# conflicts). Even small counter-flows (alpha = 0.1) cost speed. Lanes often
# form (self-organisation) and partly recover the speed; at 2 ped/m^2 in a
# 2 m corridor the flow can also jam completely (counter-flow gridlock).

plt.show()
