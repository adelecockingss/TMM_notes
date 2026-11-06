# TMM Lab 04 - Problem 2: evacuation of a 10 m x 10 m room
#   (a) one door: evacuation time vs width; time headways and their CV
#   (b) two doors, each pedestrian uses the closest
#   (c) rolling ship: periodic sideways tilt (10 s cycle)
#   (d) maze with a potential-field ('flood fill') navigation
# Headless. Set SHOW = True to watch. Run: python p2_evacuation.py (~6 min)

# %% Imports
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

from ped_common import World, social_force, run, wall_with_door, PotentialField

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
SHOW = False
RENDER = 10 if SHOW else 0
DT = 0.05

# Evacuation calibration (see NOTES.md). The provided code gives bodies a
# radius of 0.4-0.5 m (0.8-1 m wide!) and a 2 m-range social force; with those,
# the flow through a door is only ~0.3-0.6 ped/(m s). Real shoulder widths are
# ~0.5 m and measured door flows are ~1.3-1.9 ped/(m s) (Seyfried et al.).
# Realistic radii + a short-range social force give ~1.6 ped/(m s).
EVAC = {"B_social": 0.3, "A_social": 2.0}


def evac_forces(peds, b, p=None, external=None):
    social_force(peds, b, EVAC, external)


def body(ped):
    ped.radius = 0.23 + 0.05 * random.random()


def room_world(source, walls, dest_polygon, n_peds, directions, interactions=None,
               init=None, seed=0):
    world = World({
        "space": {
            "room": {"type": "rectangle", "coordinates": [0, 0, 10, 10], "colour": "lightgray",
                     "add_boundaries": False},
            "start": {"type": "rectangle", "coordinates": source, "colour": "green",
                      "add_boundaries": False},
            "exit": {"type": "polygon", "coordinates": dest_polygon, "colour": "red",
                     "add_boundaries": False},
        },
        "pedestrians": {"evacuees": {"source": "start", "destination": "exit", "colour": "black",
                                     "birth_rate": 999, "max_count": n_peds}},
        "boundaries": walls,
        "functions": {"update_directions": directions,
                      "process_interactions": interactions or evac_forces,
                      "pedestrian_initialisation": init or (lambda ped, *a: body(ped))},
    }, use_graphics=SHOW)
    random.seed(seed)        # World fixes seed 42; reseed so replicates differ
    return world


def towards_door(pos, door):
    """Aim at the nearest point of the door opening, pulled 0.35 m in from the
    door frame so people don't walk into the wall ends. door = (p0, p1)."""
    p0, p1 = np.array(door[0], float), np.array(door[1], float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    L = np.linalg.norm(p1 - p0)
    m = min(0.35, L / 2 - 0.05)
    t = np.clip((pos - p0) @ u, m, L - m)
    target = p0 + t * u
    nrm = np.array([u[1], -u[0]])               # door normal, pointing out of the room
    if (target - np.array([5, 5])) @ nrm < 0:
        nrm = -nrm
    target = target + 0.5 * nrm                 # a point just outside the door
    d = target - pos
    return d / max(np.linalg.norm(d), 1e-9)


def right_door(w):
    return ((10, 5 - w / 2), (10, 5 + w / 2))


def bottom_door(w):
    return ((5 - w / 2, 0), (5 + w / 2, 0))


def walls_for(doors_w_right=None, doors_w_bottom=None):
    walls = [[0, 0, 0, 10], [0, 10, 10, 10]]
    walls += wall_with_door(10, 0, 10, 10, 5, doors_w_right) if doors_w_right else [[10, 0, 10, 10]]
    walls += wall_with_door(0, 0, 10, 0, 5, doors_w_bottom) if doors_w_bottom else [[0, 0, 10, 0]]
    return walls


RIGHT_EXIT = [[10.3, -1], [12, -1], [12, 11], [10.3, 11]]
L_EXIT = [[0, -2], [12, -2], [12, 10], [10.3, 10], [10.3, -0.3], [0, -0.3]]


def headways(exits):
    h = np.diff(exits)
    return h.mean(), h.std(), h.std() / h.mean()


# %% (a)(i) 50 pedestrians, 2 m door
def one_door(w, seed=0, n=50, source=(0, 0, 2, 10)):
    door = right_door(w)
    def dirs(peds, b, p):
        for ped in peds.values():
            ped.desired_direction = (towards_door(ped.pos, door) if ped.pos[0] < 10
                                     else np.array([1.0, 0.0]))
    world = room_world(list(source), walls_for(doors_w_right=w), RIGHT_EXIT, n, dirs, seed=seed)
    return run(world, 200, DT, RENDER, stop_when_empty_after=n)


ex = one_door(2.0)
print(f"(a)(i) 50 people, 2 m door: last person out at {ex[-1]:.1f} s")

# %% (a)(ii) evacuation time vs door width (3 replicates each)
widths = [0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
T = {w: [one_door(w, seed=s)[-1] for s in range(3)] for w in widths}
for w in widths:
    print(f"(a)(ii) width {w:.2f} m: evacuation time {np.mean(T[w]):.1f} +- {np.std(T[w]):.1f} s,"
          f" door flow {50 / np.mean(T[w]) / w:.2f} ped/(m s)")
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
ws = np.array(widths); Tm = np.array([np.mean(T[w]) for w in widths])
axs[0].errorbar(ws, Tm, yerr=[np.std(T[w]) for w in widths], fmt="o-")
c = np.polyfit(1 / ws, Tm, 1)
wf = np.linspace(0.7, 3.1, 100)
axs[0].plot(wf, c[0] / wf + c[1], "k--", label=f"T = {c[1]:.1f} + {c[0]:.1f}/w")
axs[0].set_xlabel("door width w (m)"); axs[0].set_ylabel("evacuation time (s)"); axs[0].legend()
axs[0].set_title("(a)(ii) 50 people")
# Bottleneck theory: flow through the door Q = J_s * w (specific flow J_s
# ~ 1.3-2 ped/(m s)), so T ~ T_walk + N / (J_s w)  ->  linear in 1/w.

# %% (a)(iii) time headways for w = 2, 1.5, 1.0
print()
for w in [2.0, 1.5, 1.0]:
    stats = [headways(one_door(w, seed=s)) for s in range(3)]
    mu, sd, cv = np.mean(stats, 0)
    print(f"(a)(iii) w = {w}: headway mean {mu:.2f} s, std {sd:.2f} s, CV = {cv:.2f}")
    h = np.diff(one_door(w, seed=0))
    axs[1].hist(h, bins=np.linspace(0, 2.5, 26), histtype="step", lw=1.5, label=f"w = {w} m")
axs[1].set_xlabel("time headway (s)"); axs[1].set_ylabel("count"); axs[1].legend()
axs[1].set_title("(a)(iii) headway distributions")
fig.savefig(FIG / "p2a_door_width.png", dpi=130, bbox_inches="tight")
# Result: CV is LOWER for the narrow door (0.87 at 1 m vs ~1.0 at 2 m).
# Narrow door: a queue forms and people pass in single file at a steady
# rate - the door works like a server with an almost fixed service time, so
# headways are regular. Wide door: 2-3 people pass side by side, so some
# consecutive exits are almost simultaneous (headway ~0) and others are far
# apart, and the door is less saturated (headways reflect random arrivals,
# CV ~ 1 like a Poisson process). Under high pressure ('panic'), real narrow
# exits can clog in arches and then CV rises - our calm calibration doesn't.


# %% (b) two doors, each person uses the closest one
def two_doors(w, seed=0, n=50, both=True):
    rd, bd = right_door(w), bottom_door(w)
    centres = {"right": np.mean(rd, 0), "bottom": np.mean(bd, 0)}
    def init(ped, polygons, stats):
        body(ped)
        if both:
            ped.metadata["door"] = min(centres, key=lambda k: np.linalg.norm(ped.pos - centres[k]))
        else:
            ped.metadata["door"] = "right"
    def dirs(peds, b, p):
        for ped in peds.values():
            door = rd if ped.metadata["door"] == "right" else bd
            inside = 0 < ped.pos[0] < 10 and 0 < ped.pos[1] < 10
            if inside:
                ped.desired_direction = towards_door(ped.pos, door)
            else:
                ped.desired_direction = np.array([1.0, 0]) if ped.metadata["door"] == "right" \
                    else np.array([0, -1.0])
    walls = walls_for(doors_w_right=w, doors_w_bottom=w if both else None)
    world = room_world([0, 7, 3, 10], walls, L_EXIT, n, dirs, init=init, seed=seed)
    return run(world, 200, DT, RENDER, stop_when_empty_after=n)


print()
fig, ax = plt.subplots(figsize=(6, 4))
res_b = {}
for w in [1.0, 1.5, 2.0]:
    one = np.mean([two_doors(w, s, both=False)[-1] for s in range(3)])
    two = np.mean([two_doors(w, s, both=True)[-1] for s in range(3)])
    res_b[w] = (one, two)
    print(f"(b) w = {w}: one door {one:.1f} s, two doors (closest) {two:.1f} s")
ax.plot(list(res_b), [v[0] for v in res_b.values()], "o-", label="right door only")
ax.plot(list(res_b), [v[1] for v in res_b.values()], "s-", label="two doors, closest")
ax.set_xlabel("door width (m)"); ax.set_ylabel("evacuation time (s)"); ax.legend()
ax.set_title("(b) source in the top-left corner")
fig.savefig(FIG / "p2b_two_doors.png", dpi=130, bbox_inches="tight")
# With the source in the top-left corner, distances to the two doors are
# similar, so 'closest door' splits the crowd roughly in half and halves the
# queueing time - the gain is biggest for narrow doors (queue dominated).
# A 'closest door' rule can do badly when the split is uneven: smarter
# choice = shortest expected time (distance/speed + queue/door capacity).


# %% (c) rolling ship: periodic tilt in y, period 10 s
def ship(amp, w=2.0, seed=0, n=50, period=10.0):
    door = right_door(w)
    clock = {"t": 0.0}
    def dirs(peds, b, p):
        for ped in peds.values():
            ped.desired_direction = (towards_door(ped.pos, door) if ped.pos[0] < 10
                                     else np.array([1.0, 0.0]))
    def inter(peds, b, p):
        # tilt angle theta(t) = theta_max sin(2 pi t / T); a person feels an
        # extra acceleration ~ g sin(theta) down the slope (along -y or +y).
        # amp = effective peak acceleration (m/s^2) after people compensate.
        ay = amp * np.sin(2 * np.pi * clock["t"] / period)
        evac_forces(peds, b, external=np.array([0.0, -ay]))
    world = room_world([0, 0, 2, 10], walls_for(doors_w_right=w), RIGHT_EXIT, n, dirs, inter, seed=seed)
    def tick(wd):
        clock["t"] = wd.time
    return run(world, 300, DT, RENDER, on_step=tick, stop_when_empty_after=n)


print()
amps = [0.0, 0.5, 1.0, 1.5, 2.0]
Ts = [np.mean([ship(A, seed=s)[-1] for s in range(3)]) for A in amps]
for A, t in zip(amps, Ts):
    print(f"(c) tilt acceleration amplitude {A:.1f} m/s^2 (~{np.degrees(np.arcsin(A / 9.81)):.0f} deg):"
          f" evacuation {t:.1f} s")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(amps, Ts, "o-"); ax.set_xlabel("peak sideways acceleration g sin(theta) (m/s$^2$)")
ax.set_ylabel("evacuation time (s)"); ax.set_title("(c) rolling ship, 2 m door, period 10 s")
fig.savefig(FIG / "p2c_ship.png", dpi=130, bbox_inches="tight")
# Result: small effect at moderate tilt (within the seed-to-seed noise of
# ~2 s up to ~6 degrees), about +8% at ~12 degrees. The evacuation only lasts
# ~2 roll cycles; the crowd is pushed alternately towards the top and bottom
# walls, so people approach the door at an angle and the queue sways, which
# lowers the door flow. Steeper tilts, or people who cannot compensate
# (elderly, luggage, injured), would be much worse.


# %% (d) maze with a potential field
maze_walls = [[0, 0, 10, 0], [0, 10, 10, 10], [0, 0, 0, 10],
              *wall_with_door(10, 0, 10, 10, 5, 1.0),       # exit y in [4.5, 5.5]
              [2, 0, 2, 8], [4, 2, 4, 10], [6, 0, 6, 8], [8, 2, 8, 10]]
field = PotentialField((-1, 13), (-1, 11), maze_walls, dest_box=(10.6, 4.5, 12, 5.5),
                       h=0.1, clearance=0.35)


def maze_dirs(peds, b, p):
    for ped in peds.values():
        e = field.direction(ped.pos)
        ped.desired_direction = e if e is not None else np.array([1.0, 0.0])


world = World({
    "space": {
        "maze": {"type": "rectangle", "coordinates": [0, 0, 10, 10], "colour": "lightgray",
                 "add_boundaries": False},
        "start": {"type": "rectangle", "coordinates": [0, 0, 2, 2], "colour": "green",
                  "add_boundaries": False},
        "exit": {"type": "rectangle", "coordinates": [10.6, 4, 12, 6], "colour": "red",
                 "add_boundaries": False}},
    "pedestrians": {"g": {"source": "start", "destination": "exit", "colour": "black",
                          "birth_rate": 2, "max_count": 30}},
    "boundaries": maze_walls,
    "functions": {"update_directions": maze_dirs,
                  "process_interactions": evac_forces,
                  "pedestrian_initialisation": lambda ped, *a: body(ped)}}, use_graphics=SHOW)
tracks = []
ex_d = run(world, 300, DT, RENDER, stop_when_empty_after=30,
           on_step=lambda wd: tracks.extend(p.pos.copy() for p in wd.pedestrians.values()))
print(f"\n(d) maze: {len(ex_d)} of 30 out, last at {ex_d[-1]:.1f} s "
      f"(path length from start ~{field.value(np.array([1.0, 1.0])):.1f} m)")
fig, ax = plt.subplots(figsize=(7, 6))
Dm = np.where(np.isfinite(field.D), field.D, np.nan)
im = ax.imshow(Dm.T, origin="lower", extent=(-1, 13, -1, 11), cmap="viridis")
fig.colorbar(im, label="distance to exit (m)")
tr = np.array(tracks)
ax.plot(tr[::7, 0], tr[::7, 1], "w.", ms=1)
for b in maze_walls:
    ax.plot([b[0], b[2]], [b[1], b[3]], "r-", lw=3)
ax.set_title("(d) potential field + trajectories")
fig.savefig(FIG / "p2d_maze.png", dpi=130, bbox_inches="tight")

plt.show()
