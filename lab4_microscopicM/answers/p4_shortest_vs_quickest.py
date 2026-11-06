# TMM Lab 04 - Problem 4: shortest path vs quickest path
# 20 m x 20 m room, barrier at y = 12 with a narrow opening x in [3, 3.9]
# (on the direct line) and a wide opening x in [18, 20] (a detour).
# 1 new pedestrian per second, count how many arrive within 90 s.
# Run: python p4_shortest_vs_quickest.py  (~2 min). SHOW = True to watch.

# %% Imports
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

from ped_common import World, social_force, run, PotentialField

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
SHOW = False
RENDER = 10 if SHOW else 0

walls = [[0, 0, 20, 0], [20, 0, 20, 20], [20, 20, 0, 20], [0, 20, 0, 0],
         [0, 12, 3, 12], [3.9, 12, 18, 12]]            # barrier with the two openings
DEST = (0, 19.2, 2, 20)
narrow = dict(x=(3.0, 3.9), centre=np.array([3.45, 12.0]))
wide = dict(x=(18.0, 20.0), centre=np.array([19.0, 12.0]))

# Two potential fields: route via the narrow opening only / via the wide only
# (block the other opening with a box). 'Shortest path' = the smaller one.
via_narrow = PotentialField((0, 20), (0, 20), walls, DEST, h=0.1, clearance=0.3,
                            blocked_boxes=[(17.9, 11.7, 20.1, 12.3)])
via_wide = PotentialField((0, 20), (0, 20), walls, DEST, h=0.1, clearance=0.3,
                          blocked_boxes=[(2.9, 11.7, 4.0, 12.3)])
start = np.array([2.0, 1.0])
print(f"route length from the source: via narrow {via_narrow.value(start):.1f} m, "
      f"via wide {via_wide.value(start):.1f} m")


def make_world(directions, params, body_radius, seed=0):
    def init(ped, *a):
        if body_radius is not None:
            ped.radius = body_radius()
        ped.metadata["route"] = "narrow"
    w = World({
        "space": {
            "room": {"type": "rectangle", "coordinates": [0, 0, 20, 20], "colour": "gray",
                     "add_boundaries": False},
            "start": {"type": "rectangle", "coordinates": [0, 0, 4, 2], "colour": "green",
                      "add_boundaries": False},
            "goal": {"type": "rectangle", "coordinates": list(DEST), "colour": "red",
                     "add_boundaries": False}},
        "pedestrians": {"p": {"source": "start", "destination": "goal", "colour": "black",
                              "birth_rate": 1.0, "max_count": 200}},
        "boundaries": walls,
        "functions": {"update_directions": directions,
                      "process_interactions": lambda peds, b, p: social_force(peds, b, params),
                      "pedestrian_initialisation": init}}, use_graphics=SHOW)
    random.seed(seed)
    return w


def follow(ped):
    f = via_narrow if ped.metadata["route"] == "narrow" else via_wide
    e = f.direction(ped.pos)
    ped.desired_direction = e if e is not None else np.array([0.0, 1.0])


def shortest(peds, b, p):
    """Everyone takes the geometrically shortest route (through the narrow gap)."""
    for ped in peds.values():
        ped.metadata["route"] = ("narrow" if via_narrow.value(ped.pos) <= via_wide.value(ped.pos)
                                 else "wide")
        follow(ped)


def make_quickest(J_s=1.3, every=1.0):
    """Quickest route: estimated time = walking time + waiting time at the gap,
         T_k = distance_via_k / v0  +  (people queueing at gap k) / (J_s * width_k)
    Queue at gap k = pedestrians below the barrier within 3 m of the gap centre.
    Re-evaluated every `every` seconds by each person still below the barrier."""
    state = {"t": 0.0, "last": -1e9}

    def quickest(peds, b, p):
        state["t"] += 0.05
        if state["t"] - state["last"] >= every:
            state["last"] = state["t"]
            P = np.array([q.pos for q in peds.values()]) if peds else np.zeros((0, 2))
            below = P[:, 1] < 12 if len(P) else np.zeros(0, bool)
            q_n = np.sum(below & (np.linalg.norm(P - narrow["centre"], axis=1) < 3))
            q_w = np.sum(below & (np.linalg.norm(P - wide["centre"], axis=1) < 3))
            for ped in peds.values():
                if ped.pos[1] < 11.5:
                    v0 = max(ped.desired_walking_speed, 0.3)
                    t_n = via_narrow.value(ped.pos) / v0 + q_n / (J_s * 0.9)
                    t_w = via_wide.value(ped.pos) / v0 + q_w / (J_s * 2.0)
                    ped.metadata["route"] = "narrow" if t_n <= t_w else "wide"
        for ped in peds.values():
            follow(ped)
    return quickest


# Two calibrations (see NOTES.md / Problem 2):
calibrations = {
    "provided parameters (r = 0.4-0.5 m)": ({}, None),
    "evacuation calibration (r = 0.23-0.28 m)": ({"B_social": 0.3, "A_social": 2.0},
                                                 lambda: 0.23 + 0.05 * random.random()),
}
results = {}
for cal, (params, radius) in calibrations.items():
    for name, dirs in [("shortest path", lambda: shortest), ("quickest path", make_quickest)]:
        n_out = []
        for seed in range(3):
            world = make_world(dirs(), params, radius, seed)
            ex = run(world, 90.0, 0.05, RENDER)
            n_out.append(len(ex))
        results[(cal, name)] = n_out
        print(f"{cal:42s} {name:14s}: arrived within 90 s = {np.mean(n_out):.1f} "
              f"(runs {n_out})")

fig, ax = plt.subplots(figsize=(7, 4))
labels = [f"{c.split(' (')[0]}\n{n}" for c, n in results]
ax.bar(range(len(results)), [np.mean(v) for v in results.values()],
       yerr=[np.std(v) for v in results.values()], color=["C0", "C1"] * 2)
ax.set_xticks(range(len(results)), labels, fontsize=8)
ax.set_ylabel("people arrived within 90 s"); ax.set_title("Problem 4: shortest vs quickest path")
fig.savefig(FIG / "p4_shortest_vs_quickest.png", dpi=130, bbox_inches="tight")

plt.show()
