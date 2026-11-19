# TMM Coursework Q2(a): timetable -> directed graph -> journey search
# Run cell-by-cell or: python q2a_time_expanded.py
# Draws the one diagram for the part (a) answer and checks the worked example.
# The toy timetable uses GTFS column names (trips.txt, stop_times.txt).

# %% Imports and helpers
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)


def mins(s):
    """Time string 'HH:MM:SS' or 'HH:MM' -> minutes after midnight."""
    p = [int(x) for x in s.split(":")]
    return p[0] * 60 + p[1] + (p[2] / 60 if len(p) > 2 else 0)


def hm(t):
    """Minutes after midnight -> 'HH:MM'."""
    t = int(round(t))
    return f"{t // 60:02d}:{t % 60:02d}"


# %% The toy timetable: 3 routes, 5 trips, 5 stops
trips = pd.DataFrame({
    "trip_id":  ["R1", "R2", "B1", "B2", "G1"],
    "route_id": ["Red", "Red", "Blue", "Blue", "Green"],
})
stop_times = pd.DataFrame(
    [  # trip, seq, stop, arrival, departure
        ("R1", 1, "A", "08:00", "08:00"),
        ("R1", 2, "B", "08:10", "08:11"),
        ("R1", 3, "C", "08:20", "08:20"),
        ("R2", 1, "A", "08:30", "08:30"),
        ("R2", 2, "B", "08:40", "08:41"),
        ("R2", 3, "C", "08:50", "08:50"),
        ("B1", 1, "B", "08:14", "08:14"),
        ("B1", 2, "D", "08:27", "08:27"),
        ("B2", 1, "B", "08:46", "08:46"),
        ("B2", 2, "D", "08:59", "08:59"),
        ("G1", 1, "E", "08:25", "08:25"),
        ("G1", 2, "D", "08:35", "08:35"),
    ],
    columns=["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time"],
)
walks = [("C", "E", 3.0), ("E", "C", 3.0)]     # (from stop, to stop, walking minutes)


# %% Building the graph
# Nodes - one per timetable event:
#   ("dep", trip, stop)  trip leaves stop     ("arr", trip, stop)  trip reaches stop
# Edges - one per move a passenger can make (weight = minutes it takes):
#   ride    dep(trip, s)  -> arr(trip, next stop)   stay on to the next stop
#   stay    arr(trip, s)  -> dep(trip, s)           stay on while the bus waits at s
#   change  arr(trip1, s) -> dep(trip2, s)          trip2 leaves s at least min_transfer later
#   walk    arr(trip1, s) -> dep(trip2, s2)         s2 nearby, trip2 leaves after the walk

def build_graph(stop_times, trips, min_transfer=0.0, walks=()):
    route_of = dict(zip(trips.trip_id, trips.route_id))
    G = nx.DiGraph()

    # nodes, ride edges and stay edges: one trip at a time
    for trip, calls in stop_times.sort_values("stop_sequence").groupby("trip_id"):
        calls = calls.to_dict("records")
        for i, c in enumerate(calls):
            dep, arr = ("dep", trip, c["stop_id"]), ("arr", trip, c["stop_id"])
            info = dict(stop=c["stop_id"], trip=trip, route=route_of[trip])
            if i > 0:                       # nobody arrives at the first stop
                G.add_node(arr, kind="arr", t=mins(c["arrival_time"]), **info)
            if i < len(calls) - 1:          # nobody departs from the last stop
                G.add_node(dep, kind="dep", t=mins(c["departure_time"]), **info)
            if i > 0:
                prev = ("dep", trip, calls[i - 1]["stop_id"])
                G.add_edge(prev, arr, kind="ride", weight=G.nodes[arr]["t"] - G.nodes[prev]["t"])
            if 0 < i < len(calls) - 1:
                G.add_edge(arr, dep, kind="stay", weight=G.nodes[dep]["t"] - G.nodes[arr]["t"])

    # change and walk edges: from each arrival to every later departure it can catch
    arrivals = [(v, d) for v, d in G.nodes(data=True) if d["kind"] == "arr"]
    departures = [(v, d) for v, d in G.nodes(data=True) if d["kind"] == "dep"]
    for a, da in arrivals:
        reachable = [(da["stop"], min_transfer, "change")] + \
                    [(s2, w, "walk") for s1, s2, w in walks if s1 == da["stop"]]
        for stop, delay, kind in reachable:
            for b, db in departures:
                if db["stop"] == stop and db["trip"] != da["trip"] and db["t"] >= da["t"] + delay:
                    G.add_edge(a, b, kind=kind, weight=db["t"] - da["t"])
    return G


# %% Journey search = shortest path from START to END
def journey_search(G, origin, dest, t0):
    """Earliest arrival at stop `dest`, leaving stop `origin` at or after time t0."""
    G.add_edges_from([("START", v, dict(weight=d["t"] - t0)) for v, d in G.nodes(data=True)
                      if d.get("kind") == "dep" and d["stop"] == origin and d["t"] >= t0])
    G.add_edges_from([(v, "END", dict(weight=0.0)) for v, d in G.nodes(data=True)
                      if d.get("kind") == "arr" and d["stop"] == dest])
    try:
        path = nx.dijkstra_path(G, "START", "END")[1:-1]
    except nx.NetworkXNoPath:
        path = None
    G.remove_nodes_from(["START", "END"])
    return path


def print_journey(G, path, t0):
    if path is None:
        print("  no journey")
        return
    for u, v in zip(path[:-1], path[1:]):
        e, nu, nv = G.edges[u, v], G.nodes[u], G.nodes[v]
        if e["kind"] == "ride":
            print(f"  ride {nu['route']:5s} trip {nu['trip']}: {nu['stop']} {hm(nu['t'])} -> "
                  f"{nv['stop']} {hm(nv['t'])}")
        elif e["kind"] in ("change", "walk"):
            print(f"  {e['kind']} at {nu['stop']}" +
                  (f" -> {nv['stop']}" if e["kind"] == "walk" else "") +
                  f", wait until {hm(nv['t'])}")
    t_end = G.nodes[path[-1]]["t"]
    print(f"  arrive {hm(t_end)} ({t_end - t0:.0f} min)")


# %% The three versions of the graph, query: A -> D leaving 08:00
t0 = mins("08:00")
versions = [
    ("1. lazy passenger", dict()),
    ("2. minimum transfer time 5 min", dict(min_transfer=5)),
    ("3. 5 min transfers + walking C<->E", dict(min_transfer=5, walks=walks)),
]
for name, kw in versions:
    G = build_graph(stop_times, trips, **kw)
    assert nx.is_directed_acyclic_graph(G)          # time never goes backwards
    kinds = pd.Series([d["kind"] for *_, d in G.edges(data=True)]).value_counts()
    print(f"\n{name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges {dict(kinds)}")
    print_journey(G, journey_search(G, "A", "D", t0), t0)
# expected arrivals: 1. 08:27   2. 08:59   3. 08:35


# %% The diagram: the lazy-passenger graph, with the two improvements marked on it
G = build_graph(stop_times, trips, walks=walks)     # lazy graph + the walk edge
lazy_path = journey_search(build_graph(stop_times, trips), "A", "D", t0)

ROW = {"A": 4, "B": 3, "C": 2, "E": 1, "D": 0}
colour = {"Red": "tab:red", "Blue": "tab:blue", "Green": "tab:green"}
pos = {v: (d["t"], ROW[d["stop"]] + (0.15 if d["kind"] == "dep" else -0.15))
       for v, d in G.nodes(data=True)}
banned = (("arr", "R1", "B"), ("dep", "B1", "B"))   # the 4 min change at B

fig, ax = plt.subplots(figsize=(11, 6.5))
nx.draw_networkx_edges(G, pos, edgelist=list(zip(lazy_path[:-1], lazy_path[1:])), ax=ax,
                       width=8, edge_color="gold", alpha=0.6, arrows=False)
styles = {"ride": (None, "solid", 2.2), "stay": ("grey", "dashed", 1.2),
          "change": ("tab:orange", "solid", 1.4), "walk": ("tab:purple", "dotted", 2.0)}
for kind, (col, ls, lw) in styles.items():
    el = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == kind]
    ec = [colour[G.nodes[u]["route"]] for u, v in el] if col is None else col
    nx.draw_networkx_edges(G, pos, edgelist=el, ax=ax, edge_color=ec, style=ls, width=lw,
                           arrows=True, arrowsize=10, node_size=60)
for kind, marker in [("dep", "^"), ("arr", "v")]:
    nl = [v for v, d in G.nodes(data=True) if d["kind"] == kind]
    nx.draw_networkx_nodes(G, pos, nodelist=nl, ax=ax, node_shape=marker, node_size=60,
                           node_color=[colour[G.nodes[v]["route"]] for v in nl],
                           edgecolors="k", linewidths=0.5)
for v, d in G.nodes(data=True):
    ax.annotate(f"{d['trip']} {hm(d['t'])}", pos[v], fontsize=6.5,
                xytext=(3, 4) if d["kind"] == "dep" else (3, -10), textcoords="offset points")

# mark the two improvements
mid = ((pos[banned[0]][0] + pos[banned[1]][0]) / 2, (pos[banned[0]][1] + pos[banned[1]][1]) / 2)
ax.text(*mid, "X", color="red", fontsize=13, fontweight="bold", ha="center", va="center")
ax.annotate("Improvement 1: this 4 min change is removed\nif the minimum transfer time is 5 min",
            mid, xytext=(mins("08:16"), 3.6), fontsize=8, color="red",
            arrowprops=dict(arrowstyle="-", color="red", lw=0.7))
w = (("arr", "R1", "C"), ("dep", "G1", "E"))
ax.annotate("Improvement 2: walk edge C -> E\n(3 min walk, then wait for G1)",
            ((pos[w[0]][0] + pos[w[1]][0]) / 2, (pos[w[0]][1] + pos[w[1]][1]) / 2),
            xytext=(mins("08:28"), 1.75), fontsize=8, color="tab:purple",
            arrowprops=dict(arrowstyle="-", color="tab:purple", lw=0.7))

ax.set_axis_on()
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.set_yticks(list(ROW.values()), [f"stop {s}" for s in ROW])
ticks = range(int(mins("08:00")), int(mins("09:00")) + 1, 10)
ax.set_xticks(list(ticks), [hm(t) for t in ticks])
ax.set_xlim(mins("07:57"), mins("09:04")); ax.set_ylim(-0.6, 4.7)
ax.set_xlabel("time"); ax.grid(axis="y", color="0.92")
ax.set_title("Timetable as a directed graph: lazy passenger, journey A -> D leaving 08:00 "
             f"(arrive {hm(G.nodes[lazy_path[-1]]['t'])})", fontsize=10)
handles = [Line2D([], [], marker="^", ls="", color="k", label="node: trip departs stop"),
           Line2D([], [], marker="v", ls="", color="k", label="node: trip arrives at stop"),
           Line2D([], [], color="k", lw=2.2, label="ride edge (route colour)"),
           Line2D([], [], color="grey", ls="--", label="stay-on edge"),
           Line2D([], [], color="tab:orange", lw=1.4, label="change edge (same stop, later)"),
           Line2D([], [], color="tab:purple", ls=":", lw=2, label="walk edge (improvement 2)"),
           Line2D([], [], color="gold", lw=8, alpha=0.6, label="journey found (shortest path)")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "q2a_time_expanded.png", dpi=150)
plt.show()
