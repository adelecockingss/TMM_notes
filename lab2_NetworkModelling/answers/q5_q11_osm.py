# TMM Lab 02 - Q5-Q11: real street networks from OpenStreetMap
# Run cell-by-cell or: python q5_q11_osm.py   (first run ~1 min to read the
# 71 MB Bristol.osm.gz; after that the tables in data/ are reused)
# Writes data/<city>_nodes.csv, _ways.csv, _junctions.csv, _edges.csv, which
# the MATLAB script q9_q11_osm.m reads (the "prepare in Python, analyse in
# MATLAB" route the lab sheet mentions).

# %% Imports and settings
from pathlib import Path
import time
import numpy as np
import pandas as pd
import osmium
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse.csgraph import dijkstra
import scipy.sparse as sp

from net_utils import beta_skeleton

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
OSM_FILE = HERE.parent / "Bristol.osm.gz"

CITIES = {"bristol": (51.4558, -2.6031),     # BS8 1UB: Clifton / city centre
          "bath": (51.3811, -2.3590)}        # Bath city centre
HALF_KM = 1.5                                 # 3 x 3 km boxes
STREETS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
           "residential", "living_street", "motorway_link", "trunk_link",
           "primary_link", "secondary_link", "tertiary_link"}


def to_km(lat, lon, centre):
    """Local flat projection (fine over a few km): km east/north of centre."""
    lat0, lon0 = centre
    return ((np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)),
            (np.asarray(lat) - lat0) * 111.32)


# %% Q5/Q8: read the OSM file ONCE and build 'nodes' and 'ways' tables
# OSM data model: a NODE is a point (id, lat, lon); a WAY is an ordered list
# of node ids plus tags (highway=residential, name=..., oneway=yes...).
# Roads are ways; their nodes include both junctions and points that just
# trace the curve of the road.
class StreetHandler(osmium.SimpleHandler):
    def __init__(self, boxes):
        super().__init__()
        self.boxes = boxes                      # {city: (W, S, E, N) in degrees}
        self.ways = {c: [] for c in boxes}
        self.nodes = {c: {} for c in boxes}

    def way(self, w):
        hw = w.tags.get("highway")
        if hw not in STREETS:
            return
        ids = [n.ref for n in w.nodes]
        lon = np.array([n.lon for n in w.nodes]); lat = np.array([n.lat for n in w.nodes])
        for city, (W, S, E, N) in self.boxes.items():
            inside = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
            if inside.any():
                self.ways[city].append((w.id, hw, w.tags.get("name", ""),
                                        w.tags.get("oneway", "no"),
                                        " ".join(map(str, ids))))
                for i, x, y in zip(ids, lon, lat):
                    self.nodes[city][i] = (x, y)


def load_tables(city):
    f_n, f_w = DATA / f"{city}_nodes.csv", DATA / f"{city}_ways.csv"
    if not f_n.exists():
        boxes = {}
        for c, (lat, lon) in CITIES.items():
            dlat = HALF_KM / 111.32; dlon = dlat / np.cos(np.radians(lat))
            boxes[c] = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        t = time.time()
        h = StreetHandler(boxes)
        h.apply_file(str(OSM_FILE), locations=True)   # locations: give ways their coords
        print(f"read {OSM_FILE.name} in {time.time() - t:.0f} s")
        for c in CITIES:
            nd = pd.DataFrame([(i, x, y) for i, (x, y) in h.nodes[c].items()],
                              columns=["id", "lon", "lat"])
            nd["x_km"], nd["y_km"] = to_km(nd["lat"], nd["lon"], CITIES[c])
            nd.to_csv(DATA / f"{c}_nodes.csv", index=False)
            pd.DataFrame(h.ways[c], columns=["id", "highway", "name", "oneway", "node_ids"]) \
                .to_csv(DATA / f"{c}_ways.csv", index=False)
    nodes = pd.read_csv(f_n)
    ways = pd.read_csv(f_w, keep_default_na=False)
    ways["node_ids"] = ways["node_ids"].str.split().apply(lambda r: [int(x) for x in r])
    return nodes, ways


tables = {c: load_tables(c) for c in CITIES}
for c, (nodes, ways) in tables.items():
    print(f"{c}: {len(nodes)} nodes, {len(ways)} street ways")

# %% Q5: plot the streets
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
for ax, (c, (nodes, ways)) in zip(axs, tables.items()):
    xy = nodes.set_index("id")[["x_km", "y_km"]]
    major = {"motorway", "trunk", "primary", "secondary", "trunk_link", "primary_link"}
    for _, w in ways.iterrows():
        p = xy.loc[w["node_ids"]].to_numpy()
        ax.plot(p[:, 0], p[:, 1], color="tab:orange" if w["highway"] in major else "k",
                lw=1.5 if w["highway"] in major else 0.5)
    ax.set_xlim(-HALF_KM, HALF_KM); ax.set_ylim(-HALF_KM, HALF_KM)
    ax.set_aspect("equal"); ax.set_title(f"{c}: OSM streets (orange = major)")
fig.savefig(FIG / "q5_osm_streets.png", dpi=130, bbox_inches="tight")

# %% Q6/Q7 (see NOTES.md for sources and osmium commands). Quick file info:
for c, (nodes, ways) in tables.items():
    print(c, ways["highway"].value_counts().head(6).to_dict())


# %% Q9: nodes + ways -> junction graph with arc lengths
def junction_graph(nodes, ways):
    """Keep only nodes that are junctions (used by 2+ ways, or used twice in
    one way, e.g. a loop) or way end points; split every way at its junctions.
    Edge length = arc length along the way (sum of the small segments).
    oneway=yes -> one directed edge u->v; '-1' -> v->u; otherwise two-way."""
    xy = nodes.set_index("id")[["x_km", "y_km"]]
    uses = pd.Series(np.concatenate(ways["node_ids"].values)).value_counts()
    ends = set(ways["node_ids"].str[0]) | set(ways["node_ids"].str[-1])
    junction = set(uses.index[uses >= 2]) | ends
    rows = []
    for _, w in ways.iterrows():
        ids = w["node_ids"]
        p = xy.loc[ids].to_numpy()
        seg = np.hypot(*np.diff(p, axis=0).T)
        start, acc = ids[0], 0.0
        for k in range(1, len(ids)):
            acc += seg[k - 1]
            if ids[k] in junction:
                if ids[k] != start:
                    rows.append((start, ids[k], acc, w["oneway"], w["highway"], w["name"]))
                start, acc = ids[k], 0.0
    E = pd.DataFrame(rows, columns=["u_id", "v_id", "length_km", "oneway", "highway", "name"])
    J = nodes[nodes["id"].isin(set(E["u_id"]) | set(E["v_id"]))].reset_index(drop=True)
    idx = pd.Series(np.arange(len(J)), index=J["id"])
    E["u"], E["v"] = idx[E["u_id"]].to_numpy(), idx[E["v_id"]].to_numpy()
    return J, E


graphs = {}
for c, (nodes, ways) in tables.items():
    J, E = junction_graph(nodes, ways)
    graphs[c] = (J, E)
    J.to_csv(DATA / f"{c}_junctions.csv", index=False)
    E.to_csv(DATA / f"{c}_edges.csv", index=False)
    deg = np.bincount(np.concatenate([E["u"], E["v"]]), minlength=len(J))
    print(f"Q9 {c}: {len(nodes)} OSM nodes -> {len(J)} junctions, {len(E)} edges; "
          f"total street length {E['length_km'].sum():.1f} km; "
          f"degree counts {np.bincount(deg)[:6].tolist()}")
    # degree 1 = dead ends (or streets cut by the box edge), 3-4 = junctions


def directed_matrix(J, E, removed=()):
    """Sparse weight matrix of the DIRECTED network (one-way streets respected)."""
    keep = ~E.index.isin(removed)
    e = E[keep]
    fwd = e["oneway"] != "-1"
    bwd = ~e["oneway"].isin(["yes", "true", "1"])
    r = np.concatenate([e["u"][fwd], e["v"][bwd]])
    c = np.concatenate([e["v"][fwd], e["u"][bwd]])
    w = np.concatenate([e["length_km"][fwd], e["length_km"][bwd]])
    M = sp.coo_matrix((w, (r, c)), shape=(len(J), len(J))).tocsr()
    return M   # duplicates summed - harmless for a quick demo (rare parallel edges)


def route(M, s, t):
    d, pred = dijkstra(M, indices=s, return_predecessors=True)
    path = [t]
    while path[-1] != s and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    return d[t], path[::-1]


# %% Q10: shortest paths as streets are closed (or added)
J, E = graphs["bristol"]
xy = J[["x_km", "y_km"]].to_numpy()
M = directed_matrix(J, E)
# Only pick start/end nodes in the largest STRONGLY connected component
# (otherwise we may start in a cul-de-sac or at a one-way dead end).
from scipy.sparse.csgraph import connected_components
_, comp = connected_components(M, directed=True, connection="strong")
big = comp == np.bincount(comp).argmax()
cand = np.flatnonzero(big)
s = int(cand[np.argmin(np.hypot(xy[cand, 0] + 1.2, xy[cand, 1] + 1.2))])   # near SW corner
t = int(cand[np.argmin(np.hypot(xy[cand, 0] - 1.2, xy[cand, 1] - 1.2))])   # near NE corner
print(f"largest strongly connected component: {big.sum()} of {len(J)} junctions")
d0, p0 = route(M, s, t)
print(f"\nQ10: route SW -> NE {d0:.2f} km ({len(p0) - 1} edges), "
      f"straight line {np.hypot(*(xy[t] - xy[s])):.2f} km")


def edges_on(path):
    pairs = set(zip(path[:-1], path[1:]))
    return E.index[[(u, v) in pairs or (v, u) in pairs for u, v in zip(E["u"], E["v"])]]


# close each street on the route in turn and see how much longer the trip gets
impact = []
for e in edges_on(p0):
    d, _ = route(directed_matrix(J, E, removed=[e]), s, t)
    impact.append((e, d - d0))
n_cut = sum(np.isinf(x) for _, x in impact)
print(f"   {n_cut} of {len(impact)} streets on the route are BRIDGES: closing one "
      "disconnects s from t")
impact = sorted([x for x in impact if np.isfinite(x[1])], key=lambda x: -x[1])
for e, extra in impact[:5]:
    print(f"   closing {E.loc[e, 'name'] or E.loc[e, 'highway']:30s} adds {extra * 1000:5.0f} m")
worst = impact[0][0]
d1, p1 = route(directed_matrix(J, E, removed=[worst]), s, t)

# add a new street (e.g. a new bridge) between two nodes and see if it is used
a, b = p0[len(p0) // 3], p0[2 * len(p0) // 3]
M2 = M.tolil(); new_len = np.hypot(*(xy[a] - xy[b]))
M2[a, b] = M2[b, a] = new_len
d2, p2 = route(M2.tocsr(), s, t)
print(f"   new straight street of {new_len:.2f} km: route {d0:.2f} -> {d2:.2f} km")

fig, ax = plt.subplots(figsize=(8, 8))
ax.add_collection(LineCollection(xy[E[["u", "v"]].to_numpy()], colors="0.7", linewidths=0.6))
for p, col, lab in [(p0, "tab:blue", f"original {d0:.2f} km"),
                    (p1, "tab:red", f"worst closure {d1:.2f} km"),
                    (p2, "tab:green", f"with new street {d2:.2f} km")]:
    ax.plot(xy[p, 0], xy[p, 1], color=col, lw=2.5, label=lab, alpha=0.8)
ax.plot(*xy[[s, t]].T, "ko", ms=8)
ax.set_aspect("equal"); ax.legend(); ax.set_title("Q10 Bristol: closures and new links")
fig.savefig(FIG / "q10_closures.png", dpi=130, bbox_inches="tight")


# %% Q11: are beta-skeletons good models of real street networks?
def skeleton_fit(J, E, betas):
    """Compare the beta-skeleton of the junction points with the real
    (undirected, straightened) junction graph:
      precision = share of skeleton edges that are real streets
      recall    = share of real streets that the skeleton contains
      F1        = harmonic mean of the two (single score to maximise)."""
    xy = J[["x_km", "y_km"]].to_numpy()
    real = {tuple(sorted(e)) for e in E[["u", "v"]].to_numpy() if e[0] != e[1]}
    out = []
    for b in betas:
        S = {tuple(e) for e in beta_skeleton(xy, b, "lune")}
        tp = len(S & real)
        prec, rec = tp / max(len(S), 1), tp / len(real)
        out.append((b, len(S), prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)))
    return pd.DataFrame(out, columns=["beta", "n_edges", "precision", "recall", "F1"])


betas = np.round(np.arange(1.0, 2.61, 0.1), 2)
fig, ax = plt.subplots(figsize=(7, 4.5))
for c, (J, E) in graphs.items():
    res = skeleton_fit(J, E, betas)
    best = res.loc[res["F1"].idxmax()]
    print(f"\nQ11 {c}: real network has {len(E)} edges; best beta = {best['beta']:.1f} "
          f"(F1 {best['F1']:.3f}, precision {best['precision']:.2f}, recall {best['recall']:.2f})")
    ax.plot(res["beta"], res["F1"], "o-", label=f"{c} F1")
    ax.plot(res["beta"], res["precision"], ":", label=f"{c} precision")
    ax.plot(res["beta"], res["recall"], "--", label=f"{c} recall")
ax.set_xlabel("beta (lune)"); ax.set_ylabel("score"); ax.legend(fontsize=8)
ax.set_title("Q11: overlap between beta-skeleton and real streets")
fig.savefig(FIG / "q11_skeleton_fit.png", dpi=130, bbox_inches="tight")

plt.show()
