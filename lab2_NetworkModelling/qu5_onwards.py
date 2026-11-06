# TMM Lab 02 - Questions 5 onwards: real OpenStreetMap (OSM) data for Bristol,
# focused on a 3 x 3 km box around postcode BS8 1UB (University / Clifton / centre).
#
# Run top to bottom, or cell by cell in VS Code (each "# %%" line starts a cell).
# Each question starts at a banner like:   # %% ===== Q5 ... =====
# Outputs: ./data (extracts, tables) and ./figures (plots).


# %% ========================================================================
# ===== SETUP: imports, paths, area of interest (used by every question) =====
# ===========================================================================
import os
import time
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmium
import osmnx as ox

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SOURCE = os.path.join(HERE, "Bristol.osm.gz")       # lecturer's file: Bristol + Bath region
EXTRACT_XML = os.path.join(DATA_DIR, "bs8.osm")      # our 3 x 3 km box, XML format
EXTRACT_PBF = os.path.join(DATA_DIR, "bs8.osm.pbf")  # same box, binary PBF format

POSTCODE = "BS8 1UB"
FALLBACK_LATLON = (51.455846, -2.603077)             # BS8 1UB, from api.postcodes.io
HALF_SIZE_KM = 1.5                                   # box is 2 x 1.5 = 3 km across

# road classes treated as 'streets' (vehicle network); footpaths, cycleways etc. excluded
STREET_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                "residential", "living_street", "service",
                "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
MAJOR_TYPES = {"motorway", "trunk", "primary", "secondary", "tertiary",
               "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}


def show_now():
    """Pop the figures made so far up straight away, without pausing the script.
    (The plt.show() at the very end then keeps every window open until you close it.)"""
    plt.show(block=False)
    plt.pause(0.5)


def lookup_postcode(postcode):
    """(lat, lon) of a UK postcode via the free postcodes.io API; fallback if offline."""
    try:
        url = "https://api.postcodes.io/postcodes/" + postcode.replace(" ", "")
        res = json.load(urllib.request.urlopen(url, timeout=10))["result"]
        return res["latitude"], res["longitude"]
    except Exception as err:
        print(f"postcode lookup failed ({err}); using stored coordinates")
        return FALLBACK_LATLON


def bbox_around(lat, lon, half_km):
    """(west, south, east, north) in degrees for a square of half-width half_km."""
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * np.cos(np.radians(lat)))
    return tuple(float(v) for v in (lon - dlon, lat - dlat, lon + dlon, lat + dlat))


# --- cutting the box out of the big file (pyosmium; same job as `osmium extract`, see Q7) ---
# Bristol.osm.gz has 3.9 million nodes - far too many to load and plot directly - so
# every question below works on this smaller extract.
class FindWaysInBox(osmium.SimpleHandler):
    """Pass 1: nodes inside the box, then every way touching one of them (whole way kept,
    so streets crossing the box edge are not chopped) and all the nodes those ways need."""
    def __init__(self, bbox):
        super().__init__()
        self.west, self.south, self.east, self.north = bbox
        self.inside, self.ways, self.needed = set(), set(), set()

    def node(self, n):
        loc = n.location
        if self.west <= loc.lon <= self.east and self.south <= loc.lat <= self.north:
            self.inside.add(n.id)

    def way(self, w):
        refs = [r.ref for r in w.nodes]
        if any(r in self.inside for r in refs):
            self.ways.add(w.id)
            self.needed.update(refs)


class WriteSelected(osmium.SimpleHandler):
    """Pass 2: copy the selected nodes and ways to one or more output files
    (format chosen by extension: .osm = XML, .osm.pbf = binary)."""
    def __init__(self, writers, node_ids, way_ids):
        super().__init__()
        self.writers, self.node_ids, self.way_ids = writers, node_ids, way_ids

    def node(self, n):
        if n.id in self.node_ids:
            for wr in self.writers:
                wr.add_node(n)

    def way(self, w):
        if w.id in self.way_ids:
            for wr in self.writers:
                wr.add_way(w)


def extract_bbox(src, bbox, out_paths, force=False):
    """Write the box extract to every path in out_paths. Skipped if the files exist."""
    if not force and all(os.path.exists(p) for p in out_paths):
        print("extract already exists - skipping (force=True to redo)")
        return
    t = time.time()
    finder = FindWaysInBox(bbox)
    finder.apply_file(src)
    print(f"pass 1: {len(finder.inside)} nodes in box, {len(finder.ways)} ways touch it "
          f"({time.time() - t:.0f} s)")
    writers = [osmium.SimpleWriter(p, overwrite=True) for p in out_paths]
    WriteSelected(writers, finder.needed, finder.ways).apply_file(src)
    for wr in writers:
        wr.close()
    print(f"pass 2: wrote {len(finder.needed)} nodes, {len(finder.ways)} ways "
          f"({time.time() - t:.0f} s total)")


# run the setup
centre = lookup_postcode(POSTCODE)
bbox = bbox_around(*centre, HALF_SIZE_KM)
print(f"{POSTCODE}: lat {centre[0]:.5f}, lon {centre[1]:.5f}")
print(f"box (W, S, E, N): {tuple(round(v, 5) for v in bbox)}")
extract_bbox(SOURCE, bbox, [EXTRACT_XML, EXTRACT_PBF])     # ~2 min the first time only


# %% ========================================================================
# ===== Q5: read the Bristol OSM data and plot it ==========================
# ===========================================================================
# MATLAB: readgeotable. Python: osmnx.features_from_xml -> a GeoDataFrame (a table with
# one row per OSM feature and a geometry column). Like readgeotable it gives shapes,
# not a network: closed ways become polygons, open ways become lines.

def load_features(osm_xml):
    t = time.time()
    gdf = ox.features_from_xml(osm_xml)
    print(f"\nQ5: loaded {len(gdf)} features in {time.time() - t:.0f} s")
    return gdf


def plot_bristol_map(gdf, bbox, centre, title):
    """Buildings, green space, water and streets (orange = major road)."""
    def has(col, values=None):
        if col not in gdf:
            return np.zeros(len(gdf), dtype=bool)
        return gdf[col].notna().values if values is None else gdf[col].isin(values).values

    is_line = gdf.geom_type.isin(["LineString", "MultiLineString"]).values
    is_poly = gdf.geom_type.isin(["Polygon", "MultiPolygon"]).values
    hw = gdf["highway"] if "highway" in gdf else pd.Series(index=gdf.index, dtype=object)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf[is_poly & (has("leisure", ["park", "garden"]) | has("landuse", ["grass", "recreation_ground"]))] \
        .plot(ax=ax, color="#cfe8c4", linewidth=0)
    gdf[is_poly & has("natural", ["water"])].plot(ax=ax, color="#a9cce3", linewidth=0)
    gdf[has("waterway")].plot(ax=ax, color="#5d9cc9", linewidth=1.5)
    gdf[is_poly & has("building")].plot(ax=ax, color="#d5d0c8", linewidth=0)
    gdf[is_line & hw.isin(["footway", "path", "pedestrian", "steps", "cycleway"]).values] \
        .plot(ax=ax, color="#999999", linewidth=0.3)
    gdf[is_line & hw.isin(set(STREET_TYPES) - MAJOR_TYPES).values].plot(ax=ax, color="#444444", linewidth=0.7)
    gdf[is_line & hw.isin(MAJOR_TYPES).values].plot(ax=ax, color="#d35400", linewidth=1.8)
    gdf[has("railway", ["rail"]) & is_line].plot(ax=ax, color="k", linewidth=1.0, linestyle="--")

    ax.plot(centre[1], centre[0], "r*", ms=18, mec="k", label=POSTCODE)
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(1 / np.cos(np.radians(centre[0])))   # degrees -> roughly equal km on both axes
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(FIG_DIR, "Q5_bristol_map.png"), dpi=130, bbox_inches="tight")


# run Q5
gdf = load_features(EXTRACT_XML)
plot_bristol_map(gdf, bbox, centre, f"OpenStreetMap around {POSTCODE} (3 x 3 km)")
show_now()


# %% ========================================================================
# ===== Q6: alternative OSM sources and the binary PBF format ==============
# ===========================================================================
# Sources:
#   - Geofabrik regional extracts (PBF), e.g. Bristol:
#       https://download.geofabrik.de/europe/united-kingdom/england/bristol-latest.osm.pbf
#   - Overpass API (live download of a small area) - osmnx does this for you, see below
#   - BBBike city extracts: https://extract.bbbike.org/
# Format comparison: the same data as gzipped XML, plain XML and binary PBF.

class Counter(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.nodes = self.ways = 0

    def node(self, n):
        self.nodes += 1

    def way(self, w):
        self.ways += 1


def compare_formats(paths):
    """File size and time for one full read of each file."""
    print(f"\nQ6: {'file':<26} {'size MB':>8} {'read s':>7} {'nodes':>9} {'ways':>8}")
    for p in paths:
        c, t = Counter(), time.time()
        c.apply_file(p)
        print(f"    {os.path.basename(p):<26} {os.path.getsize(p) / 1e6:8.1f} "
              f"{time.time() - t:7.2f} {c.nodes:9d} {c.ways:8d}")


def download_area_overpass(centre, dist_m=1500):
    """Alternative source: download the drivable street network live from Overpass.
    simplify=False keeps every shape node, to match the raw file."""
    t = time.time()
    G = ox.graph_from_point(centre, dist=dist_m, dist_type="bbox", network_type="drive",
                            simplify=False)
    print(f"    Overpass download: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"({time.time() - t:.0f} s)")
    return G


# run Q6
compare_formats([SOURCE, EXTRACT_XML, EXTRACT_PBF])     # whole region file takes ~30 s
RUN_DOWNLOAD = False                                     # set True to try Overpass (needs internet)
if RUN_DOWNLOAD:
    G_overpass = download_area_overpass(centre)


# %% ========================================================================
# ===== Q7: simple manipulations with osmium ================================
# ===========================================================================
# The osmium command-line tool (osmium-tool) is not pip-installable on Windows; install
# with `conda install -c conda-forge osmium-tool` or use WSL. The pyosmium functions here
# do the same jobs as these commands:
#   osmium fileinfo -e bs8.osm.pbf                                  -> file_info()
#   osmium extract -b W,S,E,N Bristol.osm.gz -o bs8.osm.pbf         -> extract_bbox() (Setup)
#   osmium tags-filter bs8.osm.pbf w/highway -o bs8_roads.osm.pbf   -> tags_filter()
#   osmium cat bs8.osm -o bs8.osm.pbf                               -> any writer: format from extension

class InfoCounter(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.n = self.w = self.r = 0
        self.lon, self.lat = [np.inf, -np.inf], [np.inf, -np.inf]
        self.tag_keys = {}

    def node(self, n):
        self.n += 1
        lo, la = n.location.lon, n.location.lat
        self.lon = [min(self.lon[0], lo), max(self.lon[1], lo)]
        self.lat = [min(self.lat[0], la), max(self.lat[1], la)]

    def way(self, w):
        self.w += 1
        for t in w.tags:
            self.tag_keys[t.k] = self.tag_keys.get(t.k, 0) + 1

    def relation(self, r):
        self.r += 1


def file_info(path, top_keys=8):
    """Like `osmium fileinfo -e`: counts, actual extent, and the most common way tags."""
    c = InfoCounter()
    c.apply_file(path)
    print(f"\nQ7: {os.path.basename(path)}: {c.n} nodes, {c.w} ways, {c.r} relations")
    print(f"    lon {c.lon[0]:.4f} to {c.lon[1]:.4f}, lat {c.lat[0]:.4f} to {c.lat[1]:.4f}")
    top = sorted(c.tag_keys.items(), key=lambda kv: -kv[1])[:top_keys]
    print("    most common way tags: " + ", ".join(f"{k} ({v})" for k, v in top))


class FindTaggedWays(osmium.SimpleHandler):
    def __init__(self, key):
        super().__init__()
        self.key, self.ways, self.needed = key, set(), set()

    def way(self, w):
        if self.key in w.tags:
            self.ways.add(w.id)
            self.needed.update(r.ref for r in w.nodes)


def tags_filter(src, dst, key="highway"):
    """Like `osmium tags-filter src w/<key>`: keep ways with that tag + the nodes they use."""
    f = FindTaggedWays(key)
    f.apply_file(src)
    wr = osmium.SimpleWriter(dst, overwrite=True)
    WriteSelected([wr], f.needed, f.ways).apply_file(src)
    wr.close()
    print(f"    tags-filter w/{key}: kept {len(f.ways)} ways, {len(f.needed)} nodes -> "
          f"{os.path.basename(dst)} ({os.path.getsize(dst) / 1e6:.2f} MB)")


# run Q7
file_info(EXTRACT_PBF)
ROADS_PBF = os.path.join(DATA_DIR, "bs8_roads.osm.pbf")
tags_filter(EXTRACT_PBF, ROADS_PBF, key="highway")


# %% ========================================================================
# ===== Q8: 'nodes' and 'ways' tables =======================================
# ===========================================================================
# The lecturer's livescript uses readstruct (slow) to build these; the .mat product
# files are not in this folder, so we build the same two tables straight from the PBF:
#   nodes: id, lon, lat, x_km, y_km  (flat local projection, origin at BS8 1UB)
#   ways : id, highway, name, oneway, node_ids (ordered list), n_nodes

class TableBuilder(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.node_rows, self.way_rows = [], []

    def node(self, n):
        self.node_rows.append((n.id, n.location.lon, n.location.lat))

    def way(self, w):
        if "highway" in w.tags:
            self.way_rows.append((w.id, w.tags.get("highway"), w.tags.get("name", ""),
                                  w.tags.get("oneway", "no"), [r.ref for r in w.nodes]))


def build_tables(osm_path, centre, street_types=STREET_TYPES):
    """Only street-type ways are kept, and only the nodes they use."""
    b = TableBuilder()
    b.apply_file(osm_path)
    ways = pd.DataFrame(b.way_rows, columns=["id", "highway", "name", "oneway", "node_ids"])
    ways = ways[ways["highway"].isin(street_types)].reset_index(drop=True)
    ways["n_nodes"] = ways["node_ids"].apply(len)

    nodes = pd.DataFrame(b.node_rows, columns=["id", "lon", "lat"])
    used = set(np.concatenate(ways["node_ids"].values))
    nodes = nodes[nodes["id"].isin(used)].reset_index(drop=True)
    lat0, lon0 = centre
    nodes["x_km"] = (nodes["lon"] - lon0) * 111.32 * np.cos(np.radians(lat0))
    nodes["y_km"] = (nodes["lat"] - lat0) * 111.32
    return nodes, ways


def save_tables(nodes, ways):
    nodes.to_csv(os.path.join(DATA_DIR, "bs8_nodes.csv"), index=False)
    out = ways.copy()
    out["node_ids"] = out["node_ids"].apply(lambda r: " ".join(map(str, r)))  # CSV-friendly
    out.to_csv(os.path.join(DATA_DIR, "bs8_ways.csv"), index=False)
    print(f"    tables saved to {DATA_DIR}")


def summarise_tables(nodes, ways):
    print(f"\nQ8: nodes table {len(nodes)} rows, ways table {len(ways)} rows")
    print("    ways per road type: " +
          ", ".join(f"{k} {v}" for k, v in ways["highway"].value_counts().items()))
    print(f"    nodes per way: median {ways['n_nodes'].median():.0f}, max {ways['n_nodes'].max()}")
    # preview for Q9: nodes shared by 2+ ways are candidate junctions
    uses = pd.Series(np.concatenate(ways["node_ids"].values)).value_counts()
    print(f"    nodes used by 2+ ways: {(uses >= 2).sum()} of {len(nodes)} "
          f"({100 * (uses >= 2).mean():.0f}%) - the rest only trace the shape of a street")


def plot_tables(nodes, ways, title):
    """Redraw the street network purely from the two tables (checks they are consistent)."""
    xy = nodes.set_index("id")[["x_km", "y_km"]]
    fig, ax = plt.subplots(figsize=(9, 9))
    for _, w in ways.iterrows():
        pts = xy.loc[w["node_ids"]].values
        major = w["highway"] in MAJOR_TYPES
        ax.plot(pts[:, 0], pts[:, 1], color="#d35400" if major else "#444444",
                lw=1.6 if major else 0.6)
    ax.plot(nodes["x_km"], nodes["y_km"], ".", ms=0.8, color="tab:blue", alpha=0.5)
    ax.plot(0, 0, "r*", ms=16, mec="k", label=POSTCODE)
    ax.set_xlim(-HALF_SIZE_KM, HALF_SIZE_KM); ax.set_ylim(-HALF_SIZE_KM, HALF_SIZE_KM)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km east of BS8 1UB)"); ax.set_ylabel("y (km north of BS8 1UB)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(FIG_DIR, "Q8_tables_plot.png"), dpi=130, bbox_inches="tight")


# run Q8
nodes, ways = build_tables(EXTRACT_PBF, centre)
summarise_tables(nodes, ways)
save_tables(nodes, ways)
plot_tables(nodes, ways, f"Street network from the nodes/ways tables around {POSTCODE}")
show_now()


# %% ========================================================================
# ===== Q9*: nodes + ways -> junction-only graph with arc lengths ==========
# ===========================================================================
# TODO: keep only junction nodes (shared by 2+ ways, plus way end points); split each
# way at its junctions; edge weight = arc length along the way between junctions.
# Then compare with osmnx's simplify_graph.


# %% ========================================================================
# ===== Q10*: shortest paths as edges are deleted / added ===================
# ===========================================================================
# TODO


# %% ========================================================================
# ===== Q11*: how well do beta-skeletons fit the real network? ==============
# ===========================================================================
# TODO


# %% Keep all figures open
print(f"\nAll done. Figures are also saved in {FIG_DIR}")
print("Close the figure windows to end the script.")
plt.show()
