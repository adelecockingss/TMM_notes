# TMM Lab 01 - Q1: Getting started with geography
# Run cell-by-cell in VS Code (Shift+Enter) or as a whole: python q1_geography.py
# MATLAB version: q1_geography.m

# %% Imports and data
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from geo_utils import haversine_matrix, latlon_to_osgb, osgb_gridref, voronoi_adjacency

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

msoa = pd.read_csv(HERE.parent / "UK_census_msoa_stats.csv")
lat = msoa["Latitude"].to_numpy()
lon = msoa["Longitude"].to_numpy()

# %% Q1(a) Haversine distances
# Check against a known distance: City of London -> Manchester is ~262 km
places = {"London": (51.5154, -0.0957), "Manchester": (53.4808, -2.2426),
          "Bristol": (51.4545, -2.5879), "Cardiff": (51.4816, -3.1791)}
plat = np.array([p[0] for p in places.values()])
plon = np.array([p[1] for p in places.values()])
D = haversine_matrix(plat, plon)               # 4 x 4 matrix in one call
print(pd.DataFrame(D.round(1), index=places, columns=places))

# Vectorised on the whole dataset: 7201 x 7201 = 52 million distances.
# float64 -> ~400 MB, so only do the full thing if you have the memory.
# Here: distance from every MSOA to Bristol.
d_bristol = haversine_matrix(lat, lon, [51.4545], [-2.5879])[:, 0]
print("Furthest MSOA from Bristol:", msoa["MSOA Name"].iloc[d_bristol.argmax()],
      f"{d_bristol.max():.0f} km")

# %% Q1(b) lat/long -> OS grid (12-digit grid references)
E, N = latlon_to_osgb(lat, lon)
msoa["easting"], msoa["northing"] = E, N
msoa["gridref"] = osgb_gridref(E, N)
print(msoa[["MSOA Name", "Latitude", "Longitude", "gridref"]].head())

# Sanity check 1: on the grid, distance is just Pythagoras (in metres).
# It should agree with Haversine to within ~0.5% (the projection's scale error).
e4, n4 = latlon_to_osgb(plat, plon)
D_grid = np.hypot(e4[:, None] - e4[None, :], n4[:, None] - n4[None, :]) / 1000
print("Grid vs Haversine (% difference):")
print(np.round(100 * (D_grid - D) / np.where(D > 0, D, 1), 2))

# Sanity check 2: compare with the 2021 centroids (supplied as easting/northing
# in the coursework .mat file and exported to data/msoa2021_centroids.csv).
# The MSOAs common to both censuses should be within a few hundred metres.
c21 = pd.read_csv(HERE / "data" / "msoa2021_centroids.csv")
both = msoa.merge(c21, left_on="MSOA Code", right_on="code", suffixes=("", "_21"))
gap = np.hypot(both["easting"] - both["easting_21"], both["northing"] - both["northing_21"])
print(f"{len(both)} MSOAs in both: median centroid gap {gap.median():.0f} m")

# %% Q1(c) Plotting over a map of the UK
# Option 1 (needs internet + `pip install contextily`): web map tiles behind
# the points. Option 2 (always works): the 7201 centroids themselves trace
# out England and Wales well enough to recognise.
fig, ax = plt.subplots(figsize=(6, 7))
sc = ax.scatter(E / 1000, N / 1000, c=msoa["Townsend deprivation score"],
                s=2, cmap="RdYlBu_r")
fig.colorbar(sc, label="Townsend deprivation score")
ax.set_aspect("equal")                          # 1 km east = 1 km north
ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
ax.set_title("MSOA centroids, coloured by deprivation")
try:
    import contextily as cx
    cx.add_basemap(ax, crs="EPSG:27700", source=cx.providers.OpenStreetMap.Mapnik,
                   zoom=7)
    # NB contextily wants coordinates in metres, so for this branch you would
    # plot E, N rather than E/1000, N/1000.
except Exception:
    pass
fig.savefig(FIG / "q1c_msoa_map.png", dpi=150, bbox_inches="tight")

# %% Q1(d) Voronoi tessellation + adjacency matrix
# Work in OS grid metres, not degrees: one degree of longitude is only ~0.62
# of a degree of latitude in the UK, so Voronoi in degrees is distorted.
pts = np.column_stack([E, N])
vor = Voronoi(pts)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
voronoi_plot_2d(vor, ax=axs[0], show_points=False, show_vertices=False,
                line_width=0.2)
axs[0].set_xlim(E.min(), E.max()); axs[0].set_ylim(N.min(), N.max())
axs[0].set_aspect("equal"); axs[0].set_title("Voronoi cells of all MSOAs")

# Zoom on Bristol so the cells are visible
voronoi_plot_2d(vor, ax=axs[1], show_vertices=False, line_width=0.6, point_size=4)
axs[1].set_xlim(345e3, 375e3); axs[1].set_ylim(160e3, 190e3)
axs[1].set_aspect("equal"); axs[1].set_title("Zoom: Bristol")
fig.savefig(FIG / "q1d_voronoi.png", dpi=150, bbox_inches="tight")

# Adjacency: cells i, j are neighbours <=> they share a Voronoi edge
# <=> (i, j) is an edge of the Delaunay triangulation.
A_all = voronoi_adjacency(E, N)
A = voronoi_adjacency(E, N, max_dist=20_000)   # drop links > 20 km (sea crossings)
deg = np.asarray(A.sum(axis=1)).ravel()
print(f"Adjacency: {A.shape}, {A.nnz // 2} neighbour pairs "
      f"({(A_all.nnz - A.nnz) // 2} long links removed)")
print(f"Neighbours per cell: mean {deg.mean():.2f}, min {deg.min()}, max {deg.max()}")
# (For any planar triangulation the mean degree is just under 6.)

# Draw the neighbour graph around Bristol
fig, ax = plt.subplots(figsize=(6, 6))
Ac = A.tocoo()
for i, j in zip(Ac.row, Ac.col):
    if i < j:
        ax.plot([E[i], E[j]], [N[i], N[j]], "k-", lw=0.4)
ax.plot(E, N, "r.", ms=4)
ax.set_xlim(345e3, 375e3); ax.set_ylim(160e3, 190e3); ax.set_aspect("equal")
ax.set_title("Neighbour graph (Delaunay) around Bristol")
fig.savefig(FIG / "q1d_adjacency_bristol.png", dpi=150, bbox_inches="tight")

# %% Q1(e) Shapefiles (optional - needs a download)
# A shapefile stores the actual boundary polygon of each MSOA. Download
# "Middle layer Super Output Areas (December 2021) Boundaries EW BGC" from
# https://geoportal.statistics.gov.uk/ (Shapefile or GeoJSON) and unzip it into
# answers/data/msoa_boundaries/. geopandas reads it as a table with a
# 'geometry' column.
shp_dir = HERE / "data" / "msoa_boundaries"
shp_files = list(shp_dir.glob("*.shp")) + list(shp_dir.glob("*.geojson"))
if shp_files:
    import geopandas as gpd
    gdf = gpd.read_file(shp_files[0]).to_crs(epsg=27700)
    ax = gdf.plot(edgecolor="k", linewidth=0.1, figsize=(6, 7))
    ax.set_title("MSOA boundaries from shapefile")
    # True adjacency = polygons that touch. Compare with the Voronoi version.
    # nb = gpd.sjoin(gdf, gdf, predicate="touches")
    plt.savefig(FIG / "q1e_shapefile.png", dpi=150)
else:
    print("No shapefile found in data/msoa_boundaries - skipping Q1(e).")

plt.show()
