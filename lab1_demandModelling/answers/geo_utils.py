"""
TMM Lab 01 - shared geography helpers (used by every q*.py script).

    haversine_matrix(lat1, lon1, lat2, lon2)  -> great-circle distances (km)
    latlon_to_osgb(lat, lon)                  -> OS easting, northing (m)
    osgb_gridref(e, n)                        -> 12-digit grid reference string
    voronoi_adjacency(x, y)                   -> sparse neighbour matrix

Everything is vectorised with numpy broadcasting, so you can pass single
numbers, 1-D arrays or lists.

MATLAB equivalents: haversine_matrix.m, latlon_to_osgb.m (same folder).
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Q1(a) Haversine distance
# ---------------------------------------------------------------------------
def haversine_matrix(lat1, lon1, lat2=None, lon2=None):
    """Great-circle distance in km between every point in set 1 and set 2.

    Haversine formula, with phi = latitude and lambda = longitude in radians:
        a = sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2)
        d = 2 R asin(sqrt(a))

    Returns an (n1, n2) matrix. If set 2 is omitted, returns the symmetric
    (n1, n1) matrix of distances between the points of set 1.
    """
    if lat2 is None:
        lat2, lon2 = lat1, lon1
    # np.atleast_1d lets scalars work too; [:, None] makes column vectors so
    # that broadcasting a column against a row produces a full matrix.
    p1 = np.radians(np.atleast_1d(lat1).astype(float))[:, None]
    l1 = np.radians(np.atleast_1d(lon1).astype(float))[:, None]
    p2 = np.radians(np.atleast_1d(lat2).astype(float))[None, :]
    l2 = np.radians(np.atleast_1d(lon2).astype(float))[None, :]

    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Q1(b) lat/long (WGS84, i.e. GPS) -> Ordnance Survey National Grid
# ---------------------------------------------------------------------------
# Two steps (formulas from the OS guide "A guide to coordinate systems in
# Great Britain"):
#   1. Datum shift: GPS coordinates use the WGS84 ellipsoid, but the OS grid
#      is defined on the older OSGB36 datum (Airy 1830 ellipsoid). We convert
#      to 3-D Cartesian x,y,z, apply a 7-parameter Helmert transformation, and
#      convert back to lat/long on Airy. (Accurate to ~5 m: plenty here.)
#   2. Transverse Mercator projection of the OSGB36 lat/long onto the flat grid.

# Ellipsoids: semi-major axis a, semi-minor axis b
_WGS84 = (6378137.000, 6356752.3141)
_AIRY1830 = (6377563.396, 6356256.909)

# Helmert parameters WGS84 -> OSGB36: translations (m), rotations (arc-seconds),
# scale (ppm)
_TX, _TY, _TZ = -446.448, 125.157, -542.060
_RX, _RY, _RZ = -0.1502, -0.2470, -0.8421
_S_PPM = 20.4894

# National Grid projection constants
_F0 = 0.9996012717                              # scale factor on central meridian
_LAT0, _LON0 = np.radians(49.0), np.radians(-2.0)  # true origin
_E0, _N0 = 400000.0, -100000.0                  # false origin offsets (m)


def _latlon_to_xyz(lat, lon, a, b, h=0.0):
    e2 = 1 - b**2 / a**2
    nu = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (nu + h) * np.cos(lat) * np.cos(lon)
    y = (nu + h) * np.cos(lat) * np.sin(lon)
    z = ((1 - e2) * nu + h) * np.sin(lat)
    return x, y, z


def _xyz_to_latlon(x, y, z, a, b):
    e2 = 1 - b**2 / a**2
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):                        # converges in a few iterations
        nu = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * nu * np.sin(lat), p)
    lon = np.arctan2(y, x)
    return lat, lon


def _helmert(x, y, z):
    s = _S_PPM * 1e-6
    rx, ry, rz = np.radians(np.array([_RX, _RY, _RZ]) / 3600)
    x2 = _TX + (1 + s) * x - rz * y + ry * z
    y2 = _TY + rz * x + (1 + s) * y - rx * z
    z2 = _TZ - ry * x + rx * y + (1 + s) * z
    return x2, y2, z2


def latlon_to_osgb(lat, lon):
    """WGS84 latitude/longitude (degrees) -> OS easting, northing (metres)."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))

    # Step 1: datum shift WGS84 -> OSGB36
    x, y, z = _latlon_to_xyz(lat, lon, *_WGS84)
    x, y, z = _helmert(x, y, z)
    lat, lon = _xyz_to_latlon(x, y, z, *_AIRY1830)

    # Step 2: Transverse Mercator projection on the Airy ellipsoid
    a, b = _AIRY1830
    e2 = 1 - b**2 / a**2
    n = (a - b) / (a + b)
    sl, cl, tl = np.sin(lat), np.cos(lat), np.tan(lat)
    nu = a * _F0 / np.sqrt(1 - e2 * sl**2)
    rho = a * _F0 * (1 - e2) / (1 - e2 * sl**2) ** 1.5
    eta2 = nu / rho - 1

    # Meridional arc M (distance from the true origin latitude along the meridian)
    dlat, slat = lat - _LAT0, lat + _LAT0
    M = b * _F0 * (
        (1 + n + 1.25 * n**2 + 1.25 * n**3) * dlat
        - (3 * n + 3 * n**2 + 21 / 8 * n**3) * np.sin(dlat) * np.cos(slat)
        + (15 / 8 * n**2 + 15 / 8 * n**3) * np.sin(2 * dlat) * np.cos(2 * slat)
        - 35 / 24 * n**3 * np.sin(3 * dlat) * np.cos(3 * slat)
    )
    I = M + _N0
    II = nu / 2 * sl * cl
    III = nu / 24 * sl * cl**3 * (5 - tl**2 + 9 * eta2)
    IIIA = nu / 720 * sl * cl**5 * (61 - 58 * tl**2 + tl**4)
    IV = nu * cl
    V = nu / 6 * cl**3 * (nu / rho - tl**2)
    VI = nu / 120 * cl**5 * (5 - 18 * tl**2 + tl**4 + 14 * eta2 - 58 * tl**2 * eta2)

    dl = lon - _LON0
    northing = I + II * dl**2 + III * dl**4 + IIIA * dl**6
    easting = _E0 + IV * dl + V * dl**3 + VI * dl**5
    return easting, northing


def osgb_gridref(easting, northing):
    """12-digit grid reference: 6 digits of easting then 6 of northing (1 m)."""
    e = np.atleast_1d(np.round(easting).astype(int))
    n = np.atleast_1d(np.round(northing).astype(int))
    return [f"{ei:06d} {ni:06d}" for ei, ni in zip(e, n)]


# ---------------------------------------------------------------------------
# Q1(d) Voronoi neighbours
# ---------------------------------------------------------------------------
def voronoi_adjacency(x, y, max_dist=None):
    """Sparse 0/1 adjacency matrix of Voronoi cells that share an edge.

    Two Voronoi cells share an edge exactly when their centres are joined in
    the Delaunay triangulation (the two are 'dual' to each other), so we read
    the neighbours off the triangles.

    max_dist (same units as x, y): drop links longer than this. Without it,
    cells either side of a stretch of sea (e.g. Cardiff-Somerset across the
    Bristol Channel) are counted as neighbours.
    """
    pts = np.column_stack([x, y])
    tri = Delaunay(pts)
    s = tri.simplices                        # (ntri, 3) node indices
    i = np.concatenate([s[:, 0], s[:, 1], s[:, 2]])
    j = np.concatenate([s[:, 1], s[:, 2], s[:, 0]])
    if max_dist is not None:
        keep = np.hypot(x[i] - x[j], y[i] - y[j]) <= max_dist
        i, j = i[keep], j[keep]
    n = len(x)
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)      # symmetric, 0/1
    return A
