# TMM Lab 01 - Q3: Trip data - building the OD matrix
# Run cell-by-cell or: python q3_od_matrix.py   (~30 s, needs ~2 GB RAM)
# MATLAB version: q3_od_matrix.m
# Saves data/od_msoa.npz and data/od_utla.npz for Q4-Q8.

# %% Imports
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = HERE / "data"

# %% Load the 2021 census origin-destination file (1.86 million rows)
# Only read the columns we need, with compact types, to save memory.
cols = {"Middle layer Super Output Areas code": "o",
        "MSOA of workplace code": "d",
        "Place of work indicator (4 categories) code": "pow",
        "Count": "count"}
od = pd.read_csv(HERE.parent / "ODWP01EW_MSOA.csv", usecols=list(cols),
                 dtype={"Middle layer Super Output Areas code": "category",
                        "MSOA of workplace code": "category",
                        "Place of work indicator (4 categories) code": "int8",
                        "Count": "int32"}).rename(columns=cols)
print(od.groupby("pow", observed=True)["count"].agg(["size", "sum"]))
# Place-of-work codes:
#  -8 "does not apply" (not in work)       1 mainly work at/from home
#   2 offshore / outside the UK            3 work in the UK, not at home  <- trips
# Work-from-home people make no commute, so we keep only code 3.

# %% Q3(b) lookup table: row/column index <-> MSOA code
# Use the 2021 MSOA list (with centroids and UTLA) exported from the
# coursework .mat file. Index k (0-based) in Python = k+1 in MATLAB.
msoa = pd.read_csv(DATA / "msoa2021_centroids.csv")
n = len(msoa)
code_to_idx = pd.Series(np.arange(n), index=msoa["code"])
print(f"{n} MSOAs in England & Wales")

# %% Q3(a) sparse OD matrix
trips = od[od["pow"] == 3]
i = code_to_idx.reindex(trips["o"].astype(str)).to_numpy()
j = code_to_idx.reindex(trips["d"].astype(str)).to_numpy()
ok = ~np.isnan(i) & ~np.isnan(j)        # destination in Scotland/NI -> dropped
print(f"Dropped {trips['count'][~ok].sum():,} trips to workplaces outside E&W")
T = sp.coo_matrix((trips["count"].to_numpy()[ok], (i[ok].astype(int), j[ok].astype(int))),
                  shape=(n, n)).tocsr()   # duplicates (if any) are summed
print(f"T: {n} x {n}, {T.nnz:,} non-zero entries "
      f"({100 * T.nnz / n**2:.2f}% full), {T.sum():,} trips")
# Dense float64 would need n^2 * 8 bytes = 422 MB; sparse needs ~20 MB.

# %% Q3(c) row and column totals
O = np.asarray(T.sum(axis=1)).ravel()     # trips starting in each MSOA
D = np.asarray(T.sum(axis=0)).ravel()     # trips ending in each MSOA
intra = T.diagonal()
print(f"sum O = sum D = {O.sum():,};  intra-zonal trips: {intra.sum():,} "
      f"({100 * intra.sum() / O.sum():.1f}%)")
wfh = od[od["pow"] == 1].groupby("o", observed=True)["count"].sum()
print(f"(plus {wfh.sum():,} people working mainly at home - Covid-era 2021 census!)")

# %% Q3(d) distance matrix between centroids
E = msoa["easting"].to_numpy()
N = msoa["northing"].to_numpy()


def distance_matrix(E, N, dtype=np.float32):
    """Symmetric matrix of straight-line distances (km). float32 halves memory."""
    E = E.astype(dtype) / 1000
    N = N.astype(dtype) / 1000
    return np.hypot(E[:, None] - E[None, :], N[:, None] - N[None, :])


dist = distance_matrix(E, N)              # 7264^2 float32 = 211 MB
print(f"distance matrix {dist.shape}, {dist.nbytes / 1e6:.0f} MB, max {dist.max():.0f} km")

# For the trip-length distribution we only need distances for non-zero T_ij
Tc = T.tocoo()
d_trip = dist[Tc.row, Tc.col]
w_trip = Tc.data
del dist                                   # free the memory

# %% Q3(e) distribution of commuting distance (weighted by number of people)
mean_d = np.average(d_trip, weights=w_trip)
med_d = np.median(np.repeat(d_trip, w_trip))
print(f"Mean commute (centroid-to-centroid, straight line): {mean_d:.1f} km, "
      f"median {med_d:.1f} km")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
# Linear bins, first 60 km
axs[0].hist(d_trip, bins=np.arange(0, 61, 1), weights=w_trip, color="tab:blue")
axs[0].set_xlabel("Distance (km)"); axs[0].set_ylabel("People")
axs[0].set_title("Commute distance (bar at 0 = same MSOA)")
# Log-log density: heavy tail ~ power law d^-alpha (Anders: alpha ~ 1.75)
pos = d_trip > 0
bins = np.logspace(0, np.log10(d_trip.max()), 40)
h, b = np.histogram(d_trip[pos], bins=bins, weights=w_trip[pos], density=True)
mid = np.sqrt(b[1:] * b[:-1])
axs[1].loglog(mid, h, "o-", ms=3)
sel = (mid > 10) & (mid < 200) & (h > 0)
alpha, logc = np.polyfit(np.log(mid[sel]), np.log(h[sel]), 1)
axs[1].loglog(mid[sel], np.exp(logc) * mid[sel] ** alpha, "r--",
              label=f"slope {alpha:.2f} (10-200 km)")
axs[1].set_xlabel("Distance (km)"); axs[1].set_ylabel("Probability density")
axs[1].legend(); axs[1].set_title("Log-log: heavy tail")
fig.savefig(FIG / "q3e_commute_distance.png", dpi=150, bbox_inches="tight")

# Compare with the National Travel Survey: NTS reports commuting trips in
# MILE bands of road distance. Straight-line distances are shorter than road
# distances (detour factor ~1.2-1.4), and every intra-zonal trip lands in the
# shortest band - so expect the bands to differ, especially at the short end.
miles = d_trip / 1.609
bands = [0, 1, 2, 5, 10, 25, 50, np.inf]
labels = ["<1", "1-2", "2-5", "5-10", "10-25", "25-50", "50+"]
share = pd.Series(np.histogram(miles, bins=bands, weights=w_trip)[0], index=labels)
print("Share of commuters by straight-line distance band (miles):")
print((100 * share / share.sum()).round(1).to_string())

# %% Save for later questions
sp.save_npz(DATA / "od_msoa_T.npz", T)
np.savez(DATA / "od_msoa.npz", O=O, D=D, E=E, N=N, codes=msoa["code"].to_numpy(),
         labels=msoa["label"].to_numpy(), utla=msoa["UTLAindex"].to_numpy())

# %% Q3(f) coarser geography: Upper Tier Local Authorities
# Aggregation matrix P (n_utla x n_msoa): P[u, i] = 1 if MSOA i is in UTLA u.
# Then T_utla = P T P^T  sums all flows between every pair of authorities.
utla = pd.read_csv(DATA / "utla2021.csv")
u = msoa["UTLAindex"].to_numpy() - 1      # 0-based
m = len(utla)
P = sp.csr_matrix((np.ones(n), (u, np.arange(n))), shape=(m, n))
TU = (P @ T @ P.T).toarray()
OU, DU = TU.sum(1), TU.sum(0)
# Centroid of a UTLA = origin-weighted mean of its MSOA centroids
EU = (P @ (E * O)) / (P @ O)
NU = (P @ (N * O)) / (P @ O)
distU = distance_matrix(EU, NU, np.float64)
print(f"UTLA level: {m} zones, {100 * np.trace(TU) / TU.sum():.0f}% of trips "
      "are inside one authority (distance 0!)")
np.savez(DATA / "od_utla.npz", T=TU, O=OU, D=DU, E=EU, N=NU, dist=distU,
         codes=utla["code"].to_numpy(), labels=utla["label"].to_numpy())

fig, ax = plt.subplots(figsize=(6, 4))
off = ~np.eye(m, dtype=bool)
ax.hist(distU[off], bins=np.arange(0, 400, 10), weights=TU[off])
ax.set_xlabel("Distance between UTLA centroids (km)"); ax.set_ylabel("People")
ax.set_title("UTLA level (inter-authority trips only)")
fig.savefig(FIG / "q3f_utla_distance.png", dpi=150, bbox_inches="tight")

# %% Q3(g) DataShine-style flow maps
def flow_map(ax, k, T, E, N, labels, outgoing=True, top=200, color="tab:red"):
    """Lines from zone k to the zones it sends to (or receives from).
    Line width proportional to flow, like commute.datashine.org.uk."""
    row = T[k] if outgoing else T[:, k].T
    row = sp.csr_matrix(row)
    idx, val = row.indices, row.data
    order = np.argsort(val)[::-1][:top]
    for jj, v in zip(idx[order], val[order]):
        ax.plot([E[k] / 1e3, E[jj] / 1e3], [N[k] / 1e3, N[jj] / 1e3], color=color,
                lw=0.3 + 4 * v / val.max(), alpha=0.6, solid_capstyle="round")
    ax.plot(E[k] / 1e3, N[k] / 1e3, "ko", ms=5)
    ax.set_title(f"{'From' if outgoing else 'To'} {labels[k]}")


labels = msoa["label"].to_numpy()
k_city = int(np.argmax(D * (msoa["label"].str.startswith("Bristol"))))  # biggest Bristol attractor
k_home = int(np.argmax(O * (msoa["label"].str.startswith("Bristol"))))
fig, axs = plt.subplots(1, 2, figsize=(13, 6))
for ax in axs:
    ax.scatter(E / 1e3, N / 1e3, s=0.3, c="0.8")
    ax.set_aspect("equal"); ax.set_xlim(330, 400); ax.set_ylim(140, 210)
flow_map(axs[0], k_city, T, E, N, labels, outgoing=False, color="tab:blue")
flow_map(axs[1], k_home, T, E, N, labels, outgoing=True, color="tab:red")
fig.savefig(FIG / "q3g_flow_maps_bristol.png", dpi=150, bbox_inches="tight")

plt.show()
