# TMM Lab 01 - Q2: Basics of the census data
# Run cell-by-cell in VS Code (Shift+Enter) or: python q2_census_stats.py
# MATLAB version: q2_census_stats.m

# %% Imports and loading
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from geo_utils import latlon_to_osgb

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(HERE.parent / "UK_census_msoa_stats.csv")
df = df.rename(columns={
    "MSOA Code": "code", "MSOA Name": "name", "Longitude": "lon", "Latitude": "lat",
    "Population": "pop", "Area": "area_ha", "Households": "households",
    "Townsend deprivation score": "townsend", "O (trips)": "O", "D (trips)": "D",
})

# %% Q2: sensible variable types
# - counts -> unsigned integers (MATLAB: uint32)
# - the code is a unique ID -> index (MATLAB: RowNames)
# - country and local authority take a few repeated values -> categorical
count_cols = ["pop", "area_ha", "households", "O", "D"]
df[count_cols] = df[count_cols].astype("uint32")
df = df.set_index("code")
df["country"] = df.index.str[0].map({"E": "England", "W": "Wales"}).astype("category")
# Q2(d)(i): the name is "<local authority> <3 digits><letter>", e.g.
# "Barking and Dagenham 001A" -> strip the trailing number to get the authority
df["la"] = df["name"].str.replace(r"\s+\d{3}[A-Z]?$", "", regex=True).astype("category")
E, N = latlon_to_osgb(df["lat"].to_numpy(), df["lon"].to_numpy())
df["E"], df["N"] = E, N
print(df.dtypes)
print(f"{df['la'].nunique()} local authorities, {len(df)} MSOAs")


def fit_ols(y, X):
    """OLS with an intercept. Returns the statsmodels result (like MATLAB fitlm)."""
    return sm.OLS(y.astype(float), sm.add_constant(X.astype(float))).fit()


# %% Q2(a) O(i) against population
fit_a = fit_ols(df["O"], df["pop"])
print(fit_a.summary().tables[1])
print(f"R^2 = {fit_a.rsquared:.3f}, r = {np.sqrt(fit_a.rsquared):.3f}")

fig, ax = plt.subplots()
ax.scatter(df["pop"], df["O"], s=3, alpha=0.3)
x = np.linspace(0, df["pop"].max(), 2)
b0, b1 = fit_a.params
ax.plot(x, b0 + b1 * x, "r", label=f"O = {b0:.0f} + {b1:.3f} pop")
ax.plot(x, x, "k--", lw=0.8, label="O = pop (upper limit)")
ax.set_xlabel("Population"); ax.set_ylabel("O(i)"); ax.legend()
ax.set_title(f"Q2(a)  R$^2$ = {fit_a.rsquared:.2f}")
fig.savefig(FIG / "q2a_O_vs_pop.png", dpi=150, bbox_inches="tight")
# Interpretation: slope 0.76 (Anders's slide) = extra commuters per extra
# resident, i.e. roughly the fraction of residents who travel to work.
# O can never exceed the population, so this is a natural upper limit.

# %% Q2(b) correlation matrix of everything
vars_b = ["O", "D", "pop", "households", "area_ha", "townsend"]
corr = df[vars_b].corr()
print(corr.round(3))
# Spearman (rank) correlation is robust to the huge outliers in D and area
print(df[vars_b].corr(method="spearman").round(3))

# %% Q2(b) simple regressions: O on each variable separately
rows = []
for var in ["pop", "households", "area_ha", "townsend", "D"]:
    f = fit_ols(df["O"], df[var])
    rows.append({"predictor": var, "slope": f.params[var], "std_err": f.bse[var],
                 "t": f.tvalues[var], "p_value": f.pvalues[var], "R2": f.rsquared})
simple = pd.DataFrame(rows)
print(simple.to_string(index=False, float_format="%.4g"))

# %% Q2(b) multiple regression: O ~ pop + households + log(area) + townsend
# log(area) because area is hugely skewed (rural MSOAs are 100x bigger).
df["log_area"] = np.log(df["area_ha"].astype(float))
multi_vars = ["pop", "households", "log_area", "townsend"]
fit_b = fit_ols(df["O"], df[multi_vars])
print(fit_b.summary())

# Multicollinearity: pop and households are ~0.9 correlated, so their
# individual coefficients are unstable. VIF > ~5-10 is the usual warning.
Xc = sm.add_constant(df[multi_vars].astype(float))
vif = pd.Series([variance_inflation_factor(Xc.values, k) for k in range(1, Xc.shape[1])],
                index=multi_vars)
print("VIF:\n", vif.round(2))

# Standardised coefficients (all variables z-scored) let you compare the
# importance of predictors measured in different units.
z = (df[multi_vars + ["O"]].astype(float) - df[multi_vars + ["O"]].astype(float).mean()) \
    / df[multi_vars + ["O"]].astype(float).std()
print("Standardised coefficients:\n", fit_ols(z["O"], z[multi_vars]).params.round(3))

# %% Q2(c) D(i) is not well predicted by the other variables
fit_c = fit_ols(df["D"], df[["pop", "households", "log_area", "townsend"]])
print(f"D ~ pop + households + log_area + townsend:  R^2 = {fit_c.rsquared:.3f}")
for var in ["pop", "households", "area_ha", "townsend", "O"]:
    print(f"  D ~ {var:10s}  R^2 = {fit_ols(df['D'], df[var]).rsquared:.3f}")

# Even on a log scale (which tames the outliers) it is poor
pos = df["D"] > 0
fit_c_log = fit_ols(np.log(df.loc[pos, "D"].astype(float)),
                    np.log(df.loc[pos, ["pop", "households"]].astype(float)).join(
                        df.loc[pos, ["log_area", "townsend"]]))
print(f"log D on logs of the same: R^2 = {fit_c_log.rsquared:.3f}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].scatter(df["pop"], df["D"], s=3, alpha=0.3)
axs[0].set_yscale("log"); axs[0].set_xlabel("Population"); axs[0].set_ylabel("D(i)")
axs[0].set_title("D vs population (log y)")
axs[1].scatter(df["O"], df["D"], s=3, alpha=0.3)
axs[1].set_yscale("log"); axs[1].set_xlabel("O(i)"); axs[1].set_ylabel("D(i)")
axs[1].set_title("D vs O")
# label the biggest attractors - these are city centres / business districts
for code, r in df.nlargest(4, "D").iterrows():
    axs[1].annotate(r["name"], (r["O"], r["D"]), fontsize=7)
fig.savefig(FIG / "q2c_D_scatter.png", dpi=150, bbox_inches="tight")
# Why: D depends on where JOBS are (offices, hospitals, industrial estates),
# which none of these residential variables measure. Anders improves it with
# census employment-by-industry counts.

# %% Q2(d)(i) aggregate to local authority using the name strings
la = df.groupby("la", observed=True)[["pop", "O", "D"]].sum()
la["D_over_O"] = la["D"] / la["O"]
print(la.sort_values("D", ascending=False).head(10))
# "Barking and Dagenham" is a London borough = a Local Authority District
# (lower tier). For the London boroughs and unitary authorities the LAD is
# also the UTLA; for two-tier areas (e.g. "Cherwell" in Oxfordshire) the
# UTLA is the county, so the names give LADs, not UTLAs.

# %% Q2(d)(ii) clustering: (1) square grid cells
cell = 25_000                                     # 25 km squares
df["cell"] = (df["E"] // cell).astype(int).astype(str) + "_" + (df["N"] // cell).astype(int).astype(str)
grid = df.groupby("cell")[["pop", "O", "D", "E", "N"]].agg(
    {"pop": "sum", "O": "sum", "D": "sum", "E": "mean", "N": "mean"})
print(f"{len(grid)} non-empty {cell/1000:.0f} km cells")


# %% Q2(d)(ii) clustering: (2) Anders's random nearest-neighbour merging
def merge_zones(x, y, values, n_target, seed=0):
    """Pick a random zone, merge it with its nearest neighbour, repeat
    until n_target zones remain. Merged zone position = weighted centroid.
    values: (n, k) array of quantities to add up (pop, O, D...).
    Returns (x, y, values, labels) where labels[i] = merged zone of MSOA i."""
    rng = np.random.default_rng(seed)
    x, y = x.astype(float).copy(), y.astype(float).copy()
    vals = values.astype(float).copy()
    w = vals[:, 0].copy()                    # weight for the centroid (population)
    alive = np.ones(len(x), bool)
    label = np.arange(len(x))                # which zone each MSOA belongs to
    n = len(x)
    while n > n_target:
        idx = np.flatnonzero(alive)
        i = rng.choice(idx)
        d2 = (x[idx] - x[i]) ** 2 + (y[idx] - y[i]) ** 2
        d2[idx == i] = np.inf
        j = idx[np.argmin(d2)]
        # merge j into i
        x[i] = (w[i] * x[i] + w[j] * x[j]) / (w[i] + w[j])
        y[i] = (w[i] * y[i] + w[j] * y[j]) / (w[i] + w[j])
        w[i] += w[j]
        vals[i] += vals[j]
        alive[j] = False
        label[label == j] = i
        n -= 1
    return x[alive], y[alive], vals[alive], label


fig, axs = plt.subplots(2, 4, figsize=(15, 6))
vals = df[["pop", "O", "D"]].to_numpy()
for k, n_target in enumerate([7201, 2000, 500, 100]):
    if n_target < len(df):
        _, _, v, _ = merge_zones(df["E"].to_numpy(), df["N"].to_numpy(), vals, n_target)
    else:
        v = vals.astype(float)
    bins = np.linspace(0, 3, 61)              # same bins everywhere so panels compare
    axs[0, k].hist(v[:, 1] / v[:, 0], bins=bins, density=True, color="tab:blue")
    axs[0, k].set_title(f"N = {n_target}: O/pop")
    axs[1, k].hist(np.clip(v[:, 2] / v[:, 0], 0, 3), bins=bins, density=True,
                   color="tab:orange")
    axs[1, k].set_title(f"N = {n_target}: D/pop (clipped at 3)")
fig.tight_layout()
fig.savefig(FIG / "q2d_merged_histograms.png", dpi=150)
# Merging just adds up O, D and pop. Jobs are concentrated in a few zones
# (city centres, business parks), so at fine scale D/pop is very spread out.
# A big merged zone contains a mix of job centres and housing, so its D/pop
# averages out towards the national ratio sum(D)/sum(pop) = sum(O)/sum(pop)
# ~ 0.77 and the two histograms end up looking alike (Anders's N = 100 slide).
# Lesson: the conclusions depend on the zone size (the "modifiable areal
# unit problem", MAUP).

# %% Q2(d)(iii) official lookup table (2021 MSOA -> UTLA)
# The coursework .mat file includes, for each 2021 MSOA, the index of its
# Upper Tier Local Authority. (Exported to CSV in data/ - see NOTES.md.)
c21 = pd.read_csv(HERE / "data" / "msoa2021_centroids.csv")
utla = pd.read_csv(HERE / "data" / "utla2021.csv")
c21 = c21.merge(utla, on="UTLAindex", suffixes=("", "_utla"))
lookup = c21.set_index("code")["label_utla"]
df["utla"] = df.index.map(lookup)
print(f"{df['utla'].notna().sum()} of {len(df)} 2011 MSOAs found in the 2021 lookup "
      "(the rest were split/merged between censuses)")
utla_tot = df.groupby("utla")[["pop", "O", "D"]].sum().sort_values("D", ascending=False)
print(utla_tot.head(10))

# %% Q2(e) map plots
fig, axs = plt.subplots(1, 2, figsize=(12, 7))
r = (df["D"] / df["O"]).clip(lower=1e-2)
sc = axs[0].scatter(df["E"] / 1e3, df["N"] / 1e3, c=r, s=3, cmap="coolwarm",
                    norm=mcolors.LogNorm(vmin=0.1, vmax=10))
fig.colorbar(sc, ax=axs[0], label="D / O  (red = attracts, blue = generates)")
axs[0].set_title("Attractors vs generators")
sc = axs[1].scatter(grid["E"] / 1e3, grid["N"] / 1e3, s=grid["D"] / 2000,
                    c=grid["D"] / grid["O"], cmap="coolwarm",
                    norm=mcolors.LogNorm(vmin=0.3, vmax=3), alpha=0.7)
fig.colorbar(sc, ax=axs[1], label="D / O")
axs[1].set_title("25 km grid cells: size = D")
for ax in axs:
    ax.set_aspect("equal"); ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
fig.savefig(FIG / "q2e_maps.png", dpi=150, bbox_inches="tight")

plt.show()
