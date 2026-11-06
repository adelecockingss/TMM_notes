# TMM Lab 06 - Q1: exploring the Woodland Road sensor data
# Cells: load data, summary table, time-series plots, weekday/weekend profiles,
# then SKELETONS for Q1b (decomposition) and Q1c (data-quality questions).
# The TODO cells are for working through together - not answered here.
# Run: python q1_explore_data.py   (figures go to answers/figures/)

# %% Imports and data
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lab7_data import load_counts, hourly_series, MODES

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
df = load_counts()
print(df.head())

# %% Summary table: total counts by period / countline / direction / mode
summary = df.pivot_table(index=["period", "line", "direction"], columns="mode",
                         values="count", aggfunc="sum")
print(summary)

# %% (a) Hourly time series per mode, one panel per period (all countlines summed)
fig, axs = plt.subplots(len(MODES), 2, figsize=(12, 8), sharey="row")
for c, period in enumerate(["summer", "term"]):
    for r, mode in enumerate(MODES):
        s = hourly_series(df, period, mode)
        axs[r, c].plot(s.index, s.values, lw=0.8)
        axs[r, c].set_title(f"{mode} - {period}", fontsize=9)
        axs[r, c].tick_params(axis="x", labelsize=7, rotation=30)
axs[0, 0].set_ylabel("count / hour")
fig.tight_layout(); fig.savefig(FIG / "q1_timeseries.png", dpi=130)

# %% (b) Average daily profile: weekday vs weekend, per mode and period
# (mean over days of the hourly total, all countlines and directions summed)
tot = df.groupby(["period", "mode", "time", "hour", "is_weekend"])["count"].sum().reset_index()
prof = tot.groupby(["period", "mode", "is_weekend", "hour"])["count"].mean()
fig, axs = plt.subplots(1, len(MODES), figsize=(13, 3.8))
for ax, mode in zip(axs, MODES):
    for period in ["summer", "term"]:
        for wk, ls in [(False, "-"), (True, "--")]:
            p = prof.loc[(period, mode, wk)]
            ax.plot(p.index, p.values, ls, label=f"{period} {'weekend' if wk else 'weekday'}")
    ax.set_title(mode); ax.set_xlabel("hour of day"); ax.set_xticks(range(0, 24, 3))
axs[0].set_ylabel("mean count / hour"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "q1_daily_profiles.png", dpi=130)


# %% Q1b SKELETON: time-series decomposition (trend + seasonal + residual)
# Same idea as lab5 answers/s2_time_series.py -> decompose():
#   trend    = centred moving mean over a window
#   seasonal = average of the detrended series at each position in the period
#   residual = what is left
def decompose(y, trend_window, period):
    half = trend_window // 2
    trend = np.array([y[max(0, i - half):i + half].mean() for i in range(len(y))])
    detr = y - trend
    k = int(np.ceil(len(y) / period))
    padded = np.r_[detr, np.full(k * period - len(y), np.nan)].reshape(k, period)
    season = np.tile(np.nanmean(padded, 0), k)[:len(y)]
    return trend, season, y - trend - season


# TODO (choices to discuss):
#  - trend_window: we only have 14 days per period. A weekly (168 h) window
#    leaves how many independent 'trend' values? Would 24 h be better here?
#  - period: 24 h or 168 h? With only 2 weeks, how many repeats of a weekly
#    season do we get - is that enough to estimate it?
#  - which series: all countlines summed, or road vs paths separately?
#    in + out summed, or each direction?
#  - decompose summer and term separately, or join them (gap in between!)?
TREND_WINDOW = None   # TODO
PERIOD = None         # TODO
if TREND_WINDOW is not None and PERIOD is not None:
    for period in ["summer", "term"]:
        fig, axs = plt.subplots(4, len(MODES), figsize=(13, 8), sharex="col")
        for c, mode in enumerate(MODES):
            y = hourly_series(df, period, mode).values.astype(float)
            parts = [y, *decompose(y, TREND_WINDOW, PERIOD)]
            for r, (s, t) in enumerate(zip(parts, ["original", "trend", "seasonal", "residual"])):
                axs[r, c].plot(s, lw=0.8); axs[r, c].set_title(f"{mode}: {t}", fontsize=8)
        fig.suptitle(period); fig.tight_layout()
        fig.savefig(FIG / f"q1b_decomposition_{period}.png", dpi=130)
# TODO (interpretation): what does the decomposition tell us that the
# daily profiles above don't? Is it useful for the crossing assessment
# (e.g. the Leeds framework uses peak-hour flows - does a trend/seasonal
# split help us pick a 'design hour')?

# %% Q1c SKELETON: data-quality questions to investigate (facts only, no answers)
road = df[df["line"] == "road"]
term = df[df["period"] == "term"]

print("\n--- Things to look into for Q1c ---")
night = term[(term["mode"] == "Pedestrian") & term["hour"].between(1, 4)]
print("1. Term time, 01:00-04:59, mean pedestrians/hour per countline:")
print(night.groupby("line")["count"].mean().round(1).to_string())
print("   -> Who is walking on the LHS path at 3am in term time but not in summer? "
      "Real people, or something the sensor is counting wrongly?")

cars = road[road["mode"] == "Car"].pivot_table(index="period", columns="direction",
                                               values="count", aggfunc="sum")
print("\n2. Cars on the road countline, total in vs out:")
print(cars.to_string())
print("   -> Woodland Road: is it one-way, or two-way? If cars go in, where do they"
      " come out? Does the sensor see both lanes equally well?")

print("\n3. Pedestrians and cyclists counted on the ROAD countline vs the paths:")
print(df[df["mode"] != "Car"].groupby(["period", "line", "mode"])["count"].sum().unstack().to_string())
print("   -> Why are RHS path counts nearly zero? Is the RHS path visible to the"
      " sensor, and are people walking in the road being counted as 'road'?")

dup = df[df["period"] == "term"].duplicated(["time", "countline", "direction", "mode"]).sum()
print(f"\n4. Term data has {dup} rows with a repeated local timestamp (27 Oct 2024)."
      " Note the term file also ends at 23:00 on 27 Oct, summer at 00:00 on 19 Aug."
      " -> Why? (Hint: compare UTC and local time.) How should we treat them?")
print("\n5. Figure 2 shows tracks for one hour only - where is the sensor relative"
      " to the proposed crossing, and which desire lines does it miss?")
# TODO: write up the answers together (sensor position, occlusion, counting
#       line placement, classification errors, representativeness of 2 weeks).

plt.show()
