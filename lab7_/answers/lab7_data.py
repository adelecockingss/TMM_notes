# TMM Lab 06 - Practitioner studies: loading the Woodland Road sensor counts
# Small helper module used by q1_explore_data.py (and later questions).
#   load_counts()   -> one tidy long DataFrame with both periods
#   hourly_series() -> one hourly pandas Series (summed over the rows you pick)
# Data: hourly counts, 14 days each, three countlines, in/out directions.

from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent      # the CSVs live in lab7_/
FILES = {"summer": "2024-08-05_2024-08-18_counts.csv",   # summer vacation
         "term": "2024-10-14_2024-10-27_counts.csv"}     # term time
MODES = ["Car", "Pedestrian", "Cyclist"]
# short names for the three countlines (the raw names are long)
SHORT = {"S4_WoodlandRoad_path_LHS_uob001": "path_LHS",
         "S4_WoodlandRoad_path_RHS_uob001": "path_RHS",
         "S4_WoodlandRoad_road_uob001": "road"}


def load_counts():
    """Read both CSVs into one tidy (long) DataFrame.

    One row per (time, countline, direction, mode) with columns
    period, utc, time (local), countline, line (short name), direction,
    mode, count, hour, weekday (0 = Monday), is_weekend.
    """
    frames = []
    for period, fname in FILES.items():
        d = pd.read_csv(DATA / fname)
        d["period"] = period
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"UTC Datetime": "utc", "Local Datetime": "time",
                          "countlineName": "countline"})
    d["utc"] = pd.to_datetime(d["utc"])
    d["time"] = pd.to_datetime(d["time"])
    d["line"] = d["countline"].map(SHORT)
    # wide (one column per mode) -> long (a 'mode' and a 'count' column)
    d = d.melt(id_vars=["period", "utc", "time", "countline", "line", "direction"],
               value_vars=MODES, var_name="mode", value_name="count")
    d["hour"] = d["time"].dt.hour
    d["weekday"] = d["time"].dt.weekday
    d["is_weekend"] = d["weekday"] >= 5
    return d.sort_values(["period", "utc", "countline", "direction", "mode"],
                         ignore_index=True)


def hourly_series(df, period, mode, countline=None, direction=None):
    """Hourly counts for one period and mode as a pandas Series (index = local time).

    countline: raw name or short name ('path_LHS', 'path_RHS', 'road'); None = all.
    direction: 'in' or 'out'; None = both.
    The selected rows are summed for each hour.
    NB the clocks went back on 27 Oct 2024, so in the term data the local hour
    01:00 appears twice - here those two hours are summed into one (a choice!).
    """
    sel = (df["period"] == period) & (df["mode"] == mode)
    if countline is not None:
        sel &= (df["countline"] == countline) | (df["line"] == countline)
    if direction is not None:
        sel &= df["direction"] == direction
    s = df[sel].groupby("time")["count"].sum()
    return s.asfreq("h", fill_value=0)       # regular hourly index


if __name__ == "__main__":
    df = load_counts()
    print(df.head())
    print(hourly_series(df, "term", "Car", "road", "in").describe())
