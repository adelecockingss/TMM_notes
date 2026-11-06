"""
TMM Lab 05 - data for sections 1 and 2 (Melbourne pedestrian counts).

The lab needs four files from Blackboard that are NOT in the lab folder:
    pedestrian_counts.txt, lines_ys.txt, sensors_xs.txt, sensors_ys.txt
(only lines_xs.txt is there). load() uses the real files if you put them in
lab5_DataDrivenModelling/ (with or without the ' (1)' suffix); otherwise it
generates a SYNTHETIC stand-in with the same format and saves it to
answers/data/synthetic/ (MATLAB reads those too):
  - a 9 x 5 CBD-style street grid (76 street segments),
  - 13 sensors, two pairs on opposite sides of the same street (-> 11 sites),
  - hourly counts for 52 days (1248 h) starting on a Saturday (1 June 2019),
    with weekday morning / lunch / evening peaks, quieter weekends, a busy
    centre, and Poisson noise.
Everything in the scripts works the same on real or synthetic data, but
quote numbers from the real data in any write-up.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
SYN = HERE / "data" / "synthetic"


def _find(name):
    for cand in [LAB / name, LAB / name.replace(".txt", " (1).txt"), HERE / "data" / name]:
        if cand.exists():
            return cand
    return None


def hourly_profile(n_hours, start_weekday=5):
    """Expected relative count for each hour. start_weekday: 0 = Monday."""
    h = np.arange(n_hours)
    hour = h % 24
    dow = (start_weekday + h // 24) % 7
    weekday = (np.exp(-0.5 * ((hour - 8.5) / 0.8) ** 2) * 1.0 +
               np.exp(-0.5 * ((hour - 12.5) / 1.0) ** 2) * 0.8 +
               np.exp(-0.5 * ((hour - 17.3) / 1.0) ** 2) * 1.1 +
               0.25 * np.exp(-0.5 * ((hour - 14) / 4.5) ** 2) + 0.02)
    weekend = 0.7 * np.exp(-0.5 * ((hour - 14) / 3.0) ** 2) + 0.03
    return np.where(dow >= 5, weekend, weekday)


def attractiveness(x, y):
    """Smooth 'busyness' over the synthetic city (hot spot near the centre).
    x, y in the plotting frame (after ys = 655 - ys)."""
    return 150 + 2500 * np.exp(-((x - 300) ** 2 + (y - 330) ** 2) / (2 * 150 ** 2))


def make_synthetic(seed=0):
    rng = np.random.default_rng(seed)
    gx, gy = np.arange(9) * 80.0, 655 - (170 + np.arange(5) * 80.0)   # raw y, as in the files
    xs, ys = [], []
    for j in range(5):                                # east-west streets
        for i in range(8):
            xs += [gx[i], gx[i + 1]]; ys += [gy[j], gy[j]]
    for i in range(9):                                # north-south streets
        for j in range(4):
            xs += [gx[i], gx[i]]; ys += [gy[j], gy[j + 1]]
    xs, ys = np.array(xs), np.array(ys)
    # sensors: next to the midpoints of 11 segments, 5 units off the street;
    # segments 10 and 50 get one sensor on EACH side (13 sensors in total)
    segs = [2, 10, 10, 13, 20, 27, 33, 44, 50, 50, 57, 63, 70]
    sx, sy = [], []
    for k, s in enumerate(segs):
        mx, my = (xs[2 * s] + xs[2 * s + 1]) / 2, (ys[2 * s] + ys[2 * s + 1]) / 2
        horiz = ys[2 * s] == ys[2 * s + 1]
        side = -1 if (k > 0 and segs[k - 1] == s) else 1
        sx.append(mx + (0 if horiz else 5 * side)); sy.append(my + (5 * side if horiz else 0))
    # The sheet flips the street y's (ys = 655 - ys) but NOT the sensor y's,
    # so sensors are stored in the flipped (plotting) frame already.
    sx, sy = np.array(sx), 655 - np.array(sy)
    prof = hourly_profile(1248)
    level = attractiveness(sx, sy) * rng.uniform(0.7, 1.3, len(sx))
    counts = rng.poisson(level[:, None] * prof[None, :] * rng.lognormal(0, 0.15, (len(sx), 1248)))
    SYN.mkdir(parents=True, exist_ok=True)
    np.savetxt(SYN / "lines_xs.txt", xs, fmt="%.6f"); np.savetxt(SYN / "lines_ys.txt", ys, fmt="%.6f")
    np.savetxt(SYN / "sensors_xs.txt", sx, fmt="%.6f"); np.savetxt(SYN / "sensors_ys.txt", sy, fmt="%.6f")
    np.savetxt(SYN / "pedestrian_counts.txt", counts, fmt="%d")      # no name column
    return counts.astype(float), xs, ys, sx, sy


def load():
    """Returns dict(central, xs, ys, sensx, sensy, synthetic). ys is already
    flipped (ys = 655 - ys) as in the lab sheet; sensy is not (as in the sheet)."""
    files = {k: _find(k + ".txt") for k in
             ["pedestrian_counts", "lines_xs", "lines_ys", "sensors_xs", "sensors_ys"]}
    if all(files.values()):
        import pandas as pd
        central = pd.read_csv(files["pedestrian_counts"], header=None, sep=None,
                              engine="python").iloc[:, 1:1249].to_numpy(float)
        xs = np.loadtxt(files["lines_xs"]); ys = 655 - np.loadtxt(files["lines_ys"])
        sx = np.loadtxt(files["sensors_xs"]); sy = np.loadtxt(files["sensors_ys"])
        synthetic = False
    else:
        central, xs, ys, sx, sy = make_synthetic()
        ys = 655 - ys
        synthetic = True
        print("NOTE: Melbourne files not found -> using SYNTHETIC stand-in data "
              "(see melbourne_data.py)")
    return dict(central=central, xs=xs, ys=ys, sensx=sx, sensy=sy, synthetic=synthetic)
