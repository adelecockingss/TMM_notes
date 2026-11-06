# TMM Lab 05 - Section 2: time series modelling of hourly pedestrian counts
# Worked example (decomposition, ARIMA(0,0,0), ARIMA(1,0,0), SARIMA, ML model)
# + exercises 2.0-2.7.   MATLAB version: s2_time_series.m
# The sheet uses an LSTM (MATLAB Deep Learning Toolbox). PyTorch isn't
# installed here, so the ML model is a neural network (scikit-learn MLP) on
# the last 24 hours - the same idea (learn y_t from the recent past).
# Run: python s2_time_series.py  (~1 min)

# %% Imports and data
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.neural_network import MLPRegressor

from melbourne_data import load

warnings.filterwarnings("ignore")
FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
D = load()
TAG = " (SYNTHETIC)" if D["synthetic"] else ""
ts = D["central"][3].astype(float)          # sheet: central(4,:) -> Python row 3
n = len(ts)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# %% Decomposition: trend (weekly moving mean) + seasonal (daily average) + residual
def decompose(y, trend_window=24 * 7, period=24):
    # centred moving mean, like MATLAB movmean (shrinks the window at the ends)
    half = trend_window // 2
    trend = np.array([y[max(0, i - half):i + half].mean() for i in range(len(y))])
    detr = y - trend
    k = int(np.ceil(len(y) / period))
    padded = np.r_[detr, np.full(k * period - len(y), np.nan)].reshape(k, period)
    season = np.tile(np.nanmean(padded, 0), k)[:len(y)]
    return trend, season, y - trend - season


trend, season, resid = decompose(ts)
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for ax, s, t in zip(axs, [ts, trend, season, resid], ["original", "trend (weekly mean)",
                                                       "seasonal (daily)", "residual"]):
    ax.plot(s, lw=1); ax.set_title(t + TAG, fontsize=9)
axs[-1].set_xlabel("time (hours)")
fig.tight_layout(); fig.savefig(FIG / "s2_decomposition.png", dpi=130)

# %% Exercise 2.0 / 2.1: window sizes and seasonality period
# A daily period misses the weekday/weekend difference, which then shows up in
# the residual. A WEEKLY period (168 h) captures both daily and weekly cycles.
for period in [24, 168]:
    _, s_, r_ = decompose(ts, period=period)
    print(f"Ex 2.0/2.1: seasonal period {period:3d} h: residual std {r_.std():.1f} "
          f"(explains {100 * (1 - r_.var() / (ts - trend).var()):.0f}% of detrended variance)")

# %% Train / test split (80 / 20, as in the sheet)
tr = round(0.8 * n)
train, test = ts[:tr], ts[tr:]
H = len(test)

# %% ARIMA models - closed loop (forecast all H hours from the end of training)
models = {"ARIMA(0,0,0)": dict(order=(0, 0, 0)),
          "ARIMA(1,0,0)": dict(order=(1, 0, 0)),
          "SARIMA (0,1,0)x(0,1,0)24": dict(order=(0, 0, 0), seasonal_order=(0, 1, 0, 24)),
          "ARIMA(2,0,0)": dict(order=(2, 0, 0)),
          "SARIMA (1,0,0)x(0,1,1)24": dict(order=(1, 0, 0), seasonal_order=(0, 1, 1, 24))}
# NB the sheet's MATLAB arima('D',1,'Seasonality',24) = (1 - B)(1 - B^24) y_t = c + e_t
fits, closed = {}, {}
for name, kw in models.items():
    if name.startswith("SARIMA (0,1"):
        kw = dict(order=(0, 1, 0), seasonal_order=(0, 1, 0, 24))
    fits[name] = ARIMA(train, **kw).fit()
    closed[name] = fits[name].forecast(H)
    print(f"closed loop  {name:26s} RMSE {rmse(closed[name], test):7.1f}")
print("  params ARIMA(1,0,0):", np.round(fits["ARIMA(1,0,0)"].params, 3))

# %% Open loop: one step ahead, using the true previous values
# (statsmodels: append the test data to the fitted model without refitting)
open_ = {}
for name, f in fits.items():
    open_[name] = f.append(test, refit=False).predict(start=tr, end=n - 1)
    print(f"open loop    {name:26s} RMSE {rmse(open_[name], test):7.1f}")
# Closed loop forecasts decay to the mean (AR) or repeat the last day
# (seasonal differencing); open loop is much easier - one step ahead with
# the latest data. AR(1) open loop is just 'last hour, shrunk to the mean'
# and visibly lags the data by one hour.


# %% ML model: neural network on the previous 24 hours (stands in for the LSTM)
def lagged(y, L=24):
    X = np.column_stack([y[i:len(y) - L + i] for i in range(L)])
    return X, y[L:]


L = 24
mu, sd = train.mean(), train.std()
z = (ts - mu) / sd
Xtr, ytr = lagged(z[:tr], L)
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=2000, random_state=0,
                   early_stopping=True).fit(Xtr, ytr)
Xall, _ = lagged(z, L)
ml_open = mlp.predict(Xall[tr - L:]) * sd + mu                # one step ahead on test
print(f"open loop    {'MLP on 24 lags':26s} RMSE {rmse(ml_open, test):7.1f}")

# %% Exercise 2.3: closed-loop forecasting for the ML model
# Feed each prediction back in as if it were data.
hist = list(z[tr - L:tr])
ml_closed = []
for _ in range(H):
    yhat = mlp.predict(np.array(hist[-L:])[None, :])[0]
    ml_closed.append(yhat); hist.append(yhat)
ml_closed = np.array(ml_closed) * sd + mu
print(f"closed loop  {'MLP on 24 lags':26s} RMSE {rmse(ml_closed, test):7.1f}")

fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
x = np.arange(tr, n)
axs[0].plot(x, test, "k", lw=1.5, label="actual")
for name in ["ARIMA(1,0,0)", "SARIMA (0,1,0)x(0,1,0)24"]:
    axs[0].plot(x, closed[name], lw=1, label=name)
axs[0].plot(x, ml_closed, lw=1, label="MLP (closed loop)")
axs[0].set_title("closed loop forecasts" + TAG); axs[0].legend(fontsize=7)
axs[1].plot(x, test, "k", lw=1.5, label="actual")
axs[1].plot(x, open_["SARIMA (0,1,0)x(0,1,0)24"], lw=1, label="SARIMA open loop")
axs[1].plot(x, ml_open, lw=1, label="MLP open loop")
axs[1].set_title("open loop (one step ahead)"); axs[1].legend(fontsize=7)
axs[1].set_xlabel("time (hours)")
fig.savefig(FIG / "s2_forecasts.png", dpi=130, bbox_inches="tight")

# %% Exercise 2.4: extra (exogenous) predictors - simulated
# Weekend/holiday flag (real information!) and simulated 'rain' (noise).
hour = np.arange(n)
weekend = (((5 + hour // 24) % 7) >= 5).astype(float)     # 1 June 2019 = Saturday
rain = np.random.default_rng(0).uniform(0, 1, n)
for label, ex in [("weekend flag", weekend[:, None]), ("random 'rain'", rain[:, None])]:
    f = ARIMA(train, exog=ex[:tr], order=(1, 0, 0), seasonal_order=(0, 1, 1, 24)).fit()
    fc = f.forecast(H, exog=ex[tr:])
    print(f"Ex 2.4: SARIMAX + {label:14s} closed-loop RMSE {rmse(fc, test):7.1f}")
# For the MLP: add the feature as an extra input column
Xw = np.column_stack([Xall, weekend[L:]])
mlp_w = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=2000, random_state=0,
                     early_stopping=True).fit(Xw[:tr - L], ytr)
print(f"Ex 2.4: MLP + weekend flag open-loop RMSE "
      f"{rmse(mlp_w.predict(Xw[tr - L:]) * sd + mu, test):7.1f}")

# %% Exercise 2.5: more robust comparison - rolling-origin evaluation
# Several forecast origins, fixed horizon (here 24 h), refit on all data
# before each origin; report the mean and spread of the errors.
origins = range(24 * 28, n - 24, 24 * 3)
scores = {"seasonal naive (same hour yesterday)": [], "SARIMA (1,0,0)x(0,1,1)24": []}
for o in origins:
    scores["seasonal naive (same hour yesterday)"].append(rmse(ts[o - 24:o], ts[o:o + 24]))
    f = ARIMA(ts[:o], order=(1, 0, 0), seasonal_order=(0, 1, 1, 24)).fit()
    scores["SARIMA (1,0,0)x(0,1,1)24"].append(rmse(f.forecast(24), ts[o:o + 24]))
for k, v in scores.items():
    print(f"Ex 2.5: rolling origin, 24 h ahead, {k:38s} RMSE {np.mean(v):6.1f} +- {np.std(v):5.1f}")

# %% Exercise 2.6: compare locations - fit the same model per sensor
print("Ex 2.6: SARIMA (1,0,0)x(0,1,1)24 fitted per sensor  [AR(1), seasonal MA(1), sigma]")
for k in range(0, D["central"].shape[0], 3):
    y = D["central"][k].astype(float)
    f = ARIMA(y[:tr], order=(1, 0, 0), seasonal_order=(0, 1, 1, 24)).fit()
    print(f"   sensor {k + 1:2d}: mean count {y.mean():7.1f}, params "
          f"{np.round(f.params[[0, 1]], 2)}, sigma {np.sqrt(f.params[-1]):6.1f}")
# Similar AR/MA coefficients but very different scales: one could model all
# sites as 'common shape x site-specific level' (or use log counts).

# %% Exercise 2.7: an alternative model - Holt-Winters exponential smoothing
hw = ExponentialSmoothing(train, trend=None, seasonal="add", seasonal_periods=168).fit()
print(f"Ex 2.7: Holt-Winters (weekly season) closed-loop RMSE {rmse(hw.forecast(H), test):7.1f}")

plt.show()
