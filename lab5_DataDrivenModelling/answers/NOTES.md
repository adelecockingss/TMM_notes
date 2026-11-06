# Lab 5 – Data-Driven Modelling: notes and answers

Lecture source: Nikolai Bode's data-driven modelling lectures 1–4. Lab sheet: `TMM_labsheet_data_driven.pdf` (live script `..._25_26.mlx`). Also used: the lecturer's `notes_on_exercise_answers_25_26.pdf`.

## 0. How to run, and a data warning

| Section | Python | MATLAB (base only) |
|---|---|---|
| 1 direct demand (IDW) | `s1_direct_demand.py` | `s1_direct_demand.m` |
| 2 time series | `s2_time_series.py` | `s2_time_series.m` |
| 3 ML microscopic model | `s3_ml_microscopic.py` | `s3_ml_microscopic.m` |

Run from this `answers/` folder. The MATLAB scripts add `..` to the path for the lecturer's `IDWprediction.m` and `dataparcel.m`.

> ⚠️ **Sections 1–2 need four Blackboard files that are missing from the lab folder:** `pedestrian_counts.txt`, `lines_ys.txt`, `sensors_xs.txt`, `sensors_ys.txt`. Only `lines_xs.txt` is there.
>
> Until you add them, `melbourne_data.py` generates a **synthetic stand-in** in the same format: a 9×5 street grid, 13 sensors (two shared streets, so 11 sites), and 52 days of hourly counts with weekday peaks and quiet weekends. It saves this to `data/synthetic/`, which the MATLAB code reads too.
>
> **Drop the real files into `lab5_DataDrivenModelling/` and rerun:** every script switches to the real data automatically. All Section 1–2 numbers below are from the synthetic data, so only the *patterns* carry over.
>
> Section 3 uses the real bottleneck trajectories (`b160_combined.txt`).

**MATLAB toolboxes:** the sheet uses `arima` (Econometrics), `fitcnet` (Statistics & ML) and `trainNetwork`/LSTM (Deep Learning), none of which you have. The MATLAB scripts therefore use base-MATLAB equivalents: least-squares AR models, the seasonal-difference model and k-nearest-neighbours. They call the toolbox versions automatically if those are installed.

**Python:** statsmodels does the ARIMA/SARIMAX/Holt–Winters models. PyTorch isn't installed, so a scikit-learn neural network on the last 24 hours stands in for the LSTM. It's the same idea: learn $y_t$ from the recent past.

---

## 1. Concepts from the lectures

**Data (lecture 1):**

- **Experiments** are controlled and reproducible; they manipulate "treatments". **Observations** come from the real world and are usually not reproducible.
- Uses of data: inference, model fitting, validation, prediction.
- Issues: cost and ecological validity (experiments); correlation ≠ causation, sampling bias and regression to the mean (observations).
- Ethics: GDPR / Data Protection Act 2018, informed consent.

**Direct demand modelling (lecture 2)** skips the four-step model and predicts traffic at a site directly:

- **Interpolation**, e.g. inverse distance weighting:
  $$\hat y_i = \frac{\sum_j y_j/d_{ij}^p}{\sum_j 1/d_{ij}^p}$$
  Distances are **along the network**, and can be "perceived" distances, e.g. $d'_{ij} = d_{ij}/w_{ij}$. Kriging adds uncertainty estimates.
- **Regression** on site features: network centrality (degree, betweenness), space syntax integration, land use, transport stops.
- Issues: sensors are usually placed at busy spots (bias), and networks are expensive to build and maintain.

**Microscopic / mesoscopic (lecture 3):**

- Learn behaviour $\mathbf a = f(\text{neighbourhood}, \ldots)$ from trajectories, e.g. an occupancy grid fed into a neural network.
- Downsides: a black box, poor extrapolation, and long-term goals are hard to learn.
- Mesoscopic: learn a coarse variable's dynamics, e.g. drift and diffusion of the counter-flow at a bottleneck (Fokker–Planck), or equation-free modelling.

**Time series (lecture 4):**

- Properties: autocorrelation; decomposition into trend + seasonality + noise (additive or multiplicative).
- **Structural models** combine trend, AR terms, seasonal terms and exogenous variables, fitted with a Kalman filter.
- **AR(p):** $y_t = c + \sum\phi_ky_{t-k} + \varepsilon_t$.
- **MA(q):** correlated noise.
- **ARIMA:** differencing removes trends.
- **SARIMA:** adds seasonal AR/MA/differencing at lag s (24 h, 168 h).
- **SARIMAX:** adds exogenous inputs.
- **ML:** neural networks, LSTMs, graph neural networks. Often the best short-term forecasters, but less interpretable.

---

## 2. Section 1: direct demand by IDW

**Pipeline:**

1. **Network:** segments are nodes, linked if they share an end point. The weight is half of each length added together (midpoint to midpoint). I vectorised the sheet's double loop with broadcasting.
2. **Assign sensors** to the nearest segment midpoint. Merge two sensors on the same segment by **adding** their counts (opposite pavements); **double** a lone sensor (it only sees one side).
   - ⚠️ The sheet's merge loop fills `signal` in file order but uses the **sorted** `unique(idsens)` as `sensnodes`. These only match if `idsens` is already sorted. My code groups by segment, so the order is always right.
3. **Distances:** `distances(graph(A), sensnodes)` in MATLAB, `dijkstra(A, indices=sensnodes)` in Python.
4. **IDW** with p = 2, excluding the target sensor itself (distance 0).

### Exercises

- **1.0:** **One parameter**, the exponent p (= 2). The lecturer's answer says the same.
- **1.1:** Predict all segments at every hour and animate (`figures/s1_idw_animation.gif`).
  - The pattern pulses with the daily cycle.
  - IDW can only produce values *between* the sensor values, so it can't predict a busier spot than any sensor sees.
- **1.2: A rigorous check is leave-one-sensor-out cross-validation.** Predict each sensor from the others at every hour, then report RMSE/MAE per site and on average.
  - Also: relative errors (sites differ in scale), errors by time of day, and a temporal train/test split.
  - Synthetic data: mean LOO RMSE 939 (p = 2), against 977 for "mean of the other sensors" (p = 0). Not much better. With 11 sparse sensors, IDW is weak.
- **1.3: Extensions.**
  - **(i)** Fit p by cross-validation: best p ≈ 3.5, RMSE 915.
  - **(ii)** Perceived distance $\ell_e(w_{ref}/w_e)^\gamma$, using made-up pavement widths $w_e \sim U(1, 5)$ m. Random widths carry no information, so they barely help (γ = 0.5: 927). The *method* is the point: add covariates and fit their parameters out-of-sample.
  - **(iii)** Per-sensor weights. These add as many parameters as there are sensors, so there's a risk of overfitting, and the model becomes site-specific (less transferable).

## 3. Section 2: time series

**Decomposition** (sheet code):

- trend = `movmean` over a 168 h window;
- seasonal = the average of each hour of the day;
- residual = the rest.

**2.0/2.1:** With a **168 h (weekly)** seasonal period instead of 24 h, the residual std drops from 196 to 81, because weekdays and weekends differ. Arguments for each choice:

- 24 h captures the commute peaks;
- 168 h also captures weekends;
- an annual period captures holidays and weather (but needs years of data).

**Choose by out-of-sample error.**

### Models, 80/20 split (synthetic data)

| model | closed loop RMSE | open loop RMSE |
|---|---|---|
| ARIMA(0,0,0) (mean only) | 393 | 393 |
| ARIMA(1,0,0), φ = 0.72 | 391 | 264 |
| ARIMA(2,0,0) | 393 | 247 |
| sheet's SARIMA: (1−B)(1−B²⁴)y = ε | **2077 (explodes)** | 237 |
| SARIMA (1,0,0)×(0,1,1)₂₄ | 196 | 169 |
| neural net on 24 lags (LSTM stand-in) | 301 | **122** |
| seasonal naive "same hour last week" | 97 | – |
| Holt–Winters, weekly season | **79** | – |

**Closed loop vs open loop:**

- **Closed loop** forecasts all ~250 hours from the end of training:
  - AR forecasts decay to the mean;
  - seasonal differencing repeats the last day *plus its change from the day before, every day*. The training period ends at a weekday/weekend switch, so the sheet's double-differenced model drifts away. (The sheet's real data gave 416.)
- **Open loop** forecasts one step ahead using the true previous value. It is much easier. AR(1) open loop just lags the data by one hour.

**The model has to match the task.** Weekly-seasonal models win on multi-day forecasts.

### Exercises

- **2.2: Higher-order AR** looks further back, which helps open loop slightly (264 → 247). Seasonal terms matter far more than extra AR lags.
- **2.3: Closed loop for the ML model:** loop, predict one step, append the prediction to the history, predict again. It degrades from 122 to 301 as errors compound.
- **2.4: Exogenous inputs.**
  - A **weekend flag** helps (SARIMAX 196 → 177; neural net open loop 122 → 89).
  - Random "rain" adds nothing (195), as expected for noise.
  - In MATLAB's LSTM you would increase `numFeatures` in `sequenceInputLayer`.
- **2.5: Rolling-origin evaluation.** Use many forecast origins and a fixed horizon, refit each time, and report mean ± sd. Example: SARIMA 189 ± 73 vs seasonal naive 225 ± 148 for 24 h ahead.
  - Also: several error measures (MAE, MAPE), several horizons, and time-series cross-validation, *never* shuffled.
- **2.6: Fit the same model per sensor** and compare the parameters. They're similar shapes at different scales, which suggests a "common profile × site level" model or log counts.
- **2.7: Alternatives:** Holt–Winters exponential smoothing (best here), a structural time-series model with a Kalman filter, Prophet-style regression with Fourier terms, gradient boosting on lag features.

## 4. Section 3: machine-learning a microscopic model (real data)

**Data handling:**

- Bottleneck experiment trajectories; remove y > 0 (inside the bottleneck).
- Use 10 cm cells, downsampled to 2 Hz: 178 people, 4,204 positions.
- **Input:** a 9×9 occupancy grid centred on the person.
- **Output:** the cell they occupy next, in a 7×7 grid, i.e. **49-class classification**.
- ⚠️ MATLAB flattens grids **column-major** (`raus(:)`); the Python port uses `order='F'`, so class numbers match `vec2ind`.

| model / evaluation | accuracy | within 1 cell | mean error |
|---|---|---|---|
| sheet: random samples, NN | 0.41 | 0.78 | 9.4 cm |
| **split by pedestrian**, NN | **0.15** | 0.75 | 12.8 cm |
| baseline "repeat previous step" (no ML) | 0.13 | 0.71 | 13.4 cm |
| NN + previous step + direction to exit | 0.16 | 0.75 | 12.6 cm |

MATLAB kNN shows the same drop: 0.29 → 0.14.

### Exercises

- **3.0: Why is the sheet's sampling poor?** It draws random (person, time) pairs **with replacement**, separately for train and test.
  - The same person, and often the *same moment*, appears in both sets. The test score is inflated by memorisation: 41% vs 15% honestly.
  - Results change every run (no fixed seed) and samples are duplicated.
  - **Fix:** split by person, use every sample once, fix the seed.
- **3.1: Reading the confusion matrix.** The diagonal is correct predictions, off-diagonal is wrong. Only the observed classes appear (not all 49).
  - It treats a 1-cell miss the same as a 3-cell miss, so also report the **distance error** and the "within one cell" rate.
  - Most errors are near misses between neighbouring steps.
- **3.2: Limitations.**
  - Coarse discretisation (10 cm cells, 0.5 s steps) and an arbitrary grid size.
  - **No walls, no exit/goal, no velocity** in the input.
  - Classification can't reward near misses.
  - Small data (178 people from one experiment), so it may not transfer.
  - The black box teaches us little about behaviour.
  - Open-loop evaluation only.
- **3.3: Improvements.**
  - Add momentum (the previous step) and the direction and distance to the exit: a small gain.
  - **Compare against a trivial baseline:** the network barely beats "keep doing what you did".
  - Further: walls in the occupancy grid, neighbours' velocities, regression on the continuous displacement instead of classes, sequence models (LSTM), or hybrid models (e.g. learn the force terms of the social force model).
- **3.4: Closed loop.** Start from the true position, then feed each predicted position back in as the person's position (other people still follow the data). Errors accumulate and the predicted path drifts (`figures/s3_ml_model.png`). Plausible behaviour over long horizons needs goals and walls in the model.

## 5. Coursework-style patterns

- Leave-one-out / k-fold cross-validation for spatial models, and rolling-origin evaluation for time series. Never shuffle time.
- **Always compare against a naive baseline** (the mean, seasonal naive, persistence).
- Report errors that match the use case: horizon, units, near misses.
- Watch out for leakage between training and test data.
- Python ↔ MATLAB: `np.ravel(order='F')` ↔ `(:)`; `statsmodels ARIMA(order, seasonal_order)` ↔ `arima(p, D, q, 'Seasonality', s)`.
