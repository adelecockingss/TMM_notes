%% TMM Lab 05 - Section 2: time series (MATLAB, base only)
% The sheet uses arima/estimate/forecast (Econometrics Toolbox) and an LSTM
% (Deep Learning Toolbox) - not installed here. Base-MATLAB equivalents:
%   AR(p) fitted by least squares, the seasonal-difference model, seasonal naive.
% If you have the toolbox, the sheet's code runs at the bottom. Python: s2_time_series.py
clear; close all;
[central, ~, ~, ~, ~, synthetic] = load_melbourne();
timeseries = central(4, :); n = length(timeseries);

%% Decomposition (sheet code; try window sizes for exercise 2.0/2.1)
for window_size2 = [24 168]
    trend_component = movmean(timeseries, 24*7);
    detrended = timeseries - trend_component;
    k = ceil(n/window_size2);
    padded = [detrended, nan(1, k*window_size2 - n)];
    seasonal = mean(reshape(padded, window_size2, k), 2, 'omitnan')';
    seasonal = repmat(seasonal, 1, k); seasonal = seasonal(1:n);
    residual = timeseries - trend_component - seasonal;
    fprintf('season %3d h: residual std %.1f\n', window_size2, std(residual));
end
figure; subplot(4,1,1); plot(timeseries); title('original');
subplot(4,1,2); plot(trend_component); title('trend'); subplot(4,1,3); plot(seasonal);
title('seasonal (weekly)'); subplot(4,1,4); plot(residual); title('residual');

%% Train/test split
train_size = round(0.8*n); test_size = n - train_size;
train_data = timeseries(1:train_size); test_data = timeseries(train_size+1:end);
rmse = @(a, b) sqrt(mean((a(:) - b(:)).^2));

%% AR(p) by least squares:  y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + e_t
for p = [0 1 2]
    [c, phi] = fit_ar(train_data, p);
    % closed loop: feed forecasts back in
    hist = train_data; fc = zeros(1, test_size);
    for t = 1:test_size
        fc(t) = c + phi'*flip(hist(end-p+1:end))'; hist(end+1) = fc(t); %#ok<AGROW>
        if p == 0, fc(t) = c; end
    end
    % open loop: use the true previous values
    full = timeseries; op = zeros(1, test_size);
    for t = 1:test_size
        idx = train_size + t;
        op(t) = c; if p > 0, op(t) = c + phi'*flip(full(idx-p:idx-1))'; end
    end
    fprintf('AR(%d): c = %.1f, phi = %s | closed RMSE %.1f, open RMSE %.1f\n', p, c, ...
        mat2str(phi', 3), rmse(fc, test_data), rmse(op, test_data));
end

%% The sheet's SARIMA: (1 - B)(1 - B^24) y_t = e_t  (no constant)
% Forecast: y_t = y_{t-1} + y_{t-24} - y_{t-25}. Closed loop repeats the last
% day PLUS the last day-to-day change, every day -> drifts if the training
% period ends on a change (e.g. weekday -> weekend).
hist = train_data;
for t = 1:test_size
    hist(end+1) = hist(end) + hist(end-23) - hist(end-24); %#ok<AGROW>
end
full = timeseries; op = zeros(1, test_size);
for t = 1:test_size
    i = train_size + t; op(t) = full(i-1) + full(i-24) - full(i-25);
end
fprintf('SARIMA(0,1,0)x(0,1,0)24: closed RMSE %.1f, open RMSE %.1f\n', ...
        rmse(hist(train_size+1:end), test_data), rmse(op, test_data));
% seasonal naive: same hour last week
sn = timeseries(train_size+1-168:n-168);
fprintf('seasonal naive (same hour last week): RMSE %.1f\n', rmse(sn, test_data));

%% With the Econometrics Toolbox: the sheet's code
if exist('arima', 'file') == 2
    fit_sarima = estimate(arima('D', 1, 'Seasonality', 24), train_data');
    f = forecast(fit_sarima, test_size, 'Y0', train_data');
    fprintf('toolbox SARIMA closed RMSE %.1f\n', rmse(f, test_data));
end

function [c, phi] = fit_ar(y, p)
% least squares AR(p) with intercept
if p == 0, c = mean(y); phi = zeros(0, 1); return; end
Y = y(p+1:end)';
X = ones(numel(Y), 1);
for k = 1:p, X = [X, y(p+1-k:end-k)']; end %#ok<AGROW>
b = X\Y; c = b(1); phi = b(2:end);
end
