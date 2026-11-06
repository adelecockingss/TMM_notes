%% TMM Lab 05 - Section 1: direct demand modelling (MATLAB, base only)
% Worked example from the lab sheet + exercises 1.0-1.3. Python: s1_direct_demand.py
clear; close all;
addpath('..');                                 % IDWprediction.m lives in the lab folder
[central, xs, ys, sensx, sensy, synthetic] = load_melbourne();
n = length(xs)/2;

%% Network: segments are nodes, linked if they share an end point
% Vectorised version of the sheet's double loop.
x1 = xs(1:2:end); x2 = xs(2:2:end); y1 = ys(1:2:end); y2 = ys(2:2:end);
len = hypot(x2 - x1, y2 - y1);
EX = [x1 x2]; EY = [y1 y2];                     % n x 2 end points
touch = false(n);
for a = 1:2
    for b = 1:2
        touch = touch | (EX(:,a) == EX(:,b)' & EY(:,a) == EY(:,b)');
    end
end
touch(1:n+1:end) = false;
A = touch.*((len + len')/2);
fprintf('%d segments, %d adjacencies\n', n, nnz(A)/2);

%% Sensors -> nearest segment; merge pairs on the same segment
midsx = (x1 + x2)/2; midsy = (y1 + y2)/2;
idsens = zeros(numel(sensx), 1);
for i = 1:numel(sensx)
    dds = sqrt((midsx - sensx(i)).^2 + (midsy - sensy(i)).^2);
    if i == 9 && ~synthetic
        [~, oo] = sort(dds); idsens(i) = oo(2);    % sheet's manual correction
    else
        [~, idsens(i)] = min(dds);
    end
end
sensnodes = unique(idsens);                    % sorted
signal = zeros(numel(sensnodes), size(central, 2));
for k = 1:numel(sensnodes)                     % group by segment (robust to file order)
    on = idsens == sensnodes(k);
    signal(k, :) = sum(central(on, :), 1)*(1 + (nnz(on) == 1));
end

%% Distances and IDW (IDWprediction.m from Blackboard)
net = graph(A);
alldists2 = distances(net, sensnodes);
IDWprediction(sensnodes(1), signal(:, 100), sensnodes, alldists2)
signal(1, 100)
focal = 3; lala = zeros(1, 300);
for i = 1:300
    lala(i) = IDWprediction(sensnodes(focal), signal(:, i), sensnodes, alldists2);
end
figure; plot(signal(focal, 1:300), 'LineWidth', 2); hold on; plot(lala, 'b', 'LineWidth', 2);
legend('ground truth', 'IDW prediction'); xlabel('time (hours)'); ylabel('pedestrian count');

%% Exercise 1.1: predict every segment, animate over time
P = zeros(n, 48);
for t = 1:48
    P(:, t) = IDWprediction(1:n, signal(:, 48 + t), sensnodes, alldists2);
end
figure;
for t = 1:48
    cla; hold on;
    for i = 1:n
        c = min(P(i, t)/prctile_simple(P(:), 99), 1);
        line([x1(i) x2(i)], [y1(i) y2(i)], 'Color', [c 0.2 1-c], 'LineWidth', 4);
    end
    scatter(sensx, sensy, 30, 'g', 'filled'); axis equal; title(sprintf('hour %d', 48 + t));
    drawnow;   % (use getframe + VideoWriter to save a video)
end

%% Exercise 1.2: leave-one-sensor-out RMSE per site, p = 2
T = 1:3:size(signal, 2);
rmse = loo_rmse(signal, sensnodes, alldists2, 2, T);
fprintf('LOO RMSE per site (p = 2): %s\nmean %.1f\n', mat2str(round(rmse')), mean(rmse));

%% Exercise 1.3: fit the exponent p by cross-validation (fminbnd = base MATLAB)
pbest = fminbnd(@(p) mean(loo_rmse(signal, sensnodes, alldists2, p, T)), 0, 6);
fprintf('best exponent p = %.2f, mean LOO RMSE %.1f\n', pbest, ...
        mean(loo_rmse(signal, sensnodes, alldists2, pbest, T)));

function r = loo_rmse(signal, sensnodes, D, p, T)
r = zeros(numel(sensnodes), 1);
for k = 1:numel(sensnodes)
    pr = zeros(1, numel(T));
    for j = 1:numel(T)
        d = D(:, sensnodes(k)); use = d > 0;
        w = 1./d(use).^p;
        pr(j) = sum(w.*signal(use, T(j)))/sum(w);
    end
    r(k) = sqrt(mean((pr - signal(k, T)).^2));
end
end

function v = prctile_simple(x, q)
x = sort(x(:)); v = x(max(1, round(q/100*numel(x))));
end
