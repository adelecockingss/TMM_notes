%% TMM Lab 05 - Section 3: machine-learning a microscopic model (MATLAB)
% The sheet's fitcnet needs the Statistics and ML Toolbox (not installed here).
% Base-MATLAB alternative: a k-nearest-neighbour classifier written out below.
% If you have the toolbox, fitcnet is used as well. Python: s3_ml_microscopic.py
clear; close all; rng(0);
addpath('..');                                   % dataparcel.m lives in the lab folder
data = load('../b160_combined (1).txt');
data(data(:, 4) > 0, :) = [];                    % remove the bottleneck
ids = unique(data(:, 1)); times = unique(data(:, 2));
datanew = data;
datanew(:, 3:4) = [floor((data(:,3) + 250)/10) + 1, floor(abs(data(:,4))/10) + 1];  % vectorised
datanew = datanew(ismember(datanew(:, 2), min(times):10:max(times)), :);

%% Worked example: random parcels (sheet), 4000 train / 1000 test
[Xtr, ytr] = parcels(ids, 4000, datanew);
[Xte, yte] = parcels(ids, 1000, datanew);
yk = knn_predict(Xtr, ytr, Xte, 5);
fprintf('random sampling (sheet), kNN: accuracy %.3f\n', mean(yk == yte));
if exist('fitcnet', 'file') == 2
    Mdl = fitcnet(Xtr, ytr, 'LayerSizes', 100, 'Activations', 'relu', 'Standardize', true);
    fprintf('random sampling (sheet), fitcnet: accuracy %.3f\n', mean(predict(Mdl, Xte) == yte));
end

%% Exercise 3.0: split by PEDESTRIAN instead (no person in both sets)
perm = ids(randperm(numel(ids)));
trainIds = perm(1:round(0.8*numel(ids))); testIds = perm(round(0.8*numel(ids))+1:end);
[Xtr2, ytr2] = parcels(trainIds, 4000, datanew);
[Xte2, yte2] = parcels(testIds, 1000, datanew);
yk2 = knn_predict(Xtr2, ytr2, Xte2, 5);
fprintf('split by pedestrian, kNN: accuracy %.3f\n', mean(yk2 == yte2));

%% Exercise 3.1: confusion matrix and distance error
labels = unique([yte2; yk2]);
[~, ti] = ismember(yte2, labels); [~, pi_] = ismember(yk2, labels);
C = accumarray([ti pi_], 1, [numel(labels) numel(labels)]);
figure; imagesc(log1p(C)); axis square; colorbar;
xlabel('predicted class'); ylabel('true class'); title('confusion matrix (log counts)');
toxy = @(c) [mod(c - 1, 7) - 3, floor((c - 1)/7) - 3];   % vec2ind class -> (dx, dy)
err = vecnorm(toxy(yk2) - toxy(yte2), 2, 2);
fprintf('mean position error %.1f cm, within 1 cell %.3f\n', 10*mean(err), mean(err <= 1.5));

function [X, y] = parcels(idlist, N, datanew)
% N random parcels (as in the sheet) from the given pedestrians
X = zeros(N, 81); y = zeros(N, 1); k = 0;
while k < N
    pid = idlist(randi(numel(idlist)));
    if nnz(datanew(:, 1) == pid) < 2, continue; end
    [in, out] = dataparcel(pid, datanew);
    if ~any(out), continue; end                  % step bigger than 3 cells
    k = k + 1; X(k, :) = in'; y(k) = find(out);  % = vec2ind(out)
end
end

function yp = knn_predict(Xtr, ytr, Xte, k)
% k-nearest neighbours: the most common class among the k training inputs
% closest (Euclidean) to each test input. Base MATLAB.
D = sum(Xte.^2, 2) + sum(Xtr.^2, 2)' - 2*Xte*Xtr';
[~, nb] = mink(D, k, 2);
yp = mode(ytr(nb), 2);
end
