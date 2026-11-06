%% TMM Lab 01 - Q2: Basics of the census data (MATLAB version)
% Base MATLAB only. If you have the Statistics Toolbox, fitlm(X, y) gives
% the same numbers as ols_fit(y, X) used here. Python: q2_census_stats.py
clear; close all;
if ~exist('figures', 'dir'), mkdir figures; end

%% Load with sensible types (what the Import Tool generates, tidied)
opts = detectImportOptions('../UK_census_msoa_stats.csv', 'VariableNamingRule', 'preserve');
opts = setvartype(opts, {'MSOA Code', 'MSOA Name'}, 'string');
opts = setvartype(opts, {'Population', 'Area', 'Households', 'O (trips)', 'D (trips)'}, 'uint32');
T = readtable('../UK_census_msoa_stats.csv', opts);
T.Properties.VariableNames = {'code', 'name', 'lon', 'lat', 'pop', 'area_ha', ...
                              'households', 'townsend', 'O', 'D'};
T.Properties.RowNames = T.code;               % look up rows by code: T('E02000001',:)

% Categorical variables: a few repeated values
T.country = categorical(extractBefore(T.code, 2), {'E', 'W'}, {'England', 'Wales'});
% Q2(d)(i): "Barking and Dagenham 001A" -> "Barking and Dagenham"
T.la = categorical(regexprep(T.name, '\s+\d{3}[A-Z]?$', ''));
[T.E, T.N] = latlon_to_osgb(T.lat, T.lon);
fprintf('%d local authorities, %d MSOAs\n', numel(categories(T.la)), height(T));
summary(T(:, {'pop', 'O', 'D', 'townsend'}))

%% Q2(a) O against population
[c, R2] = ols_fit(T.O, T.pop, {'pop'});
disp(c); fprintf('R^2 = %.3f\n', R2);
figure; scatter(T.pop, T.O, 3, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
x = [0 double(max(T.pop))];
plot(x, c.Estimate(1) + c.Estimate(2)*x, 'r', x, x, 'k--');
xlabel('Population'); ylabel('O(i)'); legend('MSOAs', 'OLS fit', 'O = pop');
title(sprintf('Q2(a) R^2 = %.2f', R2));
exportgraphics(gcf, 'figures/q2a_O_vs_pop_matlab.png');

%% Q2(b) correlation matrix
vars = {'O', 'D', 'pop', 'households', 'area_ha', 'townsend'};
% NB: T{:, vars} would concatenate uint32 + double -> uint32 (rounding the
% Townsend score!). Convert each column to double first.
M = table2array(varfun(@double, T(:, vars)));
R = corrcoef(M);
disp(array2table(round(R, 3), 'VariableNames', vars, 'RowNames', vars))
figure; imagesc(R, [-1 1]); colorbar; axis square;
set(gca, 'XTick', 1:6, 'XTickLabel', vars, 'YTick', 1:6, 'YTickLabel', vars);
title('Pearson correlation');

%% Q2(b) simple regressions of O on each variable
preds = {'pop', 'households', 'area_ha', 'townsend', 'D'};
res = table('Size', [numel(preds) 4], 'VariableTypes', repmat({'double'}, 1, 4), ...
    'VariableNames', {'slope', 'SE', 'pValue', 'R2'}, 'RowNames', preds);
for k = 1:numel(preds)
    [c, R2] = ols_fit(T.O, T.(preds{k}), preds(k));
    res{k, :} = [c.Estimate(2), c.SE(2), c.pValue(2), R2];
end
disp(res)

%% Q2(b) multiple regression O ~ pop + households + log(area) + townsend
T.log_area = log(double(T.area_ha));
mv = {'pop', 'households', 'log_area', 'townsend'};
X = table2array(varfun(@double, T(:, mv)));   % see uint32 warning above
[c, R2] = ols_fit(T.O, X, mv);
disp(c); fprintf('Multiple regression R^2 = %.3f\n', R2);

% Variance inflation factor: regress each predictor on the others
for k = 1:numel(mv)
    [~, Rk] = ols_fit(X(:, k), X(:, setdiff(1:numel(mv), k)));
    fprintf('VIF %-10s %.2f\n', mv{k}, 1/(1 - Rk));
end

%% Q2(c) D is badly predicted by everything here
[~, R2] = ols_fit(T.D, X); fprintf('D ~ pop+households+log_area+townsend: R^2 = %.3f\n', R2);
for k = {'pop', 'households', 'area_ha', 'townsend', 'O'}
    [~, R2] = ols_fit(T.D, T.(k{1})); fprintf('  D ~ %-10s R^2 = %.3f\n', k{1}, R2);
end
figure; scatter(T.O, T.D, 3, 'filled'); set(gca, 'YScale', 'log');
xlabel('O(i)'); ylabel('D(i)'); title('D vs O');
[~, top] = maxk(T.D, 4);
text(double(T.O(top)), double(T.D(top)), T.name(top), 'FontSize', 7);

%% Q2(d)(i) aggregate to local authority via the name strings
LA = groupsummary(T, 'la', 'sum', {'pop', 'O', 'D'});
LA = sortrows(LA, 'sum_D', 'descend');
disp(LA(1:10, :))

%% Q2(d)(ii) clustering (1): 25 km grid cells
cell = 25e3;
key = string(floor(T.E/cell)) + "_" + string(floor(T.N/cell));
[g, ~] = findgroups(key);
G = table(splitapply(@sum, double(T.pop), g), splitapply(@sum, double(T.O), g), ...
          splitapply(@sum, double(T.D), g), splitapply(@mean, T.E, g), splitapply(@mean, T.N, g), ...
          'VariableNames', {'pop', 'O', 'D', 'E', 'N'});
fprintf('%d non-empty grid cells\n', height(G));

%% Q2(d)(ii) clustering (2): Anders's random nearest-neighbour merging
vals = double([T.pop T.O T.D]);
targets = [height(T) 2000 500 100];
edges = linspace(0, 3, 61);
figure('Position', [100 100 1400 500]);
for k = 1:numel(targets)
    v = merge_zones(T.E, T.N, vals, targets(k));
    subplot(2, 4, k);   histogram(v(:,2)./v(:,1), edges, 'Normalization', 'pdf');
    title(sprintf('N = %d: O/pop', targets(k)));
    subplot(2, 4, k+4); histogram(min(v(:,3)./v(:,1), 3), edges, 'Normalization', 'pdf', ...
        'FaceColor', [0.9 0.5 0.1]);
    title(sprintf('N = %d: D/pop', targets(k)));
end
exportgraphics(gcf, 'figures/q2d_merged_histograms_matlab.png');

%% Q2(d)(iii) official lookup (2021 MSOA -> UTLA, from the coursework .mat)
S = load('../TMMcoursework2024Q1 (1).mat');
lookupCodes = string(S.MSOAdata.code);
utlaNames = string(S.UTLAdata.label);
[found, loc] = ismember(T.code, lookupCodes);
T.utla = strings(height(T), 1);
T.utla(found) = utlaNames(S.MSOAdata.UTLAindex(loc(found)));
fprintf('%d of %d MSOAs matched to a UTLA\n', nnz(found), height(T));
U = groupsummary(T(found, :), 'utla', 'sum', {'pop', 'O', 'D'});
disp(sortrows(U, 'sum_D', 'descend'))

%% Q2(e) map: attractors (red) vs generators (blue)
figure; scatter(T.E/1e3, T.N/1e3, 3, log10(max(double(T.D)./double(T.O), 0.01)), 'filled');
colormap(redblue_map()); clim([-1 1]); cb = colorbar; cb.Label.String = 'log_{10}(D/O)';
axis equal; xlabel('Easting (km)'); ylabel('Northing (km)'); title('Attractors vs generators');
exportgraphics(gcf, 'figures/q2e_map_matlab.png');

%% ---- local functions ----------------------------------------------------
function vals = merge_zones(x, y, vals, nTarget)
% Pick a random zone, merge with its nearest neighbour, repeat.
rng(0);
w = vals(:, 1);
alive = true(numel(x), 1);
while nnz(alive) > nTarget
    idx = find(alive);
    i = idx(randi(numel(idx)));
    d2 = (x(idx) - x(i)).^2 + (y(idx) - y(i)).^2;
    d2(idx == i) = inf;
    [~, m] = min(d2); j = idx(m);
    x(i) = (w(i)*x(i) + w(j)*x(j))/(w(i) + w(j));     % population-weighted centroid
    y(i) = (w(i)*y(i) + w(j)*y(j))/(w(i) + w(j));
    w(i) = w(i) + w(j);
    vals(i, :) = vals(i, :) + vals(j, :);
    alive(j) = false;
end
vals = vals(alive, :);
end

function cmap = redblue_map()
t = linspace(0, 1, 128)';
cmap = [[t; ones(128,1)], [t; flipud(t)], [ones(128,1); flipud(t)]];
end
