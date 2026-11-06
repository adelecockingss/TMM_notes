%% TMM Lab 01 - Q3: Trip data - building the OD matrix (MATLAB version)
% Base MATLAB only. Saves data/od_msoa.mat and data/od_utla.mat for Q4-Q8.
% Python version: q3_od_matrix.py
clear; close all;
if ~exist('figures', 'dir'), mkdir figures; end

%% Load the census OD file - only the columns we need
f = '../ODWP01EW_MSOA.csv';
opts = detectImportOptions(f, 'VariableNamingRule', 'preserve');
opts.SelectedVariableNames = opts.VariableNames([1 3 5 7]);
opts = setvartype(opts, opts.SelectedVariableNames(1:2), 'categorical');
opts = setvartype(opts, opts.SelectedVariableNames(3:4), 'int32');
tic; od = readtable(f, opts); toc
od.Properties.VariableNames = {'o', 'd', 'pow', 'count'};
% pow: -8 not working, 1 work from home, 2 offshore/abroad, 3 work in UK (a trip)
disp(groupsummary(od, 'pow', 'sum', 'count'))

%% Q3(b) lookup table: index <-> MSOA code (2021 MSOAs from the coursework .mat)
S = load('../TMMcoursework2024Q1 (1).mat');
codes = string(S.MSOAdata.code);
labels = string(S.MSOAdata.label);
E = S.MSOAdata.e;  N = S.MSOAdata.n;
utlaIdx = double(S.MSOAdata.UTLAindex);
n = numel(codes);

%% Q3(a) sparse OD matrix
trips = od(od.pow == 3, :);
[okO, i] = ismember(string(trips.o), codes);
[okD, j] = ismember(string(trips.d), codes);    % Scotland/NI workplaces -> not found
ok = okO & okD;
T = sparse(i(ok), j(ok), double(trips.count(ok)), n, n);   % duplicates summed
fprintf('T: %d x %d, %d non-zeros (%.2f%% full), %d trips\n', n, n, nnz(T), ...
        100*nnz(T)/n^2, full(sum(T, 'all')));

%% Q3(c) row and column totals
O = full(sum(T, 2));   % origins (column vector)
D = full(sum(T, 1))';  % destinations
fprintf('Intra-zonal trips: %.1f%%\n', 100*full(sum(diag(T)))/sum(O));

%% Q3(d) distance matrix (km). single() halves the memory: 211 MB vs 422 MB
dist = hypot(single(E) - single(E'), single(N) - single(N'))/1000;
fprintf('distance matrix %d x %d, max %.0f km\n', size(dist), max(dist(:)));

%% Q3(e) commuting distance distribution (weighted by people)
[ii, jj, w] = find(T);
d = double(dist(sub2ind([n n], ii, jj)));
clear dist
fprintf('Mean straight-line commute %.1f km\n', sum(w.*d)/sum(w));

figure('Position', [100 100 1100 400]);
subplot(1, 2, 1);
cnt = accumarray(min(floor(d) + 1, 61), w, [61 1]);     % people per 1 km bin
histogram('BinEdges', 0:1:60, 'BinCounts', cnt(1:60)');
xlabel('Distance (km)'); ylabel('People'); title('Commute distance');
subplot(1, 2, 2);
edges = logspace(0, log10(max(d)), 40);
pos = d > 0;
[~, ~, bin] = histcounts(d(pos), edges);
wp = w(pos);
h = accumarray(bin(bin > 0), wp(bin > 0), [numel(edges)-1 1]);
h = h./diff(edges)'/sum(h);                    % probability density
mid = sqrt(edges(1:end-1).*edges(2:end));
loglog(mid, h, 'o-'); xlabel('Distance (km)'); ylabel('density'); title('Log-log');
sel = mid' > 10 & mid' < 200 & h > 0;
p = polyfit(log(mid(sel)), log(h(sel)), 1);
fprintf('Power-law slope 10-200 km: %.2f\n', p(1));
exportgraphics(gcf, 'figures/q3e_commute_distance_matlab.png');

% NTS-style mile bands (straight line, so shorter than road distances)
bands = [0 1 2 5 10 25 50 inf];
[~, ~, b] = histcounts(d/1.609, bands);
share = accumarray(b, w, [7 1]); disp(round(100*share'/sum(share), 1))

save('data/od_msoa.mat', 'T', 'O', 'D', 'E', 'N', 'codes', 'labels', 'utlaIdx');

%% Q3(f) coarser geography: UTLA.  T_utla = P T P'
m = numel(S.UTLAdata.code);
P = sparse(utlaIdx, 1:n, 1, m, n);
TU = full(P*T*P');
OU = sum(TU, 2); DU = sum(TU, 1)';
EU = (P*(E.*O))./(P*O);  NU = (P*(N.*O))./(P*O);   % origin-weighted centroids
distU = hypot(EU - EU', NU - NU')/1000;
utlaLabels = string(S.UTLAdata.label);
fprintf('UTLA: %d zones, %.0f%% of trips internal\n', m, 100*trace(TU)/sum(TU(:)));
save('data/od_utla.mat', 'TU', 'OU', 'DU', 'EU', 'NU', 'distU', 'utlaLabels');

%% Q3(g) DataShine-style map: flows into the biggest Bristol attractor
isBristol = startsWith(labels, "Bristol");
[~, k] = max(D.*isBristol);
[~, src, v] = find(T(:, k)');
figure; plot(E/1e3, N/1e3, '.', 'Color', [0.8 0.8 0.8], 'MarkerSize', 2); hold on;
for q = 1:numel(src)
    plot([E(k) E(src(q))]/1e3, [N(k) N(src(q))]/1e3, 'Color', [0 0.4 0.8 0.5], ...
         'LineWidth', 0.3 + 4*v(q)/max(v));
end
axis equal; xlim([330 400]); ylim([140 210]); title("To " + labels(k));
exportgraphics(gcf, 'figures/q3g_flow_map_matlab.png');
