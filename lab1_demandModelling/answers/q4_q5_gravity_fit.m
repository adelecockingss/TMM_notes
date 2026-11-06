%% TMM Lab 01 - Q4 & Q5: unconstrained gravity model T_ij ~ c O_i D_j f(d_ij)
% Base MATLAB. Run q3_od_matrix.m first. Python version: q4_q5_gravity_fit.py
clear; close all;
level = 'bristol';                 % 'bristol', 'utla' or 'all'
z = load_od_level(level);
T = z.T; O = z.O; D = z.D; dist = z.dist; n = numel(O);
fprintf('%s: %d zones, %d trips\n', level, n, sum(T(:)));

%% Q4: f(d) = 1/d^2, unravel into an n^2 x 2 matrix
OD = O*D';                          % outer product O_i D_j
X = OD./dist.^2;
pairs = [X(:), T(:)];

% Three ways to choose c (see NOTES.md):
c_ols  = sum(X(:).*T(:))/sum(X(:).^2);              % least squares, dominated by big flows
nz = T(:) > 0;
c_log  = exp(mean(log(T(nz)) - log(X(nz))));         % least squares on logs, ignores zeros
c_pois = sum(T(:))/sum(X(:));                         % Poisson maximum likelihood
fit_report(sprintf('1/d^2, c by OLS (%.2e)', c_ols), c_ols*X, T);
fit_report(sprintf('1/d^2, c by log (%.2e)', c_log), c_log*X, T);
fit_report(sprintf('1/d^2, c by Poisson (%.2e)', c_pois), c_pois*X, T);

figure;
scatter(log10(pairs(nz, 1)), log10(pairs(nz, 2)), 2, 'filled', 'MarkerFaceAlpha', 0.1); hold on;
xx = [min(log10(pairs(nz,1))) max(log10(pairs(nz,1)))];
plot(xx, xx + log10(c_ols), 'r', xx, xx + log10(c_log), 'g', xx, xx + log10(c_pois), 'b');
legend('pairs', 'OLS', 'log', 'Poisson', 'Location', 'northwest');
xlabel('log_{10} O_iD_j/d_{ij}^2'); ylabel('log_{10} T_{ij}'); title('Q4 gravity fit');

%% Q5: f(d) = d^-gamma, scan gamma (c by Poisson for each)
gammas = 0.5:0.1:4;
cp = zeros(size(gammas));
for k = 1:numel(gammas)
    Xg = OD./dist.^gammas(k);
    cp(k) = cpc(sum(T(:))/sum(Xg(:))*Xg, T);
end
[~, kb] = max(cp);
fprintf('Best power by CPC: gamma = %.1f\n', gammas(kb));
figure; plot(gammas, cp, 'o-'); xline(2, ':'); xlabel('\gamma'); ylabel('CPC');
title('Power-law deterrence d^{-\gamma}');

%% Q5: empirical f(d) = sum T / sum O_iD_j in distance bins
edges = logspace(log10(min(dist(:))), log10(max(dist(:))), 30);
edges([1 end]) = edges([1 end]).*[0.999 1.001];   % rounding: keep min and max inside
[~, ~, bin] = histcounts(dist(:), edges);
f_emp = accumarray(bin, T(:), [numel(edges)-1 1])./accumarray(bin, OD(:), [numel(edges)-1 1]);
mid = sqrt(edges(1:end-1).*edges(2:end))';
ok = isfinite(f_emp) & f_emp > 0;
p = polyfit(log(mid(ok)), log(f_emp(ok)), 1);
figure; loglog(mid, f_emp, 'o', mid, exp(p(2))*mid.^p(1), 'r-');
xlabel('distance (km)'); ylabel('f(d)'); title(sprintf('Empirical f(d), slope %.2f', p(1)));

%% Q5: generalised gravity T = k O^alpha D^beta d^-gamma as a Poisson GLM
[Oi, Dj] = ndgrid(O, D);              % Oi(i,j) = O(i), Dj(i,j) = D(j)
ok = OD(:) > 0;
beta = poisson_glm(T(ok), [log(Oi(ok)) log(Dj(ok)) log(dist(ok))]);
fprintf('Poisson GLM: alpha = %.2f, beta = %.2f, gamma = %.2f\n', beta(2), beta(3), -beta(4));
Tm = zeros(n); Tm(ok) = exp([ones(nnz(ok),1) log(Oi(ok)) log(Dj(ok)) log(dist(ok))]*beta);
fit_report('generalised gravity (GLM)', Tm, T);
