%% TMM Lab 01 - Q6 & Q7: singly and doubly constrained gravity models
% Base MATLAB. Run q3_od_matrix.m first. Python version: q6_q7_constrained.py
clear; close all;
z = load_od_level('bristol');
T = z.T; O = z.O; D = z.D; dist = z.dist;
F = 1./dist.^2;

%% Q6(a) unconstrained model: wrong row and column sums
X = (O*D').*F;
T_un = sum(T(:))/sum(X(:))*X;
fprintf('Unconstrained: row sums off by %.0f%%, col sums off by %.0f%% (mean abs)\n', ...
    100*mean(abs(sum(T_un, 2)./O - 1)), 100*mean(abs(sum(T_un, 1)'./D - 1)));

%% Q6(b) origin constrained: A_i = O_i / sum_j T_un_ij
A = O./sum(T_un, 2);
T_oc = A.*T_un;
fprintf('Origin-constrained: max row error %.1e, col sums off by %.0f%%\n', ...
    max(abs(sum(T_oc, 2) - O)), 100*mean(abs(sum(T_oc, 1)'./D - 1)));
B = D./sum(T_un, 1)';
T_dc = T_un.*B';
fprintf('Destination-constrained: max col error %.1e, row sums off by %.0f%%\n', ...
    max(abs(sum(T_dc, 1)' - D)), 100*mean(abs(sum(T_dc, 2)./O - 1)));

%% Q7(a,b) doubly constrained by Furness iteration, then verify
[a, b, T_dbl, iters] = furness(O, D, F);
fprintf('Furness: %d iterations; max row error %.1e, max col error %.1e\n', iters, ...
    max(abs(sum(T_dbl, 2) - O)), max(abs(sum(T_dbl, 1)' - D)));
% (With the Optimization Toolbox you could instead solve the 2n equations
%  with fsolve/lsqnonlin on x = [log a; log b] - see the Python version.)

%% Q7(c) fit quality, and calibrating gamma
fit_report('unconstrained 1/d^2', T_un, T);
fit_report('origin-constrained 1/d^2', T_oc, T);
fit_report('doubly constrained 1/d^2', T_dbl, T);

meanObs = sum(T.*dist, 'all')/sum(T(:));
gammas = 0.5:0.1:4; cp = zeros(size(gammas)); mtl = cp;
for k = 1:numel(gammas)
    [~, ~, Tg] = furness(O, D, dist.^-gammas(k), 1e-6);
    cp(k) = cpc(Tg, T); mtl(k) = sum(Tg.*dist, 'all')/sum(Tg(:));
end
[cmax, kb] = max(cp); [~, km] = min(abs(mtl - meanObs));
fprintf('Best gamma by CPC = %.1f (CPC %.3f); mean trip length %.1f km matched at gamma = %.1f\n', ...
    gammas(kb), cmax, meanObs, gammas(km));
figure;
subplot(1, 2, 1); plot(gammas, cp, 'o-'); xlabel('\gamma'); ylabel('CPC');
subplot(1, 2, 2); plot(gammas, mtl, 'o-'); yline(meanObs, 'r--');
xlabel('\gamma'); ylabel('mean trip length (km)');
