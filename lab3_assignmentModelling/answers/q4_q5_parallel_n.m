%% TMM Lab 03 - Q4 & Q5: n parallel links (MATLAB version, base MATLAB)
% Python version: q4_q5_parallel_n.py
clear; close all; rng(0);
sysCost = @(x, a, b) sum(x.*(a + b.*x));

%% Q4: 5 random affine links; water-filling vs the QP formulation
n = 5; a = 2*rand(n, 1); b = 0.1 + 1.9*rand(n, 1); d = 2;
xUE = parallel_affine(a, b, d, 'UE'); xSO = parallel_affine(a, b, d, 'SO');
xUEqp = qp_small(diag(b), a, ones(1, n), d);
fprintf('UE flows %s\nUE costs %s\nQP agrees: %d\n', mat2str(xUE', 4), ...
    mat2str((a + b.*xUE)', 4), norm(xUE - xUEqp) < 1e-8);
worst = 1; ds = linspace(0.01, 6, 300);
for trial = 1:200
    a = 2*rand(n, 1); b = 2*rand(n, 1);
    poa = arrayfun(@(dd) sysCost(parallel_affine(a, b, dd, 'UE'), a, b)/ ...
                         sysCost(parallel_affine(a, b, dd, 'SO'), a, b), ds);
    worst = max(worst, max(poa));
end
fprintf('Q4 worst PoA over 200 random networks: %.4f (bound 4/3)\n', worst);

%% Q5: Pigou with c1 = x^p, c2 = 1, d = 1 - closed form
% UE: x1 = 1 (c1(1) = 1 = c2), cost 1.  SO: marginal (p+1) x1^p = 1.
for p = 1:8
    x1 = (p + 1)^(-1/p);
    fSO = x1^(p+1) + (1 - x1);
    fprintf('p = %d: PoA = %.4f\n', p, 1/fSO);
end
% For general nonlinear costs use fzero (base MATLAB) on the equal-cost
% condition, e.g. BPR links t = t0 (1 + 0.15 (x/cap)^4):
t0 = [10 15]; cap = [1000 2000]; d = 2000;
c = @(x, i) t0(i)*(1 + 0.15*(x/cap(i))^4);
x1 = fzero(@(x) c(x, 1) - c(d - x, 2), [0 d]);                 % UE: equal costs
m = @(x, i) t0(i)*(1 + 0.75*(x/cap(i))^4);                     % marginal cost
x1so = fzero(@(x) m(x, 1) - m(d - x, 2), [0 d]);               % SO: equal marginal
f = @(x1) x1*c(x1, 1) + (d - x1)*c(d - x1, 2);
fprintf('BPR d = %d: UE x1 = %.0f, SO x1 = %.0f, PoA %.4f\n', d, x1, x1so, f(x1)/f(x1so));
