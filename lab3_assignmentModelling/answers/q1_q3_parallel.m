%% TMM Lab 03 - Q1-Q3: two parallel links (MATLAB version)
% Uses quadprog if you have the Optimization Toolbox, else qp_small (base MATLAB).
% Python version: q1_q3_parallel.py. By-hand working: NOTES.md
clear; close all;
hasQP = exist('quadprog', 'file') == 2;
solve = @(H, f, Aeq, beq) qpwrap(H, f, Aeq, beq, hasQP);
sysCost = @(x, a, b) sum(x.*(a + b.*x));

%% Q1: c1 = 1/2 + x1, c2 = 1 + x2/2, d = 1
a = [0.5; 1]; b = [1; 0.5]; d = 1;
xUE = solve(diag(b), a, [1 1], d);          % Beckmann: H = diag(b)
xSO = solve(2*diag(b), a, [1 1], d);        % system cost: H = 2 diag(b)
fUE = sysCost(xUE, a, b); fSO = sysCost(xSO, a, b);
fprintf('UE x = [%.4f %.4f], cost %.4f (7/6)\n', xUE, fUE);
fprintf('SO x = [%.4f %.4f], cost %.4f (9/8)\n', xSO, fSO);
fprintf('PoA = %.4f (28/27 = %.4f)\n', fUE/fSO, 28/27);

%% Q1(e) PoA against demand
ds = linspace(1e-3, 4, 800);
poa = arrayfun(@(dd) sysCost(parallel_affine(a, b, dd, 'UE'), a, b)/ ...
                     sysCost(parallel_affine(a, b, dd, 'SO'), a, b), ds);
[pm, k] = max(poa);
fprintf('Q1(e) max PoA %.4f at d = %.3f (12/11 at d = 1/2)\n', pm, ds(k));
figure; plot(ds, poa); xlabel('demand d'); ylabel('PoA');

%% Q2(b) as in AssignmentQ2.mlx
a2 = 0.5; b2 = 0.5; d2 = 1000; H = [1 0; 0 b2]; avec = [a2; 1];
x = solve(H, avec, [1 1], d2);
fprintf('Q2(b) UE x = [%.3f %.3f], costs %.2f %.2f\n', x, a2 + x(1), 1 + b2*x(2));
x = solve(2*H, avec, [1 1], d2);
fprintf('Q2(b) SO x = [%.3f %.3f], system cost %.1f\n', x, avec'*x + x'*H*x);

%% Q2(e) and Q3: maximum PoA over d, then over a, b in [0, 1]
[A, B] = meshgrid(linspace(0, 1, 21));
P = zeros(size(A));
for k = 1:numel(A)
    P(k) = max_poa(A(k), B(k));
end
[pm, k] = max(P(:));
fprintf('Q3 max PoA = %.4f at a = %.2f, b = %.2f (theory 4/3)\n', pm, A(k), B(k));
figure; contourf(A, B, P, 20); colorbar; xlabel('a'); ylabel('b'); title('max PoA');

function p = max_poa(a, b)
% PoA(d) for c1 = a + x1, c2 = 1 + b x2: grid search then fminbnd (base MATLAB)
av = [a; 1]; bv = [1; b];
cst = @(x) sum(x.*(av + bv.*x));
negpoa = @(d) -cst(parallel_affine(av, bv, d, 'UE'))/cst(parallel_affine(av, bv, d, 'SO'));
grid = linspace(1e-4, 10*(1 + 1/max(b, 1e-3)), 400);
v = arrayfun(negpoa, grid); [~, k] = min(v);
[~, fv] = fminbnd(negpoa, grid(max(k-1, 1)), grid(min(k+1, end)));
p = -fv;
end

function x = qpwrap(H, f, Aeq, beq, hasQP)
if hasQP
    x = quadprog(H, f, [], [], Aeq, beq, zeros(numel(f), 1), [], [], ...
                 optimoptions('quadprog', 'Display', 'off'));
else
    x = qp_small(H, f, Aeq, beq);
end
end
