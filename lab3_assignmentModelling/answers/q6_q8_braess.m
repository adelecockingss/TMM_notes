%% TMM Lab 03 - Q6-Q8: Braess network in route variables (MATLAB version)
% Links 1 O->N, 2 N->D, 3 O->S, 4 S->D, 5 N->S. Routes N {1,2}, S {3,4}, M {1,5,4}.
% Uses quadprog if available, else qp_small. Python: q6_q8_braess.py
clear; close all;

%% Q6(a) lecture network at d = 1
a = [0 1 1 0 0]'; b = [1 0 0 1 0]';
report('Q6(a)', a, b, 1);

%% Q6(b) paradox range in d
ds = linspace(0.05, 3, 300);
w = arrayfun(@(d) braess(a, b, d, 'UE', true)/d, ds);
wo = arrayfun(@(d) braess(a, b, d, 'UE', false)/d, ds);
p = ds(w > wo + 1e-9);
fprintf('Q6(b) paradox for %.2f < d < %.2f (2/3 to 2)\n', min(p), max(p));
figure; plot(ds, w, ds, wo); legend('with midtown', 'without'); xlabel('d'); ylabel('cost per user');

%% Q6(c) swapped costs: no paradox
ac = [1 0 0 1 0]'; bc = [0 1 1 0 0]';
diffc = arrayfun(@(d) braess(ac, bc, d, 'UE', true) - braess(ac, bc, d, 'UE', false), ds);
fprintf('Q6(c) max(with - without) = %.1e\n', max(diffc));

%% Q7 Sheffi
report('Q7 Sheffi', [0 50 50 0 10]', [10 1 1 10 1]', 6);

%% Q8 check of the analytic UE cost at a few points
for d = [0.5 1 2]
    err = 0;
    for aa = linspace(0, 1, 6)
        for bb = linspace(0, 1, 6)
            num = braess([aa 1 1 aa 0]', [1 bb bb 1 0]', d, 'UE', true)/d;
            C0 = aa + 1 + d*(1 + bb)/2;
            if aa + d <= 1, C1 = 2*(aa + d);
            else
                y = (aa + d - 1)/(1 + bb);
                if y < d/2, C1 = 2*(aa + d - y); else, C1 = C0; end
            end
            err = max(err, abs(num - C1));
        end
    end
    fprintf('Q8 d = %.1f: max |numeric - analytic| = %.1e\n', d, err);
end

function [f, y, x] = braess(a, b, d, mode, midtown)
A = [1 0 1; 1 0 0; 0 1 0; 0 1 1; 0 0 1];
R = 1:3; if ~midtown, R = 1:2; end
Ar = A(:, R);
H = Ar'*diag(b)*Ar; if strcmp(mode, 'SO'), H = 2*H; end
if exist('quadprog', 'file') == 2
    yr = quadprog(H, Ar'*a, [], [], ones(1, numel(R)), d, zeros(numel(R), 1), [], [], ...
                  optimoptions('quadprog', 'Display', 'off'));
else
    yr = qp_small(H, Ar'*a, ones(1, numel(R)), d);
end
y = zeros(3, 1); y(R) = yr; x = A*y;
f = sum(x.*(a + b.*x));                       % system cost
end

function report(name, a, b, d)
fprintf('--- %s, d = %g\n', name, d);
A = [1 0 1; 1 0 0; 0 1 0; 0 1 1; 0 0 1];
for c = {{'UE', true}, {'SO', true}, {'UE', false}}
    [f, y, x] = braess(a, b, d, c{1}{1}, c{1}{2});
    fprintf('  %s midtown=%d: y = %s, route costs %s, system cost %.3f\n', c{1}{1}, ...
        c{1}{2}, mat2str(y', 3), mat2str((A'*(a + b.*x))', 4), f);
end
end
