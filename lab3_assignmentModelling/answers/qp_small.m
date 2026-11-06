function [x, fval] = qp_small(H, f, Aeq, beq)
%QP_SMALL min 1/2 x'Hx + f'x subject to Aeq x = beq, x >= 0 (H pos. semi-definite)
%   Base-MATLAB stand-in for quadprog(H, f, [], [], Aeq, beq, zeros(n,1), [])
%   on SMALL problems. It is the complementarity method, automated: guess the
%   set S of used variables (x_i > 0), solve the linear KKT system
%       [H_SS -Aeq_S'; Aeq_S 0] [x_S; mu] = [-f_S; beq]
%   and accept if x_S >= 0 and the unused variables have nu = Hx+f-Aeq'mu >= 0.
%   Tries all 2^n supports, smallest first - fine for n up to ~15.
%   Python equivalent: assign_utils.qp_small
f = f(:); beq = beq(:); n = numel(f); m = numel(beq);
x = []; fval = inf;
for k = 1:n
    C = nchoosek(1:n, k);
    for r = 1:size(C, 1)
        S = C(r, :);
        K = [H(S,S), -Aeq(:,S)'; Aeq(:,S), zeros(m)];
        rhs = [-f(S); beq];
        sol = lsqminnorm(K, rhs);                 % min-norm if singular
        if norm(K*sol - rhs) > 1e-7*max(1, norm(rhs)), continue; end
        xt = zeros(n, 1); xt(S) = sol(1:k); mu = sol(k+1:end);
        nu = H*xt + f - Aeq'*mu;
        if min(xt) >= -1e-9 && min(nu) >= -1e-7
            v = 0.5*xt'*H*xt + f'*xt;
            if v < fval - 1e-12, x = max(xt, 0); fval = v; end
        end
    end
    if ~isempty(x), return; end
end
error('qp_small: no KKT point found');
end
