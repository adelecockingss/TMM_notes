function [x, gap, it] = frank_wolfe(n, arcs, a, b, demand, mode, tol, maxIter)
%FRANK_WOLFE UE or SO assignment with affine link costs a + b x.
%   1. costs c = a + b x (UE) or marginal a + 2 b x (SO)
%   2. all-or-nothing load y on shortest paths for those costs
%   3. x <- x + lam (y - x), lam in [0,1] minimising the (quadratic) objective
%   Stop when relative gap (c'x - c'y)/c'x < tol. Works for any number of OD
%   pairs with no 'colouring'. Base MATLAB alternative to quadprog.
%   Python equivalent: assign_utils.frank_wolfe
if nargin < 7, tol = 1e-6; end
if nargin < 8, maxIter = 5000; end
a = a(:); b = b(:);
k = 1; if strcmp(mode, 'SO'), k = 2; end
x = aon_flows(n, arcs, a, demand);
for it = 1:maxIter
    c = a + k*b.*x;
    [y, sptt] = aon_flows(n, arcs, c, demand);
    gap = (c'*x - sptt)/max(c'*x, eps);
    if gap < tol, break; end
    dvec = y - x;
    curv = k*sum(b.*dvec.^2);
    lam = 1; if curv > 0, lam = min(max(-(c'*dvec)/curv, 0), 1); end
    x = x + lam*dvec;
end
end
