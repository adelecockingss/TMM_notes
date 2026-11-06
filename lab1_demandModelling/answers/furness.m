function [a, b, T, iters] = furness(O, D, F, tol, maxIter)
%FURNESS Doubly constrained balancing factors by iterative proportional fitting.
%   Finds a, b so that T = (a.*O) .* F .* (b.*D)' has row sums O, column sums D.
%     a_i = 1 / sum_j b_j D_j F_ij ,   b_j = 1 / sum_i a_i O_i F_ij
%   Alternate the two updates until the row sums are right (the column sums
%   are exactly right after every b update). O, D column vectors; F = f(dist).
%   Python equivalent: gravity_utils.furness
if nargin < 4, tol = 1e-8; end
if nargin < 5, maxIter = 1000; end
b = ones(size(D));
for iters = 1:maxIter
    a = 1./max(F*(b.*D), realmin);
    b = 1./max(F'*(a.*O), realmin);
    T = (a.*O).*F.*(b.*D)';
    if max(abs(sum(T, 2) - O))/max(O) < tol, break; end
end
end
