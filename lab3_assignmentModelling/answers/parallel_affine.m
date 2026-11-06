function x = parallel_affine(a, b, d, mode)
%PARALLEL_AFFINE Exact UE or SO flows on parallel links with costs a_i + b_i x_i.
%   UE: used links share a common cost C, x_i(C) = max(0, (C - a_i)/b_i),
%   choose C so that sum x_i = d ('water-filling'). SO: same with marginal
%   cost a_i + 2 b_i x_i, i.e. b -> 2b. Links with b_i = 0 absorb any flow at a_i.
%   Python equivalent: assign_utils.parallel_affine
a = a(:); b = b(:);
if strcmp(mode, 'SO'), b = 2*b; end
x = zeros(size(a));
var = b > 0; const = ~var;
S = @(C) sum(max(0, (C - a(var))./b(var)));
Cc = inf; if any(const), Cc = min(a(const)); end
if ~any(var) || S(Cc) < d
    C = Cc;
    x(var) = max(0, (C - a(var))./b(var));
    k = find(const & a == Cc, 1);
    x(k) = d - sum(x(var));
else
    [av, ord] = sort(a(var)); bv = b(var); bv = bv(ord);
    for k = 1:numel(av)              % use the k cheapest links
        C = (d + sum(av(1:k)./bv(1:k)))/sum(1./bv(1:k));
        if k == numel(av) || C <= av(k+1), break; end
    end
    x(var) = max(0, (C - a(var))./b(var));
end
end
