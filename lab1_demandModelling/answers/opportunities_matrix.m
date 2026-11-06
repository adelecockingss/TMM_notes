function s = opportunities_matrix(D, dist)
%OPPORTUNITIES_MATRIX s(i,j) = jobs in zones closer to i than j is (excl. i and j).
%   For each origin: sort by distance, then a cumulative sum of D.
%   Python equivalent: gravity_utils.opportunities_matrix
n = numel(D);
s = zeros(n);
for i = 1:n
    [~, order] = sort(dist(i, :));
    Ds = D(order);
    before = cumsum(Ds) - Ds;          % everything earlier in the order
    row = zeros(1, n);
    row(order) = before - D(i);        % exclude the origin zone
    row(i) = 0;
    s(i, :) = row;
end
end
