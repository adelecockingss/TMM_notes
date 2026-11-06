function E = beta_skeleton(xy, beta, kind)
%BETA_SKELETON Edge list (m x 2) of the lune- or circle-based beta-skeleton.
%   Edge pq exists iff no other point is strictly inside the forbidden region:
%     beta = 1: disc on diameter pq (Gabriel graph)
%     beta = 2, 'lune': Relative Neighbourhood Graph
%     beta > 1: bigger region -> fewer edges;  beta < 1: thin lens -> more edges
%   For beta >= 1 only Delaunay edges need testing (skeleton is a subgraph).
%   Base MATLAB (no ProximityGraphs toolbox, whose beta convention differs).
%   Python equivalent: net_utils.beta_skeleton
if nargin < 3, kind = 'lune'; end
n = size(xy, 1);
if beta >= 1
    T = delaunay(xy(:,1), xy(:,2));
    cand = unique(sort([T(:,[1 2]); T(:,[2 3]); T(:,[1 3])], 2), 'rows');
else
    [i, j] = find(triu(true(n), 1)); cand = [i j];
end
keep = true(size(cand, 1), 1);
for k = 1:size(cand, 1)
    a = cand(k,1); b = cand(k,2);
    p = xy(a,:); q = xy(b,:); d = norm(q - p);
    m = (p + q)/2; nrm = [-(q(2)-p(2)), q(1)-p(1)]/d;
    if beta >= 1
        R = beta*d/2;
        if strcmp(kind, 'lune')
            c1 = (1 - beta/2)*p + beta/2*q;  c2 = beta/2*p + (1 - beta/2)*q;
            in = vecnorm(xy - c1, 2, 2) < R - 1e-12 & vecnorm(xy - c2, 2, 2) < R - 1e-12;
        else
            h = d/2*sqrt(beta^2 - 1);
            in = vecnorm(xy - (m + h*nrm), 2, 2) < R - 1e-12 | vecnorm(xy - (m - h*nrm), 2, 2) < R - 1e-12;
        end
    else
        R = d/(2*beta); h = sqrt(max(R^2 - (d/2)^2, 0));
        in = vecnorm(xy - (m + h*nrm), 2, 2) < R - 1e-12 & vecnorm(xy - (m - h*nrm), 2, 2) < R - 1e-12;
    end
    in([a b]) = false;
    keep(k) = ~any(in);
end
E = cand(keep, :);
end
