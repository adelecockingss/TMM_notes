function [x, routes, y] = gradient_projection(n, arcs, a, b, o, dn, d, mode, tol, maxIter)
%GRADIENT_PROJECTION Exact single-OD UE/SO with affine costs, path-based.
%   Keep a set of routes (logical arc-incidence columns) with flows y.
%   Repeat: costs c = a + k b x (k = 1 UE, 2 SO); add the current shortest
%   route if new; shift flow from each dearer used route p to the cheapest q
%   by the exact affine step  delta = (c_p - c_q)/(k * sum b over links in
%   exactly one of p, q), capped at y_p. Stop when all used routes cost the
%   same as the shortest route (Wardrop). Base MATLAB.
%   Python equivalent: assign_utils.gradient_projection
if nargin < 9, tol = 1e-12; end
if nargin < 10, maxIter = 10000; end
a = a(:); b = b(:); m = size(arcs, 1);
k = 1; if strcmp(mode, 'SO'), k = 2; end
R = shortest_route(a);
routes = R; y = d; x = d*double(R);
for it = 1:maxIter
    c = a + k*b.*x;
    [R, cmin] = shortest_route(c);
    if ~any(all(routes == R, 1)), routes(:, end+1) = R; y(end+1) = 0; end %#ok<AGROW>
    cr = c'*routes;
    used = y > 0;
    if max(cr(used)) - cmin <= tol*max(cmin, 1), break; end
    [~, q] = min(cr);
    for p = find(used)
        if p == q, continue; end
        c = a + k*b.*x;
        gap = c'*routes(:, p) - c'*routes(:, q);
        sym = xor(routes(:, p), routes(:, q));
        curv = k*sum(b(sym));
        if curv > 0, delta = min(y(p), gap/curv); else, delta = y(p); end
        if delta > 0
            y(p) = y(p) - delta; y(q) = y(q) + delta;
            x = x + delta*(double(routes(:, q)) - double(routes(:, p)));
        end
    end
end
keep = y > 1e-14*d;
routes = routes(:, keep); y = y(keep);

    function [r, dist] = shortest_route(c)
        G = digraph(arcs(:,1), arcs(:,2), table(c + 1e-15, (1:m)', 'VariableNames', {'Weight', 'id'}), n);
        [~, dist, ep] = shortestpath(G, o, dn);
        r = false(m, 1); r(G.Edges.id(ep)) = true;
    end
end
