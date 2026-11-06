function [flow, D] = edge_flows(G, demand)
%EDGE_FLOWS All-or-nothing assignment of an OD matrix onto shortest paths.
%   flow = edge_flows(G, demand) returns the flow on each edge of G (in the
%   order of G.Edges). G can be a graph (two-way) or digraph (one-way).
%   demand: n x n OD matrix; default all-to-all (1 per ordered pair).
%
%   Trick: shortestpathtree gives pred(v) = previous node on the route s->v.
%   Visit nodes furthest first; each passes its load up to its parent:
%       flow(edge pred(v)->v) += load(v);  load(pred(v)) += load(v)
%   so one Dijkstra per origin is enough (no need to list 10,000 paths).
%   Python equivalent: net_utils.edge_flows
n = numnodes(G);
if nargin < 2, demand = ones(n) - eye(n); end
flow = zeros(numedges(G), 1);
D = inf(n);
for s = find(sum(demand, 2) > 0)'
    [pred, d] = shortestpathtree(G, s, 'OutputForm', 'vector');
    D(s, :) = d;
    load = demand(s, :);
    [~, order] = sort(d, 'descend');
    for v = order
        p = pred(v);
        if p == 0 || load(v) == 0 || isinf(d(v)), continue; end
        e = findedge(G, p, v);
        flow(e) = flow(e) + load(v);
        load(p) = load(p) + load(v);
    end
end
end
