function [y, sptt] = aon_flows(n, arcs, cost, demand)
%AON_FLOWS All-or-nothing assignment on shortest paths (directed arcs).
%   arcs: m x 2 (u -> v); a two-way street must appear as two arcs.
%   Returns arc flows y and sptt = sum over OD pairs of demand x shortest cost.
%   Uses a shortest-path tree per origin and accumulates demand from the
%   leaves back to the origin. Python equivalent: assign_utils.aon
m = size(arcs, 1);
G = digraph(arcs(:,1), arcs(:,2), table(cost(:) + 1e-15, (1:m)', ...
            'VariableNames', {'Weight', 'id'}), n);
y = zeros(m, 1); sptt = 0;
for s = find(sum(demand, 2) > 0)'
    [pred, d] = shortestpathtree(G, s, 'OutputForm', 'vector');
    load = demand(s, :);
    ok = isfinite(d);
    sptt = sptt + sum(load(ok).*d(ok));
    [~, order] = sort(d, 'descend');
    for v = order
        p = pred(v);
        if p == 0 || load(v) == 0 || ~isfinite(d(v)), continue; end
        e = G.Edges.id(findedge(G, p, v));
        y(e) = y(e) + load(v);
        load(p) = load(p) + load(v);
    end
end
end
