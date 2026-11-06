%% TMM Lab 02 - Q9-Q11 in MATLAB, from tables prepared in Python
% Run q5_q11_osm.py first: it reads the OSM file and writes
%   data/<city>_junctions.csv  (junction nodes, x_km, y_km)
%   data/<city>_edges.csv      (u, v (0-based), length_km, oneway, highway, name)
% Base MATLAB only.
clear; close all;

%% Q9: build a digraph that respects one-way streets
J = readtable('data/bristol_junctions.csv');
E = readtable('data/bristol_edges.csv', 'TextType', 'string');
u = E.u + 1; v = E.v + 1;                       % Python 0-based -> MATLAB 1-based
fwd = E.oneway ~= "-1";
bwd = ~ismember(E.oneway, ["yes" "true" "1"]);
src = [u(fwd); v(bwd)]; dst = [v(fwd); u(bwd)];
len = [E.length_km(fwd); E.length_km(bwd)];
id  = [find(fwd); find(bwd)];                   % which street each arc belongs to
G = digraph(src, dst, table(len, id, 'VariableNames', {'Weight', 'street'}), height(J));
xy = [J.x_km J.y_km];
fprintf('%d junctions, %d streets, %d directed arcs, %.1f km of street\n', ...
    numnodes(G), height(E), numedges(G), sum(E.length_km));
figure; plot(G, 'XData', xy(:,1), 'YData', xy(:,2), 'NodeLabel', {}, 'MarkerSize', 1, ...
             'ArrowSize', 2, 'EdgeColor', [0.5 0.5 0.5]);
axis equal; title('Bristol junction graph');

%% Q10: shortest path SW -> NE, then close each street on it
bins = conncomp(G, 'Type', 'strong');
big = find(bins == mode(bins));
[~, a] = min(hypot(xy(big,1) + 1.2, xy(big,2) + 1.2)); s = big(a);
[~, b] = min(hypot(xy(big,1) - 1.2, xy(big,2) - 1.2)); t = big(b);
[p0, d0, ep] = shortestpath(G, s, t);
fprintf('route %.2f km, straight line %.2f km\n', d0, norm(xy(t,:) - xy(s,:)));
streets = unique(G.Edges.street(ep));
extra = zeros(numel(streets), 1);
for k = 1:numel(streets)
    G2 = rmedge(G, find(G.Edges.street == streets(k)));   % close both directions
    [~, d] = shortestpath(G2, s, t);
    extra(k) = d - d0;
end
fprintf('%d bridges on the route; worst finite detour +%.0f m\n', ...
    nnz(isinf(extra)), 1000*max(extra(isfinite(extra))));
hold on; highlight(findobj(gca, 'Type', 'GraphPlot'), p0, 'EdgeColor', 'r', 'LineWidth', 2);

%% Q11: beta-skeleton vs real network (precision / recall / F1)
for city = ["bristol" "bath"]
    J = readtable("data/" + city + "_junctions.csv");
    E = readtable("data/" + city + "_edges.csv", 'TextType', 'string');
    xy = [J.x_km J.y_km];
    real = unique(sort([E.u E.v] + 1, 2), 'rows');
    real = real(real(:,1) ~= real(:,2), :);
    betas = 1:0.1:2.6; F1 = zeros(size(betas));
    for k = 1:numel(betas)
        S = beta_skeleton(xy, betas(k), 'lune');
        tp = nnz(ismember(S, real, 'rows'));
        prec = tp/size(S, 1); rec = tp/size(real, 1);
        F1(k) = 2*prec*rec/(prec + rec);
    end
    [Fbest, kb] = max(F1);
    fprintf('%s: best beta = %.1f (F1 %.3f)\n', city, betas(kb), Fbest);
end
