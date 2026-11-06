%% TMM Lab 02 - Q1: square grid networks with MATLAB's graph object
% Base MATLAB. Python version: q1_grid.py
clear; close all;
if ~exist('figures', 'dir'), mkdir figures; end

n = 10; L = 0.1;
A = -L*delsq(numgrid('S', n+2));
G = graph(A, 'omitselfloops');
h = plot(G);
h.XData = L*floor((0:n^2-1)/n);
h.YData = L*rem((0:n^2-1), n);
axis equal
xy = [h.XData' h.YData'];
G.Edges(1:5, :)                     % weights are all L = 0.1

%% Q1(a) neighbours and degree
neighbors(G, 1)'                    % corner: 2 neighbours
G.neighbors(56)'                    % interior: 4
deg = degree(G);
fprintf('degree 2: %d, 3: %d, 4: %d\n', sum(deg==2), sum(deg==3), sum(deg==4));

%% Q1(b) shortest path 1 -> 100
[path, d, edgepath] = shortestpath(G, 1, 100);
fprintf('path length %.1f km via %d edges\n', d, numel(edgepath));
G.Edges(edgepath(1), :)             % an edge is identified by its row in G.Edges
highlight(h, path, 'EdgeColor', 'r', 'LineWidth', 3);
title('Q1(b) one shortest path 1 \rightarrow 100');
exportgraphics(gcf, 'figures/q1b_shortest_path_matlab.png');

%% Q1(c) all 10,000 shortest paths: (1) double loop, (2) tree accumulation
tic;
count = zeros(numedges(G), 1);
for s = 1:n^2
    for t = 1:n^2
        if s ~= t
            [~, ~, ep] = shortestpath(G, s, t);
            count(ep) = count(ep) + 1;
        end
    end
end
fprintf('double loop: %.2f s\n', toc);

tic; flow = edge_flows(G); fprintf('edge_flows (one tree per origin): %.2f s\n', toc);
Dmat = distances(G);
fprintf('sum f*l = %.1f, sum of all distances = %.1f\n', sum(flow.*G.Edges.Weight), sum(Dmat(:)));

%% Q1(d) hotness plot
figure; plot_flows(G, xy, count, 'Q1(d) edge use, all-to-all demand');
exportgraphics(gcf, 'figures/q1d_hotness_matlab.png');

%% Q1(e) LTN: two cul-de-sac halves, each entered from one side only
ltn = @(x, y) x >= 3 & x <= 6 & y >= 3 & y <= 6;   % node coords in blocks
[s, t] = findedge(G);
xs = [floor((s-1)/n) floor((t-1)/n)]; ys = [rem(s-1, n) rem(t-1, n)];
inLTN = ltn(xs(:,1), ys(:,1)) | ltn(xs(:,2), ys(:,2));
xset = sort(xs, 2); yset = sort(ys, 2);
filt = inLTN & ( all(xset == [4 5], 2) | all(xset == [2 3], 2) | all(xset == [6 7], 2) ...
       | (all(yset == [6 7], 2) & all(xset <= 4, 2)) | (all(yset == [2 3], 2) & all(xset >= 5, 2)) );
G2 = rmedge(G, find(filt));
outside = ~ltn(floor((0:n^2-1)'/n), rem((0:n^2-1)', n));
demand = double(outside & outside'); demand(1:n^2+1:end) = 0;   % through trips only
f1 = edge_flows(G, demand); f2 = edge_flows(G2, demand);
[s2, t2] = findedge(G2);
in2 = ltn(floor((s2-1)/n), rem(s2-1, n)) & ltn(floor((t2-1)/n), rem(t2-1, n));
in1 = ltn(xs(:,1), ys(:,1)) & ltn(xs(:,2), ys(:,2));
fprintf('LTN: %d filters; through traffic inside %d -> %d; total distance +%.1f%%\n', ...
    nnz(filt), sum(f1(in1)), sum(f2(in2)), 100*(sum(distances(G2), 'all')/sum(Dmat(:)) - 1));
figure; plot_flows(G2, xy, f2, 'through traffic with LTN');
