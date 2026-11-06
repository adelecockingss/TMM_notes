%% TMM Lab 02 - Q4: beta-skeleton on a torus (doubly periodic tile)
% Tile the N nodes 3x3, build the skeleton of all 9N points, keep edges that
% touch the central tile, map ends back mod N. Python: q3_q4_beta_skeletons.py
clear; close all;
W = 1.2; N = 144;
betas = 1:0.1:2;
R = nan(numel(betas), 3);
for b = 1:numel(betas)
    vals = [];
    for sd = 1:3
        rng(sd); xy = W*rand(N, 2);
        [G, euclid] = torus_skeleton(xy, W, betas(b));
        [flow, D] = edge_flows(G);
        if any(isinf(D(:))), continue; end
        off = ~eye(N);
        vals(end+1, :) = [sum(G.Edges.Weight), mean(D(off)./euclid(off)), max(flow)/mean(flow)]; %#ok<SAGROW>
    end
    R(b, :) = mean(vals, 1);
end
disp(array2table([betas' R], 'VariableNames', {'beta', 'length', 'detour', 'maxOverMeanFlow'}))

function [G, euclid] = torus_skeleton(xy, W, beta)
N = size(xy, 1);
[dx, dy] = meshgrid(-1:1); shifts = W*[dx(:) dy(:)];   % row 5 = (0,0)
big = zeros(9*N, 2);
for k = 1:9, big((k-1)*N + (1:N), :) = xy + shifts(k,:); end
E = beta_skeleton(big, beta, 'lune');
tile = ceil(E/N);
E = E(any(tile == 5, 2), :);
len = vecnorm(big(E(:,1),:) - big(E(:,2),:), 2, 2);
a = mod(E(:,1) - 1, N) + 1; b = mod(E(:,2) - 1, N) + 1;
[~, first] = unique([min(a,b) max(a,b) round(len, 9)], 'rows');
G = graph(a(first), b(first), len(first), N);
G = simplify(G, 'min');                % (rare) duplicate pairs: keep shortest
d1 = xy(:,1) - xy(:,1)'; d2 = xy(:,2) - xy(:,2)';
d1 = d1 - W*round(d1/W); d2 = d2 - W*round(d2/W);   % minimum image
euclid = hypot(d1, d2);
end
