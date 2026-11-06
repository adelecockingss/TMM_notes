%% TMM Lab 02 - Q2: perturbing the square grid (MATLAB version)
% Base MATLAB. Python version: q2_perturbed_grid.py
clear; close all; rng(1);
n = 10; L = 0.1; N = n^2;
[G, xy] = grid_network(n, L);
[s, t] = findedge(G);

%% Q2(a) random weights U(0, 2L): shortest paths become unique
Ga = G; Ga.Edges.Weight = 2*L*rand(numedges(G), 1);
figure; plot_flows(Ga, xy, edge_flows(Ga), 'Q2(a) random weights');

%% Q2(b) jittered nodes; edge weights = new lengths (new A matrix)
xyb = xy + L*(rand(N, 2) - 0.5);
len = vecnorm(xyb(s,:) - xyb(t,:), 2, 2);
A = sparse([s; t], [t; s], [len; len], N, N);     % the new weighted adjacency
Gb = graph(A);
fb = edge_flows(Gb);
figure; plot_flows(Gb, xyb, fb, 'Q2(b) jittered nodes');

%% Q2(c) closing streets: protect a minimum spanning tree
T = minspantree(Gb);
inT = ismember(Gb.Edges.EndNodes, T.Edges.EndNodes, 'rows');
fprintf('%d streets, MST %d: up to %d can close\n', numedges(Gb), numedges(T), nnz(~inT));
cand = find(~inT); cand = cand(randperm(numel(cand)));
ks = 0:5:numel(cand); meanD = zeros(size(ks));
for q = 1:numel(ks)
    Gc = rmedge(Gb, cand(1:ks(q)));
    Dq = distances(Gc); meanD(q) = mean(Dq(~eye(N)));
end
figure; plot(ks, meanD, 'o-'); xlabel('streets closed'); ylabel('mean shortest path (km)');

%% Q2(c,d) one-way streets: greedy, keep if still strongly connected
E = Gb.Edges.EndNodes; w = Gb.Edges.Weight; m = size(E, 1);
dirE = E; oneway = false(m, 1);
for k = randperm(m)
    trial = dirE; if rand < 0.5, trial(k,:) = trial(k, [2 1]); end
    ow = oneway; ow(k) = true;
    arcs = [trial(ow,:); E(~ow,:); E(~ow,[2 1])];
    D = digraph(arcs(:,1), arcs(:,2), [w(ow); w(~ow); w(~ow)], N);
    if max(conncomp(D, 'Type', 'strong')) == 1     % strongly connected?
        oneway = ow; dirE = trial;
    end
end
fprintf('one-way streets possible: %d of %d\n', nnz(oneway), m);
figure; hold on; axis equal;
for k = find(~oneway)'
    plot(xyb(E(k,:),1), xyb(E(k,:),2), 'Color', [0.5 0.5 0.5], 'LineWidth', 2.5);
end
p = xyb(dirE(oneway,1),:); d = xyb(dirE(oneway,2),:) - p;
quiver(p(:,1), p(:,2), d(:,1), d(:,2), 0, 'r');
title('Q2(d) grey = two-way, red = one-way');

%% Q2(e) gravity demand T_ij = P_i P_j / d_ij^2
P = randi([50 500], N, 1);
Dst = hypot(xyb(:,1) - xyb(:,1)', xyb(:,2) - xyb(:,2)'); Dst(1:N+1:end) = inf;
Tg = (P*P')./Dst.^2; Tg = Tg*(N*(N-1))/sum(Tg(:));
figure; plot_flows(Gb, xyb, edge_flows(Gb, Tg), 'Q2(e) gravity demand');
