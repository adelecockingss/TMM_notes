%% TMM Lab 03 - Q10-Q13: assignment on bigger networks (MATLAB version)
% With the Optimization Toolbox you can use the link-based quadprog
% formulation from BigNetworkExperiments.mlx (shown at the bottom, untested
% here). Without it: gradient_projection.m (exact, single OD) for Q10 and
% frank_wolfe.m (any OD matrix) for Q13.
clear; close all;

%% Check FW on the Q9 network (exact UE 0.5 0.5 1 0.5 0.5)
arcs9 = [1 3; 2 3; 3 4; 1 4; 2 4];
dem9 = zeros(4); dem9(1, 4) = 1; dem9(2, 4) = 1;
x9 = frank_wolfe(4, arcs9, [2 2 1 4 4], ones(5, 1), dem9, 'UE', 1e-10);
fprintf('FW on Q9: %s\n', mat2str(x9', 4));

%% Q10 beta-skeleton network: every street as TWO arcs
S = load('../BigNetworkExperimentsBetaSkeleton (1).mat');
G = S.G; xy = S.nodeCoordinates; n = numnodes(G);
E = G.Edges.EndNodes; len = G.Edges.Weight;
arcs = [E; E(:, [2 1])]; a = [len; len]; b = ones(size(a));
o = double(S.oNode); dn = double(S.dNode);
cost = @(x) sum(x.*(a + b.*x));
for d = [0.001 1]
    [xu, Ru] = gradient_projection(n, arcs, a, b, o, dn, d, 'UE');
    xs = gradient_projection(n, arcs, a, b, o, dn, d, 'SO');
    fprintf('d = %g: UE uses %d routes, cost %.5g; SO cost %.5g; PoA %.4f\n', d, ...
        size(Ru, 2), cost(xu), cost(xs), cost(xu)/cost(xs));
end

%% Q10(c) flows as demand increases
figure('Position', [50 50 1500 350]);
ds = [0.01 0.3 1 3 10];
for k = 1:numel(ds)
    x = gradient_projection(n, arcs, a, b, o, dn, ds(k), 'UE');
    f = x(1:numedges(G)) + x(numedges(G)+1:end);       % both directions
    subplot(1, numel(ds), k);
    plot(G, 'XData', xy(:,1), 'YData', xy(:,2), 'NodeLabel', {}, 'MarkerSize', 1, ...
         'LineWidth', 0.3 + 4*f/max(f), 'EdgeCData', f);
    colormap(flipud(hot)); axis equal off; title(sprintf('UE d = %g', ds(k)));
end

%% Q10(d) with UE costs as weights, every used arc is on a shortest path
d = 3;
[xu, routes] = gradient_projection(n, arcs, a, b, o, dn, d, 'UE');
c = a + b.*xu;
D = digraph(arcs(:,1), arcs(:,2), c, n);
pi = distances(D, o);
slack = pi(arcs(:,1))' + c - pi(arcs(:,2))';
used = xu > 1e-6*d;
fprintf('Q10(d): %d used routes; max slack on used arcs %.1e (0 = on a shortest path)\n', ...
        size(routes, 2), max(slack(used)));

%% Q13 size of the 'coloured' multi-OD QP vs Frank-Wolfe
n6 = 6; G6 = graph(-0.1*delsq(numgrid('S', n6+2)), 'omitselfloops');
E6 = G6.Edges.EndNodes; arcs6 = [E6; E6(:, [2 1])];
a6 = 0.1*ones(size(arcs6, 1), 1); b6 = 0.01*ones(size(a6));
dem6 = ones(n6^2) - eye(n6^2);
fprintf('Q13: stacked QP would have %d variables, %d constraints\n', n6^2*size(arcs6, 1), n6^4);
xu6 = frank_wolfe(n6^2, arcs6, a6, b6, dem6, 'UE', 1e-5, 5000);
xs6 = frank_wolfe(n6^2, arcs6, a6, b6, dem6, 'SO', 1e-5, 5000);
c6 = @(x) sum(x.*(a6 + b6.*x));
fprintf('   UE %.2f, SO %.2f, PoA %.4f\n', c6(xu6), c6(xs6), c6(xu6)/c6(xs6));
% PoA = 1: all arcs have the same free-flow cost and only Manhattan-shortest
% routes are used, so every used route has the same sum(a). Then 'equal
% sum(b x)' (UE) and 'equal sum(2 b x)' (SO) are the same condition.

%% (For reference) the quadprog version - needs the Optimization Toolbox
if exist('quadprog', 'file') == 2
    m = size(arcs, 1);
    B = sparse([arcs(:,1); arcs(:,2)], [1:m 1:m]', [ones(m,1); -ones(m,1)], n, m);
    s = zeros(n, 1); s(o) = 1; s(dn) = -1;
    xq = quadprog(diag(b), a, [], [], B, 3*s, zeros(m, 1), []);
    fprintf('quadprog vs FW max difference: %.1e\n', max(abs(xq - xu)));
end
