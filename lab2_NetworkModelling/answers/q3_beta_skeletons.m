%% TMM Lab 02 - Q3: beta-skeleton cities (MATLAB version, base MATLAB)
% Uses beta_skeleton.m instead of the File Exchange ProximityGraphs toolbox.
% Python version: q3_q4_beta_skeletons.py
clear; close all; rng(3);

%% Gallery: random nodes (as in BuildBetaSkeletonsExample.mlx)
L = 1; numNode = 150;
xy = L*rand(numNode, 2);
betas = [0.8 1 1.5 2 2.5];
figure('Position', [50 50 1600 650]);
for c = 1:numel(betas)
    for r = 1:2
        kinds = {'lune', 'circle'};
        E = beta_skeleton(xy, betas(c), kinds{r});
        G = graph(E(:,1), E(:,2), vecnorm(xy(E(:,1),:) - xy(E(:,2),:), 2, 2), numNode);
        subplot(2, numel(betas), (r-1)*numel(betas) + c);
        plot(G, 'XData', xy(:,1), 'YData', xy(:,2), 'NodeLabel', {}, 'MarkerSize', 2);
        axis equal off; title(sprintf('%s \beta=%.1f: %d edges', kinds{r}, betas(c), numedges(G)));
    end
end

%% Trade-off sweep: length vs detour vs congestion
betas = 0.8:0.1:2;
nSeeds = 3;
S = zeros(numel(betas), 4, 2);        % running sums of [length/MST, detour, max flow, mean flow]
C = zeros(numel(betas), 1, 2);        % number of connected networks in each sum
for placement = 1:2
    for sd = 1:nSeeds
        rng(sd);
        if placement == 1
            xy = 1.2*rand(144, 2);                                  % random
        else
            [gx, gy] = meshgrid(0:11); xy = 0.1*[gx(:) gy(:)] + 0.06*(rand(144, 2) - 0.5);  % jittered grid
        end
        n = size(xy, 1);
        euclid = hypot(xy(:,1) - xy(:,1)', xy(:,2) - xy(:,2)');
        Lmst = sum(minspantree(graph(euclid)).Edges.Weight);
        for b = 1:numel(betas)
            E = beta_skeleton(xy, betas(b), 'lune');
            G = graph(E(:,1), E(:,2), vecnorm(xy(E(:,1),:) - xy(E(:,2),:), 2, 2), n);
            [flow, D] = edge_flows(G);
            if any(isinf(D(:))), continue; end                    % disconnected
            off = ~eye(n);
            vals = [sum(G.Edges.Weight)/Lmst, mean(D(off)./euclid(off)), max(flow), mean(flow)];
            S(b, :, placement) = S(b, :, placement) + vals;
            C(b, 1, placement) = C(b, 1, placement) + 1;
        end
    end
end
M = S./C;                             % mean over connected networks
titles = {'length / MST', 'mean detour', 'max flow', 'mean flow'};
figure('Position', [50 50 1400 350]);
for k = 1:4
    subplot(1, 4, k); plot(betas, M(:,k,1), 'o-', betas, M(:,k,2), 's-');
    xlabel('\beta'); title(titles{k});
end
legend('random', 'jittered grid');
disp(array2table([betas' M(:,:,1)], 'VariableNames', [{'beta'}, titles]))
