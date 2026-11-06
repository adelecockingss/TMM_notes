function h = plot_flows(G, xy, flow, ttl)
%PLOT_FLOWS 'Hotness' plot: edge colour and width proportional to flow.
h = plot(G, 'XData', xy(:,1), 'YData', xy(:,2), 'NodeLabel', {}, 'MarkerSize', 2, ...
         'EdgeCData', flow, 'LineWidth', 0.5 + 5*flow/max(flow));
colormap(flipud(hot)); colorbar; axis equal;
if nargin > 3, title(ttl); end
end
