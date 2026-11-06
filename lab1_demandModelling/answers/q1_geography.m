%% TMM Lab 01 - Q1: Getting started with geography (MATLAB version)
% Run section by section (Ctrl+Enter) from the answers/ folder.
% Base MATLAB only (no toolboxes). Python version: q1_geography.py
clear; close all;
if ~exist('figures', 'dir'), mkdir figures; end

msoa = readtable('../UK_census_msoa_stats.csv', 'VariableNamingRule', 'preserve');
lat = msoa.Latitude;  lon = msoa.Longitude;

%% Q1(a) Haversine distances (see haversine_matrix.m)
names = ["London" "Manchester" "Bristol" "Cardiff"];
plat = [51.5154 53.4808 51.4545 51.4816];
plon = [-0.0957 -2.2426 -2.5879 -3.1791];
D = haversine_matrix(plat, plon);
disp(array2table(round(D, 1), 'VariableNames', names, 'RowNames', names))

% Distance from every MSOA to Bristol (7201 x 1)
d_bristol = haversine_matrix(lat, lon, 51.4545, -2.5879);
[dmax, imax] = max(d_bristol);
fprintf('Furthest MSOA from Bristol: %s (%.0f km)\n', msoa.("MSOA Name"){imax}, dmax);

%% Q1(b) lat/long -> OS grid (see latlon_to_osgb.m)
[E, N] = latlon_to_osgb(lat, lon);
gridref = compose('%06d %06d', round(E), round(N));
disp(table(msoa.("MSOA Name")(1:5), gridref(1:5), 'VariableNames', {'MSOA', 'gridref'}))

% On the grid, distance is just Pythagoras - compare with Haversine
[e4, n4] = latlon_to_osgb(plat, plon);
D_grid = hypot(e4' - e4, n4' - n4)/1000;
disp('Grid vs Haversine (% difference):'); disp(round(100*(D_grid - D)./max(D, 1), 2))

%% Q1(c) Plot over a map
% geoscatter/geobasemap are in base MATLAB but download map tiles from the
% internet. The plain scatter in grid coordinates always works.
figure;
scatter(E/1000, N/1000, 2, msoa.("Townsend deprivation score"), 'filled');
axis equal; colormap(flipud(parula)); colorbar;
xlabel('Easting (km)'); ylabel('Northing (km)');
title('MSOA centroids, coloured by deprivation');
exportgraphics(gcf, 'figures/q1c_msoa_map_matlab.png');

try
    figure; geoscatter(lat, lon, 2, msoa.("Townsend deprivation score"), 'filled');
    geobasemap streets-light;                  % needs internet
    title('MSOA centroids on a basemap');
catch err
    fprintf('geoscatter basemap unavailable: %s\n', err.message);
end

%% Q1(d) Voronoi tessellation and adjacency matrix
% Use grid metres, not degrees (degrees of longitude are "shorter").
figure; voronoi(E, N);
axis equal; xlim([345e3 375e3]); ylim([160e3 190e3]);
title('Voronoi cells around Bristol');
exportgraphics(gcf, 'figures/q1d_voronoi_matlab.png');

% Neighbouring Voronoi cells <=> edges of the Delaunay triangulation
T = delaunay(E, N);                       % ntri x 3 node indices
i = [T(:,1); T(:,2); T(:,3)];
j = [T(:,2); T(:,3); T(:,1)];
keep = hypot(E(i) - E(j), N(i) - N(j)) <= 20e3;   % drop links across the sea
n = numel(E);
A = sparse(i(keep), j(keep), 1, n, n);
A = double((A + A') > 0);                 % symmetric 0/1
deg = full(sum(A, 2));
fprintf('%d neighbour pairs; neighbours per cell mean %.2f, max %d\n', nnz(A)/2, mean(deg), max(deg));

% The adjacency matrix can go straight into MATLAB's graph object
G = graph(A);
figure; plot(G, 'XData', E, 'YData', N, 'NodeLabel', {}, 'MarkerSize', 2);
axis equal; xlim([345e3 375e3]); ylim([160e3 190e3]);
title('Neighbour graph around Bristol');

%% Q1(e) Shapefiles (needs the Mapping Toolbox + a downloaded shapefile)
% Download MSOA (Dec 2021) boundaries from https://geoportal.statistics.gov.uk/
% into data/msoa_boundaries/, then:
%   S = shaperead('data/msoa_boundaries/<file>.shp');   % Mapping Toolbox
%   mapshow(S);
% or (R2021b+): T = readgeotable('data/msoa_boundaries/<file>.shp'); geoplot(T)
