function z = load_od_level(level, radius_km)
%LOAD_OD_LEVEL Dense OD data for Q4-Q8. Run q3_od_matrix.m first.
%   z = load_od_level('bristol')  MSOAs within 50 km of Bristol, internal flows only
%   z = load_od_level('utla')     174 Upper Tier Local Authorities
%   z = load_od_level('all')      all 7264 MSOAs (needs ~3 GB RAM)
%   Returns struct with T (n x n), O, D (column vectors), dist (km, with
%   intra-zonal distance = half the distance to the nearest neighbour), labels.
%   Python equivalent: gravity_utils.load_level
if nargin < 2, radius_km = 50; end
if strcmp(level, 'utla')
    S = load('data/od_utla.mat');
    T = S.TU; dist = S.distU; labels = S.utlaLabels; E = S.EU; N = S.NU;
else
    S = load('data/od_msoa.mat');
    keep = true(size(S.E));
    if strcmp(level, 'bristol')
        keep = hypot(S.E - 358800, S.N - 173000) <= radius_km*1000;
    end
    T = full(S.T(keep, keep));
    E = S.E(keep); N = S.N(keep); labels = S.labels(keep);
    dist = hypot(E - E', N - N')/1000;
end
% Intra-zonal distance: 0 would make 1/d^2 infinite
d = dist; d(1:size(d,1)+1:end) = inf;
dist(1:size(d,1)+1:end) = 0.5*min(d, [], 2);
z = struct('T', T, 'O', sum(T, 2), 'D', sum(T, 1)', 'dist', dist, ...
           'labels', labels, 'E', E, 'N', N);
end
