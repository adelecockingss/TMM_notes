function D = haversine_matrix(lat1, lon1, lat2, lon2)
%HAVERSINE_MATRIX Great-circle distances (km) between two sets of points.
%   D = haversine_matrix(lat1, lon1, lat2, lon2) returns an n1-by-n2 matrix.
%   D = haversine_matrix(lat, lon) returns the symmetric n-by-n matrix.
%   Inputs in degrees; any shape (they are turned into vectors).
%
%   a = sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2)
%   d = 2 R asin(sqrt(a))
%
%   Python equivalent: geo_utils.haversine_matrix

if nargin < 3
    lat2 = lat1; lon2 = lon1;
end
R = 6371;                         % mean Earth radius, km

% Column vs row vectors: MATLAB's implicit expansion then makes a full
% matrix, e.g. (n1x1) - (1xn2) is n1xn2. No loops needed.
p1 = deg2rad(lat1(:));  l1 = deg2rad(lon1(:));
p2 = deg2rad(lat2(:)'); l2 = deg2rad(lon2(:)');

a = sin((p2 - p1)/2).^2 + cos(p1).*cos(p2).*sin((l2 - l1)/2).^2;
D = 2*R*asin(sqrt(min(max(a, 0), 1)));
end
