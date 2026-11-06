function [E, N] = latlon_to_osgb(lat, lon)
%LATLON_TO_OSGB WGS84 lat/long (degrees) -> OS National Grid easting/northing (m).
%   Vectorised: lat and lon can be arrays of any (matching) size.
%   Step 1: Helmert datum shift WGS84 -> OSGB36 (via 3-D Cartesian coords).
%   Step 2: Transverse Mercator projection on the Airy 1830 ellipsoid.
%   Formulas: OS "A guide to coordinate systems in Great Britain".
%   Accurate to ~5 m (the Helmert step). 12-digit grid ref = sprintf('%06d %06d', E, N).
%
%   Python equivalent: geo_utils.latlon_to_osgb

lat = deg2rad(lat); lon = deg2rad(lon);

% ---- Step 1: datum shift --------------------------------------------------
[x, y, z] = ll2xyz(lat, lon, 6378137.000, 6356752.3141);       % WGS84
tx = -446.448; ty = 125.157; tz = -542.060;                       % metres
rx = deg2rad(-0.1502/3600); ry = deg2rad(-0.2470/3600); rz = deg2rad(-0.8421/3600);
s  = 20.4894e-6;
x2 = tx + (1+s)*x - rz*y + ry*z;
y2 = ty + rz*x + (1+s)*y - rx*z;
z2 = tz - ry*x + rx*y + (1+s)*z;
a = 6377563.396; b = 6356256.909;                                 % Airy 1830
[lat, lon] = xyz2ll(x2, y2, z2, a, b);

% ---- Step 2: Transverse Mercator -------------------------------------------
F0 = 0.9996012717; lat0 = deg2rad(49); lon0 = deg2rad(-2);
E0 = 400000; N0 = -100000;
e2 = 1 - b^2/a^2;  n = (a-b)/(a+b);
sl = sin(lat); cl = cos(lat); tl = tan(lat);
nu  = a*F0./sqrt(1 - e2*sl.^2);
rho = a*F0*(1-e2)./(1 - e2*sl.^2).^1.5;
eta2 = nu./rho - 1;

dlat = lat - lat0; slat = lat + lat0;
M = b*F0*( (1 + n + 5/4*n^2 + 5/4*n^3).*dlat ...
         - (3*n + 3*n^2 + 21/8*n^3).*sin(dlat).*cos(slat) ...
         + (15/8*n^2 + 15/8*n^3).*sin(2*dlat).*cos(2*slat) ...
         - 35/24*n^3.*sin(3*dlat).*cos(3*slat) );
I    = M + N0;
II   = nu/2.*sl.*cl;
III  = nu/24.*sl.*cl.^3.*(5 - tl.^2 + 9*eta2);
IIIA = nu/720.*sl.*cl.^5.*(61 - 58*tl.^2 + tl.^4);
IV   = nu.*cl;
V    = nu/6.*cl.^3.*(nu./rho - tl.^2);
VI   = nu/120.*cl.^5.*(5 - 18*tl.^2 + tl.^4 + 14*eta2 - 58*tl.^2.*eta2);

dl = lon - lon0;
N = I + II.*dl.^2 + III.*dl.^4 + IIIA.*dl.^6;
E = E0 + IV.*dl + V.*dl.^3 + VI.*dl.^5;
end

function [x, y, z] = ll2xyz(lat, lon, a, b)
e2 = 1 - b^2/a^2;
nu = a./sqrt(1 - e2*sin(lat).^2);
x = nu.*cos(lat).*cos(lon);
y = nu.*cos(lat).*sin(lon);
z = (1-e2)*nu.*sin(lat);
end

function [lat, lon] = xyz2ll(x, y, z, a, b)
e2 = 1 - b^2/a^2;
p = hypot(x, y);
lat = atan2(z, p*(1-e2));
for k = 1:10                                  % fixed-point iteration
    nu = a./sqrt(1 - e2*sin(lat).^2);
    lat = atan2(z + e2*nu.*sin(lat), p);
end
lon = atan2(y, x);
end
