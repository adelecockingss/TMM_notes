%% TMM Lab 02 - Q5 in MATLAB: readgeotable (NEEDS the Mapping Toolbox - untested here)
% Your MATLAB has no Mapping Toolbox, so this was not run. With the toolbox:
gunzip('../Bristol.osm.gz', tempdir);                 % readgeotable wants the .osm
osm = fullfile(tempdir, 'Bristol.osm');
lines = readgeotable(osm, Layer = "lines");           % ways as geolinestrings
roads = lines(ismember(lines.highway, ["primary" "secondary" "tertiary" "residential"]), :);
figure; geoplot(roads); geobasemap streets
% As the lab sheet says, this gives SHAPES, not a network (node/way links are
% lost), so for Q9 onwards use the tables built in Python: q9_q11_osm.m
