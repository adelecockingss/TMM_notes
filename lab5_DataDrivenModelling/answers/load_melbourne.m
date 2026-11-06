function [central, xs, ys, sensx, sensy, synthetic] = load_melbourne()
%LOAD_MELBOURNE Real Blackboard files if present in the lab folder, else the
%   synthetic stand-in written by melbourne_data.py (answers/data/synthetic).
%   ys is flipped (655 - ys) as in the lab sheet; sensy is not.
names = {'pedestrian_counts', 'lines_xs', 'lines_ys', 'sensors_xs', 'sensors_ys'};
f = cell(size(names)); synthetic = false;
for k = 1:numel(names)
    c = {fullfile('..', [names{k} '.txt']), fullfile('..', [names{k} ' (1).txt'])};
    f{k} = '';
    for j = 1:2, if isfile(c{j}), f{k} = c{j}; end, end
    if isempty(f{k}), synthetic = true; end
end
if synthetic
    d = fullfile('data', 'synthetic');
    if ~isfile(fullfile(d, 'pedestrian_counts.txt'))
        error('Run melbourne_data.py (or s1_direct_demand.py) once to create the synthetic data');
    end
    f = fullfile(d, strcat(names, '.txt'));
    central = load(f{1});
    disp('NOTE: using SYNTHETIC stand-in data (Melbourne files not found)');
else
    central = readmatrix(f{1});
    central = central(:, 2:1249);             % drop the location-name column
end
xs = load(f{2}); ys = 655 - load(f{3});
sensx = load(f{4}); sensy = load(f{5});
end
