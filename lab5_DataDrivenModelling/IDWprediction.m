%% Predict counts using IDW interpolation
function predicts = IDWprediction(nodes, counts, sensors, alldists)
    n = length(nodes);
    predicts = NaN(n, 1);
    for i = 1:n
        shortpaths = alldists(:, nodes(i));
        ids = 1:length(sensors);
        if any(shortpaths == 0)
            ids(shortpaths == 0) = [];
        end
        predicts(i) = sum(counts(ids) ./ shortpaths(ids).^2) / sum(1 ./ shortpaths(ids).^2);
    end
end