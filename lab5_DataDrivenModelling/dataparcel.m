%% create data parcels for machine learning a microscopic pedestrian model
% iid refers to the pedestrian selected
% datall stores all trajectory data
% the function creates an input for the model, that captures the presence
% of other pedestrians in the vicinity
% it produces an output that captures the next position of the pedestrian
% on a grid.
% and it also records the current position of the pedestrian on the grid
function [input, output, currpos] = dataparcel(iid, datall)
    raus = zeros(9, 9);
    raus(5, 5) = 1; % Position of individual
    
    indics2 = find(datall(:, 1) == iid);
    itime = randi([1, length(indics2)-1], 1); % Randomly pick a time out of the available observed positions for the pedestrian selected
    time = datall(indics2(itime), 2);
    nextpos = datall(indics2(itime + 1), 3:4);
    
    indics = find(datall(:, 2) == time);
    indics2 = intersect(indics, indics2);

    for u = indics'
        ddx = -datall(indics2, 3) + datall(u, 3);
        ddy = -datall(indics2, 4) + datall(u, 4);
        if abs(ddx) <= 4 && abs(ddy) <= 4
            raus(5 + ddx, 5 + ddy) = 1;
        end
    end

    % Create output grid
    output = zeros(7, 7);
    ddx = -datall(indics2, 3) + nextpos(1);
    ddy = -datall(indics2, 4) + nextpos(2);
    if abs(ddx) <= 3 && abs(ddy) <= 3
        output(4 + ddx, 4 + ddy) = 1;
    end
    
    input = raus(:);
    output = output(:);
    currpos = datall(indics2, 3:4);
end
