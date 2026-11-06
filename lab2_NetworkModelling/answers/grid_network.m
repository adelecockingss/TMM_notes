function [G, xy] = grid_network(n, L)
%GRID_NETWORK n x n Manhattan grid with pitch L (the lab's code, as a function).
A = -L*delsq(numgrid('S', n+2));      % 5-point Laplacian -> grid adjacency
G = graph(A, 'omitselfloops');        % off-diagonal entries are -(-L) = L
k = (0:n^2-1)';
xy = [L*floor(k/n), L*rem(k, n)];
end
