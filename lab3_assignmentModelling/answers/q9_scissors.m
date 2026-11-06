%% TMM Lab 03 - Q9: the lecture's two-origin 'scissors' network
% networkExampleLecture2021.m, completed (incl. the SO-in-link-variables
% exercise). Uses quadprog if available, else qp_small (base MATLAB).
clear;
A = [0 1 0 0; 0 0 0 1; 0 1 0 1; 1 0 0 0; 0 0 1 0];   % link-route incidence
a = [2 2 1 4 4]'; b = ones(5, 1);
M = [1 1 0 0; 0 0 1 1];                              % OD-route incidence
d = [1; 1];
B = [-1 0 0 -1 0; 0 -1 0 0 -1; 1 1 -1 0 0; 0 0 1 1 1];  % node-link conservation
s = [-d(1); -d(2); 0; sum(d)];
Q = @(H, f, Aeq, beq) solveqp(H, f, Aeq, beq);

% UE, route variables (Beckmann): H = A' diag(b) A, f = A' a, My = d
yUE = Q(A'*diag(b)*A, A'*a, M, d);  xUE_r = A*yUE;
% SO, route variables: double the Hessian
ySO = Q(2*A'*diag(b)*A, A'*a, M, d); xSO_r = A*ySO;
% UE, link variables: H = diag(b), f = a, Bx = s
xUE_l = Q(diag(b), a, B, s);
% SO, link variables - the exercise: H = 2 diag(b)
xSO_l = Q(2*diag(b), a, B, s);

disp(table(xUE_r, xUE_l, xSO_r, xSO_l, 'VariableNames', ...
     {'UE_routes', 'UE_links', 'SO_routes', 'SO_links'}))
fprintf('route costs at UE: %s (used routes of each OD are equal)\n', mat2str((A'*(a + b.*xUE_l))', 4));
fprintf('system cost UE %.4f, SO %.4f, PoA %.4f\n', sum(xUE_l.*(a + b.*xUE_l)), ...
        sum(xSO_l.*(a + b.*xSO_l)), sum(xUE_l.*(a + b.*xUE_l))/sum(xSO_l.*(a + b.*xSO_l)));
fprintf('route flows recovered from link flows: %s\n', mat2str((A\xUE_l)', 4));

function x = solveqp(H, f, Aeq, beq)
if exist('quadprog', 'file') == 2
    x = quadprog(H, f, [], [], Aeq, beq, zeros(numel(f), 1), [], [], optimoptions('quadprog', 'Display', 'off'));
else
    x = qp_small(H, f, Aeq, beq);
end
end
