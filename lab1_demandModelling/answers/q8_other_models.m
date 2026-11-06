%% TMM Lab 01 - Q8: intervening opportunities and radiation models
% Base MATLAB. Run q3_od_matrix.m first. Python version: q8_other_models.py
clear; close all;
z = load_od_level('bristol');
T = z.T; O = z.O; D = z.D; dist = z.dist; n = numel(O);
s = opportunities_matrix(D, dist);
offdiag = ~eye(n);

%% Intervening opportunities: P_ij = exp(-L V_ij) - exp(-L (V_ij + D_j))
% V_ij = D_i + s_ij jobs passed before reaching j (V_ii = 0). Rows -> O_i.
V = s + D; V(1:n+1:end) = 0;
io = @(L) O.*normalise_rows(exp(-L*V) - exp(-L*(V + D')));
logL = fminbnd(@(q) -cpc(io(10^q), T), -8, -2);   % 1-D search, base MATLAB
T_io = io(10^logL);
fprintf('Intervening opportunities: best L = %.2e per job\n', 10^logL);

%% Radiation model: T_ij = O_i m_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij))
Trad = O.*(O.*D')./((O + s).*(O + D' + s));
Trad(1:n+1:end) = 0;
T_out = T.*offdiag;
Trad = Trad.*(sum(T_out, 2)./max(sum(Trad, 2), eps));   % match trips leaving i

%% Compare (between-zone trips only)
[~, ~, T_dbl] = furness(O, D, dist.^-1.4);
fit_report('doubly constrained d^-1.4', T_dbl.*offdiag, T_out);
fit_report('intervening opportunities', T_io.*offdiag, T_out);
fit_report('radiation (no parameters)', Trad, T_out);

function P = normalise_rows(P)
P = P./sum(P, 2);
end
