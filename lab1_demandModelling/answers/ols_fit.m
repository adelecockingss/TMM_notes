function [coef, R2, yhat] = ols_fit(y, X, names)
%OLS_FIT Ordinary least squares with intercept, base MATLAB only.
%   [coef, R2] = ols_fit(y, X, names) returns a table with Estimate, SE,
%   tStat and pValue for [intercept; each column of X] - the same numbers
%   as fitlm(X, y) from the Statistics Toolbox prints.
%
%   Maths:  beta = (X'X)^-1 X'y        (computed stably with backslash)
%           s^2  = RSS / (n - p)        residual variance
%           SE   = sqrt(diag(s^2 (X'X)^-1))
%           t    = beta / SE,  p = 2 P(T_{n-p} > |t|)

y = double(y(:));
X = [ones(size(X, 1), 1), double(X)];
[n, p] = size(X);
beta = X \ y;
yhat = X*beta;
res = y - yhat;
s2 = (res'*res)/(n - p);
SE = sqrt(diag(s2*inv(X'*X)));
t = beta./SE;
nu = n - p;
pval = betainc(nu./(nu + t.^2), nu/2, 0.5);    % two-sided t-test p-value
R2 = 1 - (res'*res)/sum((y - mean(y)).^2);

if nargin < 3, names = compose('x%d', 1:p-1); end
coef = table(beta, SE, t, pval, 'VariableNames', {'Estimate', 'SE', 'tStat', 'pValue'}, ...
    'RowNames', [{'(Intercept)'}, cellstr(names)]);
end
