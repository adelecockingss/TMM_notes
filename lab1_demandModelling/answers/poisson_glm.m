function beta = poisson_glm(y, X)
%POISSON_GLM Poisson regression with log link, by iteratively reweighted
%   least squares (IRLS) - what glmfit(X, y, 'poisson') does (Statistics
%   Toolbox). Model: log E[y] = [1 X] * beta.
%   Each step solves a weighted least-squares problem with weights mu
%   (Poisson variance = mean) on the "working response" z.
X = [ones(size(X, 1), 1) X];
beta = X \ log(y + 0.5);                 % start from a log-linear fit
for it = 1:50
    eta = X*beta;  mu = exp(eta);
    z = eta + (y - mu)./mu;              % working response
    W = mu;                              % working weights
    betaNew = (X'*(W.*X)) \ (X'*(W.*z));
    if max(abs(betaNew - beta)) < 1e-8, beta = betaNew; break; end
    beta = betaNew;
end
end
