function [X_norm, mu, sigma] = normalize(X)
X_norm = X;
mu = mean(X);
sigma = std(X);
X_norm = (X-mu)./sigma;
end
