
function [X_norm, X_max, X_min, from, to] = normalizeBetween(X,from,to)
X_max = max(X);
X_min = min(X);
X_norm = from + ((X - X_min)*(to-from))./(X_max - X_min);
end
