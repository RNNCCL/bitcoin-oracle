
% denormalizes X_norm from to, with X_min X_max
function [X] = denormalizeBetween(X_norm, X_max, X_min, from, to)

X = (((X_norm-from).*(X_max-X_min)) / (to-from) ) + X_min;

end
