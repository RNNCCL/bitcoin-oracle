function [pred_norm, pred_max, pred_min, pred] = normalizePrediction(Theta1, Theta2, X_norm, y_max, y_min)


% Useful values
m = size(X_norm, 1);
% num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X_norm, 1), 1);

h1 = sigmoid([ones(m, 1) X_norm] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

% size(h2)
% [dummy, p] = max(h2, [], 2);

% Denormalize output
% p = h2;
% printf('h2: ');
% h2

% p = h2;

[pred_norm, pred_max, pred_min] = normalizeBetween(h2,y_min,y_max);
pred = h2;
% p = (((h2-from).*(y_max-y_min)) / (to-from) ) + y_min;


% =========================================================================


end