function pred = predictLinear(Theta1, Theta2, X, X_max, X_min, pred_max, pred_min, y_max, y_min, from, to)

X_norm = from + ((X - X_min)*(to-from))./(X_max - X_min);

% Useful values
m = size(X_norm, 1);
% num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
pred = zeros(size(X_norm, 1), 1);

h1 = sigmoid([ones(m, 1) X_norm] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');



pred = y_min + ((h2 - pred_min)*(y_max-y_min))./(pred_max - pred_min);



% =========================================================================


end