function p = predict2(Theta1, Theta2, X, X_max, X_min, from, to, y_max, y_min)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Normalize input
% X_norm = (X-X_mu)./X_sigma;
X_norm = from + ((X - X_min)*(to-from))./(X_max - X_min);


% Useful values
m = size(X_norm, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X_norm, 1), 1);

h1 = sigmoid([ones(m, 1) X_norm] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

p = (((h2-from).*(y_max-y_min)) / (to-from) ) + y_min;

% [dummy, p] = max(h2, [], 2);

% Denormalize output
% p = (p.*y_sigma)+y_mu;

% =========================================================================


end