function p = predictNormalized(Theta1, Theta2, X, X_mu, X_sigma, y_mu, y_sigma)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Normalize input
X_norm = (X-X_mu)./X_sigma;

% Useful values
% [X_mu, X_sigma, y_mu, y_sigma]
m = size(X_norm, 1);
% num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X_norm, 1), 1);

h1 = [ones(m, 1) X_norm] * Theta1';
h2 = [ones(m, 1) h1] * Theta2';

% size(h2)
% [dummy, p] = max(h2, [], 2);

% Denormalize output
% p = h2;
% printf('h2: ');
% h2

p = (h2.*y_sigma)+y_mu;

% =========================================================================


end