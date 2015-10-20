% Load data
clear all
% data = load('bitstampUSD-test.csv');
% [X y] = parseData2(data);
X = [0:.0005:1]';
X = [X [0:.0005:1]'];
y = [0:.0005:1]';
% [X_norm, X_mu, X_sigma] = normalize(X)
% [y_norm, y_mu, y_sigma] = normalize(y);
% load('ex4data1.mat');
[m n] = size(X);

% Set parameters
input_layer_size = n;
hidden_layer_size = 3;
num_labels = 1;
lambda = 0;


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Train NN
options = optimset('MaxIter', 20);



% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnLinearCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


pred = predictNormalized(Theta1, Theta2, X, 0, 1, 0, 1);
[pred y]

fprintf('\nTraining Set Accuracy: %f\n', mean(double((pred - y).^2)));