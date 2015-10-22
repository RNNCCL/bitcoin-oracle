% Load data
clear all
% data from https://www.quandl.com/data/BCHARTS/BITSTAMPUSD-Bitcoin-Markets-bitstampUSD

data = load('bitstamp-6hours-open-high-low-close-volumeBtc-volumeUsd-weightedPrice.csv');


[X y] = parseData(data,5);
% X = [-2:.01:3]';
% X = [X -([-4:.02:6]'.^(2))];
% X = [X rand(size(X,1),1)];
% y = (X(:,1)+9).^4;
% X = [X y];
% size(X)
% size(y)
% y = [-10:.01:0]';
% y = [y;[0:-.01:(-10+0.01)]'];





% hidden_layer_size = 100;
lambda = 0.7;



bestError = 99999999;

from = 0.25;
to = 0.75;
num_labels = 1;
[m n] = size(X);
n0 = n;
for i=1:(6*60*8)),
	t = cputime;
	printf('Training %d...\n',i);
	[m n] = size(X);
	input_layer_size = n;
	hidden_layer_size = n+2;

	[X_norm, X_max, X_min, from, to] = normalizeBetween(X,from,to);
	[y_norm, y_max, y_min, from, to] = normalizeBetween(y,from,to);

	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

	% Unroll parameters
	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

	% Train NN
	options = optimset('MaxIter', 1000);



	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunctionVectorized(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X_norm, y_norm, lambda);

	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));



	pred = predict2(Theta1, Theta2, X, X_max, X_min, from, to, y_max, y_min);

	% plotPrediction(m,y,pred);
	err = mean(double((pred - y).^2));
	fprintf('\nTraining Error: %f (%f seconds)\n', err,cputime-t);
	if(err < bestError),
		save bestfit.mat Theta1 Theta2 X X_max X_min from to y_max y_min;
		printf('best fit saved!\n');
	end


	X(:,end+1) = pred;

end
