function [J grad] = nnLinealCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Generate Y in format [0 1 0 0 0 0 0 0 0 0; ....]
Y = y;

%  output_layer_costs(example,label)
output_layer_costs = zeros(m,1); % cost function for every example in output layer
delta_3 = zeros(m,1); % delta "error" for every example in output layer
delta_2 = zeros(m,hidden_layer_size); % delta "error" for every example in hidden layer
DELTA_1 = zeros(size(Theta1)); % DELTA acumulative error for layer 1
DELTA_2 = zeros(size(Theta2)); % DELTA acumulative error for layer 2
for i = 1:m,
	% FORWARD PROPAGATION for example i
	% Layer 1
	% size (m+1) x 1
	a1 = [1 X(i,:)];

	% Layer 2
	% size 1 x (hidden_layer_size+1)
	a2 = (a1 * Theta1');

	% Layer 3
	a2 = [1 a2];
	a3 = (a2 * Theta2');	% a3 prediction of example i
	

	output_layer_costs(i,:) = (1/(2*m))*((a3-Y(i,:)).^2);
	% J = (1/(2*m))*sum((h - y).^2);
	% J = (-(1/m)*sum(y.*log(h) + (1-y).*log(1-h))) + (lambda/(2*m))*sum(theta_temp.^2);

	% BACK PROPAGATION for example i

	% size(delta_3(i,:)) = num_labels x 1
	delta_3(i,:) = (a3 - Y(i,:))';
	
	% size(((delta_3(i,:)*Theta2).*(a2.*(1-a2)))') = (hidden_layer_size+1) x 1
	delta_2_temp = ((delta_3(i,:)*Theta2).*(a2.*(1-a2)))';

	% remove bias element
	delta_2_temp(1,:) = [];

	% size(delta_2(i,:)) = hidden_layer_size x 1
	delta_2(i,:) = delta_2_temp;

	
	DELTA_1 = DELTA_1 + delta_2(i,:)'*a1;
	DELTA_2 = DELTA_2 + delta_3(i,:)'*a2;

end

J = sum(sum(output_layer_costs));


% Regularization
theta1_temp = Theta1;
theta1_temp(:,1) = zeros(hidden_layer_size,1);
theta2_temp = Theta2;
theta2_temp(:,1) = zeros(num_labels,1);;



J = J + sum((lambda/(2*m))*sum(theta1_temp.^2)) + sum((lambda/(2*m))*sum(theta2_temp.^2));



Theta1_grad = (1/m)*DELTA_1 + (lambda/m)*theta1_temp;
Theta2_grad = (1/m)*DELTA_2 + (lambda/m)*theta2_temp;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
