function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
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
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

m = size(X, 1);
X = [ones(m, 1), X];

a2 = sigmoid(X*Theta1');
a2 = [ones(m, 1), a2];

a3 = sigmoid(a2 * Theta2');
Y = eye(num_labels)(y,:);
J = (1/m) * (sum(sum(-Y .* log(a3) - (1-Y).* log(1- a3))));

new_Theta1 = Theta1;
new_Theta1(:, 1) = 0;

new_Theta2 = Theta2;
new_Theta2(:, 1) = 0;

reg_param = (lambda / (2*m)) * (sum(sum(new_Theta1 .* new_Theta1)) + sum(sum(new_Theta2 .* new_Theta2)));

J = J + reg_param;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for i=1:m
%set a1 = X(1)
%perform forward propagation to compute aL
%using y(i) compute del(L) = a(L) - y(I)
%compute del(L-1), del(L-2)... del(2)
%capital_del := capital_del + a(L) * del(L+1)
	%FORWARD PROPAGATION%
     X_i = X(i, :);
	%a1 = [ones(1,1) a1];
    z2_i = X_i*Theta1';
	a2_i = sigmoid(z2_i);
    a2_i = [ones(1, 1) a2_i];
    z3_i = a2_i * Theta2';
    a3_i = sigmoid(z3_i);
    %BACK PROPAGATION%
    del3_i = a3_i - Y(i, :);
    del2_i = (del3_i * Theta2) .* [ones(1, 1) sigmoidGradient(X_i*Theta1')];
    del2_i = del2_i(2:end);
    
    Theta1_grad  = Theta1_grad + (del2_i' * X_i);
    Theta2_grad  = Theta2_grad + (del3_i' * a2_i);
    
end

% unregularized
Theta1_grad = (1/m) .* Theta1_grad;
Theta2_grad = (1/m) .* Theta2_grad;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


grad1 = (lambda / m) .* [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
grad2 = (lambda / m) .* [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = Theta1_grad + grad1;
Theta2_grad = Theta2_grad + grad2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
