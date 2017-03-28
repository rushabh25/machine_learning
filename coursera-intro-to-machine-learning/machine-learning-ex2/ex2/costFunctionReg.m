function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

actual_cost = ((-y' * log(sigmoid(X * theta))) - ((1-y') * log(1 - sigmoid(X * theta)))) ./ m;

no_of_cols = columns(X);
temp = zeros(size(theta));

%for j=2:no_of_cols
%	temp = theta(j) .* theta(j);
%end
%J = actual_cost + ((lambda * sum(temp)) / (2*m));

%exclude theta1 from regularization, start from theta2
theta_new = zeros(size(theta));
theta_new(2:size(theta_new),:) = theta(2:size(theta),:);

J = actual_cost + ((lambda * (theta_new' * theta_new)) / (2*m));

grad = ((X' * (sigmoid(X*theta) - y)) / m) + ((lambda * theta_new) /m);

% =============================================================

end
