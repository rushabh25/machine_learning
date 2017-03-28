function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    J_history(iter) = computeCost(X, y, theta);

    % theta = theta - alpha * delta
    hypothesis = X * theta;
    difference = hypothesis - y;
    product0 = alpha * (sum(difference .* X(:,1)) / m);
    product1 = alpha * (sum(difference .* X(:,2)) / m);

    theta0 = theta(1,1) - product0;
    theta1 = theta(2,1) - product1;

    theta = [theta0; theta1];

    % ============================================================

    % Save the cost J in every iteration    
    


    %disp("display all the variables");
    %printf("iteration: %d", iter);
    %disp(theta);
    %disp(J_history);


end

end
