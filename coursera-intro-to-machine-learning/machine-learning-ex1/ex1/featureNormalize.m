function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by its standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

%X1 = X(:,1);
%X2 = X(:,2);


%mu1 = mean(X1);
%mu2 = mean(X2);


%sigma1 = std(X1);
%sigma2 = std(X2);


%X1 = (X1-mu1) ./ sigma1;
%X2 = (X2-mu2) ./ sigma2;

%mu = [mu1 mu2];
%sigma = [sigma1 sigma2];
%X_norm = [X1 X2];

no_of_cols = columns(X);
for i=1:no_of_cols
	temp = X(:,i);
	mu(i) = mean(temp);
	sigma(i) = std(temp);
	X_norm(:,i) = (temp - mu(i)) ./ sigma(i);
end



% ============================================================

end
