function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
n
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%	
% =======================================================================================================================================================
% 'Vectorized , Regularized Cost Function (Yeeaaahh!!!)'
% In Linear Regression, 'H of theta' means that Sigmoid Function with 'X' and 'Theta'

J_func = @(_x,_y) ((sum(((_x * theta) - _y).^2) + lambda * sum(theta(2:length(theta)).^2)) / (2*m));
J = J_func(X,y);

% =======================================================================================================================================================
gradient_func = @(_x,_y,_lambda,_j)((sum((_x*theta - _y) .* _x(:,_j)) / m) + (_lambda / m * theta(_j)));

for i=1:n
	if(i == 1)
		grad(i) = gradient_func(X,y,0,i);
	else
		grad(i) = gradient_func(X,y,lambda,i);
	end
end

end
