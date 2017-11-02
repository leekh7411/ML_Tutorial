function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values

m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% ====================== Cost Function Reg ===========================================================|
for i = 1:m
	J = J + (-1 * y(i,1) * log(hypothesis(i,theta,X)) - (1 - y(i,1)) * log(1 - hypothesis(i,theta,X)));
end
J = J / m;

regular = 0;
% j = 1, theta(0) should not regularize
for j = 2:n
	regular = regular + (theta(j) * theta(j));
end

J = J + (lambda/(2*m)) * (regular);

% ================================================================================================|


% ====================== Gradient ================================================================|
for j = 1:length(theta)
	grad(j) = 0;
	for i = 1:m
		grad(j) = grad(j) + ((hypothesis(i,theta,X) - y(i)) * X(i,j));
	end

	if(j == 1)
		grad(j) = (grad(j) / m);
	else 
		grad(j) = (grad(j) / m) + (lambda/m) * theta(j);
	endif
	
end
% ================================================================================================|
% ================================================================================================|

end
