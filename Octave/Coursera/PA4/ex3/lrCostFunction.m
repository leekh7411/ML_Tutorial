function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there re many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we dont add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% =======================================================================================================================================================
% 'Vectorized , Regularized Cost Function (Yeeaaahh!!!)'
J_func = @(_x,_y) ((-1 * _y) .* log(sigmoid(_x * theta)) - (1 - _y) .* log(1 - sigmoid(_x * theta)));
regularize_func = @(_t) _t(2:end) .* _t(2:end);
J = J_func(X,y);
J = sum(J) / m;
sqr_theta = theta;
sqr_theta(2:end) = regularize_func(theta);
sqr_theta(1) = 0;
J = J + (lambda * sum(sqr_theta) / (2*m));
% =======================================================================================================================================================


% =======================================================================================================================================================
temp_theta = theta;
temp_theta(1) = 0;
grad = (X' * (sigmoid(X * theta) - y)) / m + temp_theta * lambda / m;
%printf("And the Grad value is .. %f\n",grad);
end
% =======================================================================================================================================================