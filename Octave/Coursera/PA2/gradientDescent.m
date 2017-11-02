function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
m_theta = length(theta);
temp = zeros(m_theta,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    for i_theta = 1:m_theta
        for i=1:m
            temp(i_theta,1) = temp(i_theta,1) + ((theta(1,1) * X(i,1) + theta(2,1) * X(i,2)) - y(i,1))*X(i,i_theta);
        end
    end    

    for i_theta = 1:m_theta
        theta(i_theta,1) = theta(i_theta,1) - alpha * temp(i_theta,1) / m;
        temp(i_theta,1) = 0;
    end

    % fprintf('result : %f\n',computeCost(X, y, theta));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
