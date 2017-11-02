function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;
average = 999999;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
mult_factor = 2;
best_C = 0;
best_sigma = 0;
while (C < 10)
  while (sigma < 10)
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    avg_temp = mean(double(predictions ~= yval)); 
    if (avg_temp < average)
      average = avg_temp;
      best_C = C;
      best_sigma = sigma;
    endif
    sigma = sigma * mult_factor;
  end
  C = C * mult_factor;
  sigma = 0.01;
end 

C = best_C;
sigma = best_sigma;


% =========================================================================

end
