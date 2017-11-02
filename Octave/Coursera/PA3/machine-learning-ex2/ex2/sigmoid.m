function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


s = size(z);
m = length(s);
%printf("Size   of Z : %f\n" , s(2));
%printf("Length of s : %f\n" , length(s));

if(m == 2)
	for row = 1:s(1)
		for col = 1:s(2)
			g(row,col) = 1 / (1 + exp(-1*z(row,col)));
		end
	end
else 
	for i=1:m
		g(i) = 1/(1+exp(-1*z(i)));
	end
endif
% =============================================================
end
