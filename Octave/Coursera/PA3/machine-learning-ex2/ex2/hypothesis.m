function H = hypothesis(i,theta, X)
% i : Index of X(i)
len = length(theta);
H = 0;
for idx = 1:len
	H = H + theta(idx,1) * X(i,idx);
end

H = sigmoid(H);