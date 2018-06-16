function [qlike, mae, Popt] = loss_comput(y,x) 
% -- PURPOSE :  This function returns the loss functions L(t)


% -- Rolling window regression based on 3000 past obs
for i = 1:500
beta = [ones(3000,1) x(i:2999+i,:)]\y(i:2999+i);
yhat_out(i) = [1 x(3000+i,:)]*beta;
end

% -- Selecting out-of-sample obs for forecasting performance comparison                                                             
y_out = y(3001:3500);

% -- Loss function 1 : Mean Absolute Error (MAE)
mae  = abs(y_out - yhat_out');                              % 500x1 matrix                           

% -- Loss function 2 : Negative Gaussian Quasi-likelihood (QLIKE)
qlike = log(yhat_out') + bsxfun(@rdivide,y_out,yhat_out');  % 500x1 matrix

% -- Expected option price
Popt = 2*cdf('norm',0.5*sqrt(yhat_out'))-1;

end