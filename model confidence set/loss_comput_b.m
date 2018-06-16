function [qlike, mae, Popt] = loss_comput_b(yb,xb,y,x) 
% -- PURPOSE :  This function returns the bootstrap loss functions L(t)

% -- Rolling window regression based on 3000 past obs, forward obs are
% taken from the original serie
mdl = LinearModel.fit(xb,yb); 
beta = mdl.Coefficients.Estimate;
for i = 1:500
    yhat_out(i) = [1 x(3000+i,:)]*beta;
    beta = [ones(3000,1) [xb(i+1:end,:);x(3001:3000+i,:)]]\[yb(i+1:end);y(3001:3000+i)];
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