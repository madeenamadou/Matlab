function  [H0_MAE, H0_QLIKE, H0_LOSS_OPT, ER_MAE, ER_QLIKE, ER_LOSS_OPT] = mcs_iter(ind1, ind2, ind3, it)
% -- PURPOSE : This function is responsible of each iteration of the procedure 
% of the tes based on Model Confidence Set 
% and returns the decision, statistics and the p-values

% -- Loading data from initial collection of models
load mcs

% -- Selecting loss function values of surviving models
MAE_it = MAE(:,ind1);
QLIKE_it = QLIKE(:,ind2);
LOSS_OPT_it = LOSS_OPT(:,ind3);

% -- Same number of bootstrap resamples
B = 500;

% -- Computing di. the loss of every model relative to the average accross the
% surviving models

di_mae_t = bsxfun(@plus,MAE_it,-mean(MAE_it,2)); 
di_qlike_t = bsxfun(@plus,QLIKE_it,- mean(QLIKE_it,2));

di_mae = mean(di_mae_t,1)';
di_qlike = mean(di_qlike_t,1)';

di_loss_opt_t = bsxfun(@plus,LOSS_OPT_it,-mean(LOSS_OPT_it,2));                               % 500x1 matrix
di_loss_opt = mean(di_loss_opt_t,1)'; 

% -- For each bootstrap resample, computing di.(b) the loss of every model 
% relative to the average accross the surviving models 
MAE_b_it = MAE_b(:,ind1,:);
QLIKE_b_it = QLIKE_b(:,ind2,:);
LOSS_OPT_b_it = LOSS_OPT_b(:,ind3,:);

    for b = 1:B
        di_mae_t_b(:,:,b) = bsxfun(@plus,MAE_b_it (:,:,b),- mean(MAE_b_it (:,:,b),2));
        di_qlike_t_b(:,:,b) = bsxfun(@plus,QLIKE_b_it (:,:,b),- mean(QLIKE_b_it (:,:,b),2));
        di_mae_b(:,b) = mean(di_mae_t_b(:,:,b),1)';
        di_qlike_b(:,b) = mean(di_qlike_t_b(:,:,b),1)';
        di_loss_opt_t_b(:,:,b) = bsxfun(@plus,LOSS_OPT_b_it(:,:,b),-mean(LOSS_OPT_b(:,:,b),2));                               % nx1 matrix
        di_loss_opt_b(:,b) = mean(di_loss_opt_t_b(:,:,b),1)';
    end

% -- Computing the standard deviation of di.(b) 
std_di_mae_b = std(di_mae_b,1,2);             
std_di_qlike_b = std(di_qlike_b,1,2);  
std_di_loss_opt_b = std(di_loss_opt_b,1,2); 

% -- Computing the student statistic for original sample and bootstraps
% pairs resamples
ti_mae = di_mae.*(std_di_mae_b.^(-1));      
ti_qlike = di_qlike.*(std_di_qlike_b.^(-1));
ti_loss_opt = di_loss_opt.*(std_di_loss_opt_b.^(-1));
ti_mae_b = bsxfun(@times,bsxfun(@plus,di_mae_b,-di_mae),(std_di_mae_b.^(-1)));       
ti_qlike_b = bsxfun(@times,bsxfun(@plus,di_qlike_b,-di_qlike),(std_di_qlike_b.^(-1)));    
ti_loss_opt_b = bsxfun(@times,bsxfun(@plus,di_loss_opt_b,-di_loss_opt),(std_di_loss_opt_b.^(-1)));    

% -- Computing the statistic test for each loss function 
[T_mae_sort,Ind_mae_sort] = sort(ti_mae,'descend');
Tmax_mae = T_mae_sort(1);     
Ind_max_mae = Ind_mae_sort(1);
[T_qlike_sort,Ind_qlike_sort] = sort(ti_qlike,'descend');
Tmax_qlike = T_qlike_sort(1);     
Ind_max_qlike = Ind_qlike_sort(1);
[T_loss_opt_sort,Ind_loss_opt_sort] = sort(ti_loss_opt,'descend');
Tmax_loss_opt = T_loss_opt_sort(1);     
Ind_max_loss_opt = Ind_loss_opt_sort(1);

% -- Getting the statistic test bootstrap distribution for each loss function 
[Tmax_mae_b_sort,Ind_mae_b_sort] = sort(ti_mae_b,1,'descend');
Tmax_mae_b = Tmax_mae_b_sort(1,:);
[Tmax_qlike_b_sort,Ind_qlike_b_sort] = sort(ti_qlike_b,1,'descend');
Tmax_qlike_b = Tmax_qlike_b_sort(1,:);
[Tmax_loss_opt_b_sort,Ind_loss_opt_b_sort] = sort(ti_loss_opt_b,1,'descend');
Tmax_loss_opt_b = Tmax_loss_opt_b_sort(1,:);

% -- Alpha, the significance thresold
alpha = 0.1;

% -- Computing the 1-alpha quantile of the test statistic distribution
Q_Tmax_mae_b = quantile(Tmax_mae_b,1-alpha);                  
Q_Tmax_qlike_b = quantile(Tmax_qlike_b,1-alpha);                
Q_Tmax_loss_opt_b = quantile(Tmax_loss_opt_b,1-alpha); 

% -- Computing p-value
pvalue_mae = mean((Tmax_mae_b > Tmax_mae ));
pvalue_qlike = mean((Tmax_qlike_b > Tmax_qlike ));        
pvalue_loss_opt = mean((Tmax_loss_opt_b > Tmax_loss_opt ));  

display(sprintf('+------------------+'));
display(sprintf('+   Iteration %u    +',it));
display(sprintf('+------------------+'));

display(sprintf('Individual tests for each model'));
display(sprintf('---Loss function MAE'));
for i=1:size(ind1,2)
    display(sprintf('(MAE): Model[%u]| P-Value: [%1.1g] | t-stat: [%1.2g] | Q(%g): [%1.2g]',ind1(i),mean((Tmax_mae_b > ti_mae(i))),ti_mae(i),1-alpha,Q_Tmax_mae_b));

end
display(sprintf('---Loss function QLIKE'));
for i=1:size(ind2,2)
    display(sprintf('(QLIKE): Model[%u]| P-Value: [%1.1g] | t-stat: [%1.2g] | Q(%g): [%1.2g]',ind2(i),mean((Tmax_qlike_b > ti_qlike(i))),ti_qlike(i),1-alpha,Q_Tmax_qlike_b));
end
display(sprintf('---Loss on option trade'));
for i=1:size(ind3,2)
    display(sprintf('(LOSS_OPT): Model[%u]| P-Value: [%1.1g] | t-stat: [%1.2g] | Q(%g): [%1.2g]',ind3(i),mean((Tmax_loss_opt_b > ti_loss_opt(i))),ti_loss_opt(i),1-alpha,Q_Tmax_loss_opt_b));
end

display(sprintf('------------------'));
display(sprintf('Global test for M'));
display(sprintf('------------------'));
% -- Rendering decision according to each loss function
if  pvalue_mae < alpha
    H0_MAE = 0;
    ER_MAE = ind1(Ind_max_mae);
    display('H0 rejected based on MAE'); 
    display(sprintf('P-value: [%1.1g] | Q(%g): [%1.2g] | Tmax: [%1.2g]',pvalue_mae,1-alpha,Q_Tmax_mae_b,Tmax_mae));
    display(sprintf('Eliminate model %u',ER_MAE));
else
    H0_MAE = 1;
    ER_MAE = 0;
    display('H0 accepted based on MAE');
    display(sprintf('P-value: [%1.1g] | Q(%g): [%1.2g] | Tmax: [%1.2g]',pvalue_mae,1-alpha,Q_Tmax_mae_b,Tmax_mae));
end

if  pvalue_qlike < alpha
    H0_QLIKE = 0;
    ER_QLIKE = ind2(Ind_max_qlike);
    display('H0 rejected based on QLIKE');
    display(sprintf('P-value: [%1.1g] | Q(%g): [%1.2g] | Tmax: [%1.2g]',pvalue_qlike,1-alpha,Q_Tmax_qlike_b,Tmax_qlike));
    display(sprintf('Eliminate model %u',ER_QLIKE));
else
    H0_QLIKE = 1;
    ER_QLIKE = 0;
    display('H0 accepted based on QLIKE');
    display(sprintf('P-value: [%1.1g] | Q(%g): [%1.2g] | Tmax: [%1.2g]',pvalue_qlike,1-alpha,Q_Tmax_qlike_b,Tmax_qlike));
end

if  pvalue_loss_opt < alpha
    H0_LOSS_OPT = 0;
    ER_LOSS_OPT = ind3(Ind_max_loss_opt);
    display('H0 rejected based on Loss after option trade');
    display(sprintf('P-value: [%1.1g] | Q(%g): [%1.2g] | Tmax: [%1.2g]',pvalue_loss_opt,1-alpha,Q_Tmax_loss_opt_b,Tmax_loss_opt));
    display(sprintf('Eliminate model %u',ER_LOSS_OPT));
else
    H0_LOSS_OPT = 1;
    ER_LOSS_OPT = 0;
    display('H0 accepted based on Loss after option trade');
    display(sprintf('P-value: [%1.1g] | Q(%g): [%1.2g] | Tmax: [%1.2g]',pvalue_loss_opt,1-alpha,Q_Tmax_loss_opt_b,Tmax_loss_opt));
end

display(sprintf('+---------------------------------------------------+'));

end 