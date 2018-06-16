function main()
% -- PURPOSE : This main function imports data on SP500, do the filtering,
% do the computation of estimators and loss-functions, do the bootstrap resampling, 
% and save the data needed for next steps

clc;clear;

disp('Vous devez choisir le fichier de données HF_SPX.mat');
disp('Copiez le fichier HF_SPX.mat dans le répertoire MATLAB puis appuyez sur une touche pour continuer...');
filename = uigetfile;
if isequal(filename,0)
    disp('Vous devez choisir le fichier de données HF_SPX.mat')
    disp('Copiez le fichier HF_SPX.mat dans le répertoire MATLAB puis appuyez sur une touche pour continuer...');
    pause
    return
end

clc;

% -- Importing data
file = load(filename);
data = file.spx;
nhf = find(data(:,3)==-1,1) - 1;
spx = data(1:nhf,3);
save ('spx','spx');

load spx
nhf = size(spx,1);

% -- Filtering out for days with more than 30 fake 5-min zero returns
d = 0;
sev = 30;
for j = 1:nhf/79
    t0 = j*79-78 ; tf = j*79;
    temp1(:,j) = spx(t0:tf);
    temp2(:,j) = diff(log(temp1(:,j)));
    temp3 = find(temp2(:,j)==0);

    if size(temp3,1) < sev
    d = d+1;
    t0 = d*79-78 ; tf = d*79;
    temp4(:,d) = temp2(:,j);
    end
end

% -- Filtering out for days with significant 1st order 5-min returns AC,
% -- and computing variables
k = 0;
for i=1:size(temp4,2)
    temp5=autocorr(temp4(:,i),1);
    [temp6, temp7]=lbqtest(temp4(:,i),'lags',1);
    if temp5(2)<0 && temp7 <=0.05
    else
    k = k+1;
    r_hf(:,k)=temp4(:,i);
    t0 = k*79-78 ; tf = k*79;
    r_d(k) = log(spx(tf))-log(spx(t0));

    rv_d(k) = sum((r_hf(:,k)).^2);
    pos = (r_hf(:,k)>0);
    neg = (r_hf(:,k)<0);
    rspos_d(k) = sum((r_hf(pos,k)).^2);
    rsneg_d(k) = sum((r_hf(neg,k)).^2);
    dJ(k) = rspos_d(k)-rsneg_d(k);
         
    for t=2:77    
    temp8(t) = median(abs(r_hf(t-1:t+1,k)));
    end
    medrv_d(k) = (pi/(6-4*sqrt(3)+pi))*(78/76)*sum(temp8.^2);
      
    end
    
 end

for l=6:size(rv_d,2)
    rv_w(l-5)=(1/5)*sum(rv_d(l-5:l-1));
    rspos_w(l-5)=(1/5)*sum(rspos_d(l-5:l-1));
    rsneg_w(l-5)=(1/5)*sum(rsneg_d(l-5:l-1));
end

for l=21:size(rv_d,2)
    rv_m(l-20)=(1/20)*sum(rv_d(l-20:l-1));
    rspos_m(l-20)=(1/20)*sum(rspos_d(l-20:l-1));
    rsneg_m(l-20)=(1/20)*sum(rsneg_d(l-20:l-1));
end

lev = (r_d<0)'.*rv_d';
dJ_pos = dJ.*(dJ>0);
dJ_neg = dJ.*(dJ<0);

% -- Saving Y and X for Regressions and the MCS procedure
Y = rv_d(22:end)';
X0 = [rv_d(21:end-1)' rv_w(16:end-1)' rv_m(1:end-1)'];
X1 = [rv_d(21:end-1)' lev(21:end-1) rv_w(16:end-1)' rv_m(1:end-1)'];
X2 = [rspos_d(21:end-1)' rv_w(16:end-1)' rv_m(1:end-1)'];
X3 = [rsneg_d(21:end-1)' rv_w(16:end-1)' rv_m(1:end-1)'];
X4 = [rspos_d(21:end-1)' rsneg_d(21:end-1)' rv_w(16:end-1)' rv_m(1:end-1)'];
X5 = [medrv_d(21:end-1)' rv_w(16:end-1)' rv_m(1:end-1)'];
X6 = [medrv_d(21:end-1)' dJ(21:end-1)' rv_w(16:end-1)' rv_m(1:end-1)'];
X7 = [medrv_d(21:end-1)' dJ_neg(21:end-1)' rv_w(16:end-1)' rv_m(1:end-1)'];
X8 = [medrv_d(21:end-1)' dJ_pos(21:end-1)' dJ_neg(21:end-1)' rv_w(16:end-1)' rv_m(1:end-1)'];

save ('var','Y','X0','X1','X2','X3','X4','X5','X6','X7', 'X8');

% -- Getting loss-functions series L(t) for all models
[qlike0, mae0, Popt0] = loss_comput(Y,X0);
[qlike1, mae1, Popt1] = loss_comput(Y,X1);
[qlike2, mae2, Popt2] = loss_comput(Y,X2);
[qlike3, mae3, Popt3] = loss_comput(Y,X3);
[qlike4, mae4, Popt4] = loss_comput(Y,X4);
[qlike5, mae5, Popt5] = loss_comput(Y,X5);
[qlike6, mae6, Popt6] = loss_comput(Y,X6);
[qlike7, mae7, Popt7] = loss_comput(Y,X7);
[qlike8, mae8, Popt8] = loss_comput(Y,X8);

MAE = [mae0 mae1 mae2 mae3 mae4 mae5 mae6 mae7 mae8];                            % 500x9 matrix
QLIKE = [qlike0 qlike1 qlike2 qlike3 qlike4 qlike5 qlike6 qlike7 qlike8];        % 500x9 matrix
POPT = [Popt0 Popt1 Popt2 Popt3 Popt4 Popt5 Popt6 Popt7 Popt8]; 

% -- Computing d(t) and d_bar the loss-statistics average for all model
di_mae_t = bsxfun(@plus,MAE,-mean(MAE,2));                               % 500x1 matrix
di_qlike_t = bsxfun(@plus,QLIKE,-mean(QLIKE,2));                         % 500x1 matrix
di_mae = mean(di_mae_t,1)';                                              % 9x1 matrix
di_qlike = mean(di_qlike_t,1)';                                          % 9x1 matrix

r = r_d(22:end)';

for j=1:size(POPT,2)
    POPT_trans(:,:,j) = bsxfun(@plus, 0.5*POPT(:,j), 0.5*POPT(:,find(1:9 ~= j)));
    buy = bsxfun(@le,POPT_trans(:,:,j),POPT(:,j));
    sell = bsxfun(@ge,POPT_trans(:,:,j),POPT(:,j));

    profit_buy(:,:,j) = bsxfun(@times, buy,bsxfun(@plus,abs(r(3001:3500)) - r(3001:3500).*POPT(:,j), - 2*POPT(:,j)));
    profit_sell(:,:,j) = bsxfun(@times, sell,bsxfun(@plus,- abs(r(3001:3500)) + r(3001:3500).*POPT(:,j) , 2*POPT(:,j)));
    profit_ij (:,:,j) = bsxfun(@plus,profit_buy(:,:,j),profit_sell(:,:,j));
    profit(:,j) = mean(profit_ij(:,:,j),2);
    LOSS_OPT(:,j) = -profit(:,j);
    
end

LOSS_OPT_MEAN = mean(LOSS_OPT,1)';

di_loss_opt_t = bsxfun(@plus,LOSS_OPT,-mean(LOSS_OPT,2));                           % 500x1 matrix
di_loss_opt = mean(di_loss_opt_t,1)';                                               % 9x1 matrix

% -- Number of bootstrap resamples
B = 500;

% -- Re-initializing the random number generator as if we restarted Matlab
rng('default'); 

% -- Getting indexes resamples
Ind_boot = bstrap(3000,B);

% -- Getting bootstrap loss-functions series for all models
for b = 1:B
        [qlike0_b(:,b), mae0_b(:,b), Popt0_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X0(Ind_boot(:,b),:),Y,X0);       
        [qlike1_b(:,b), mae1_b(:,b), Popt1_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X1(Ind_boot(:,b),:),Y,X1);
        [qlike2_b(:,b), mae2_b(:,b), Popt2_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X2(Ind_boot(:,b),:),Y,X2);
        [qlike3_b(:,b), mae3_b(:,b), Popt3_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X3(Ind_boot(:,b),:),Y,X3);
        [qlike4_b(:,b), mae4_b(:,b), Popt4_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X4(Ind_boot(:,b),:),Y,X4);
        [qlike5_b(:,b), mae5_b(:,b), Popt5_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X5(Ind_boot(:,b),:),Y,X5);
        [qlike6_b(:,b), mae6_b(:,b), Popt6_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X6(Ind_boot(:,b),:),Y,X6);
        [qlike7_b(:,b), mae7_b(:,b), Popt7_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X7(Ind_boot(:,b),:),Y,X7);
        [qlike8_b(:,b), mae8_b(:,b), Popt8_b(:,b)] = loss_comput_b(Y(Ind_boot(:,b)),X8(Ind_boot(:,b),:),Y,X8);
        
    
        % -- For each bootstrap resample, computing the loss_statistics average of all model 
        % MAE_b (and QLIKE_b) is 500x9xB matrix
        % di_mae_t_b (and di_qlike_t_b) is 500x9xB matrix
        % di_mae_b (and di_qlike_b) is 9xB matrix
        
        MAE_b (:,:,b) = [mae0_b(:,b) mae1_b(:,b) mae2_b(:,b) mae3_b(:,b) mae4_b(:,b) mae5_b(:,b) mae6_b(:,b) mae7_b(:,b) mae8_b(:,b)];
        di_mae_t_b(:,:,b) = bsxfun(@plus,MAE_b (:,:,b),- mean(MAE_b (:,:,b),2));
        QLIKE_b (:,:,b) = [qlike0_b(:,b) qlike1_b(:,b) qlike2_b(:,b) qlike3_b(:,b) qlike4_b(:,b) qlike5_b(:,b) qlike6_b(:,b) qlike7_b(:,b) qlike8_b(:,b)];
        di_qlike_t_b(:,:,b) = bsxfun(@plus,QLIKE_b (:,:,b),- mean(QLIKE_b (:,:,b),2));
        di_mae_b(:,b) = mean(di_mae_t_b(:,:,b),1)';
        di_qlike_b(:,b) = mean(di_qlike_t_b(:,:,b),1)';
        
        POPT_b(:,:,b) = [Popt0_b(:,b) Popt1_b(:,b) Popt2_b(:,b) Popt3_b(:,b) Popt4_b(:,b) Popt5_b(:,b) Popt6_b(:,b) Popt7_b(:,b) Popt8_b(:,b)]; 
        for j=1:size(POPT,2)
        POPT_trans_b(:,:,j) = bsxfun(@plus, 0.5*POPT_b(:,j,b), 0.5*POPT_b(:,find(1:9 ~= j),b));
        buy_b = bsxfun(@le,POPT_trans_b(:,:,j),POPT_b(:,j,b));
        sell_b = bsxfun(@ge,POPT_trans_b(:,:,j),POPT_b(:,j,b));
             
        profit_buy_b(:,:,j) = bsxfun(@times, buy_b,bsxfun(@plus,abs(r(3001:3500)) - r(3001:3500).*POPT_b(:,j,b) , - 2*POPT_b(:,j,b)));
        profit_sell_b(:,:,j) = bsxfun(@times, sell_b,bsxfun(@plus,- abs(r(3001:3500)) + r(3001:3500).*POPT_b(:,j,b) , 2*POPT_b(:,j,b)));
        profit_ij_b (:,:,j) = bsxfun(@plus,profit_buy_b(:,:,j),profit_sell_b(:,:,j));
        profit_b(:,j) = mean(profit_ij_b(:,:,j),2);
        LOSS_OPT_b(:,j,b) = -profit_b(:,j);
        end
        
        di_loss_opt_t_b(:,:,b) = bsxfun(@plus,LOSS_OPT_b(:,:,b),-mean(LOSS_OPT_b(:,:,b),2));                               % nx1 matrix
        di_loss_opt_b(:,b) = mean(di_loss_opt_t_b(:,:,b),1)';   
 
        
end

% -- Saving data for the iteration of the procedure for surviving models
save('mcs','MAE','QLIKE','LOSS_OPT','MAE_b','QLIKE_b', 'LOSS_OPT_b');

end



