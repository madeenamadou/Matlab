function  mcs()
% -- PURPOSE : This function is responsible of implementing the procedure for
% the test based on Model Confidence Set until first 'acceptance'
% and returns the decision of the test, statistics and p-values

clc;clear;

main();

ind1 = [1 2 3 4 5 6 7 8 9];
ind2 = [1 2 3 4 5 6 7 8 9];
ind3 = [1 2 3 4 5 6 7 8 9];


for it=1:1000

[H0_MAE, H0_QLIKE, H0_LOSS_OPT, ER_MAE, ER_QLIKE, ER_LOSS_OPT] = mcs_iter(ind1, ind2, ind3, it);

if H0_MAE==0 && size(ind1,2)~=1
tmp1 = find(ind1~=ER_MAE);
ind1 = ind1(tmp1);
end
if H0_QLIKE==0 && size(ind2,2)~=1
tmp2 = find(ind2~=ER_QLIKE);
ind2 = ind2(tmp2);
end
if H0_LOSS_OPT==0 && size(ind3,2)~=1
tmp3 = find(ind3~=ER_LOSS_OPT);
ind3 = ind3(tmp3);
end


if H0_MAE==1 && H0_QLIKE==1 && H0_LOSS_OPT==1, 
display(sprintf('H0 is now accepted based on all three loss-functions.'));
display(sprintf('The procedure has come to the end.'));
    break;
else
    if size(ind1,2)==1 || size(ind2,2)==1 || size(ind3,2)==1
    break;
    end
end


end

display(sprintf('+---------------------------------------------------+'));
display(sprintf('The 0.9 Model Confidence Set is...'));
display(sprintf('+-------+'));
display(sprintf('|  MAE  |'));
display(sprintf('+-------+'));
for i=1:size(ind1,2)
    display(sprintf('Model[%u]',ind1(i)));
end
display(sprintf('---------'));
display(sprintf('+-------+'));
display(sprintf('| QLIKE |'));
display(sprintf('+-------+'));
for i=1:size(ind2,2)
    display(sprintf('Model[%u]',ind2(i)));
end
display(sprintf('---------'));
display(sprintf('+-------------------------+'));
display(sprintf('| LOSS AFTER OPTION TRADE |'));
display(sprintf('+-------------------------+'));
for i=1:size(ind3,2)
    display(sprintf('Model[%u]',ind3(i)));
end
display(sprintf('---------'));

end 