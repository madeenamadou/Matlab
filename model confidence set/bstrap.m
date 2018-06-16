function Ind_boot = bstrap(n,B)
% -- PURPOSE : This function returns B resamples of n consecutives 
% indexes (starting from 1), with Moving Block Bootstrap (MMB) method

% block size (in days)
l = 2;           
% the number of blocks
k = n/l;                                               

% -- Constructing blocks (columns) of l consecutives indexes
for i = 1:l
    blocks_ind(i,:) = i:n-l+i;                         
end

Ind_boot = zeros(n,B);
for b = 1:B
% -- Picking randomly with replacement k blocks out of n-l+1 blocks
randblock = randi(n-l+1,1,k);

% -- Resampling blocks
Ind = blocks_ind(:,randblock); 

% -- Stacking indexes
Ind_boot(:,b) = Ind (:);                                
end
