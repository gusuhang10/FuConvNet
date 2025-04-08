function [result,wt] = cal_fuzzy_convol(featuremap,W,RuleBunch,convDim)
    R = size(RuleBunch,1);
    [ker_1,ker_2,~,~] = size(W);
    Wt = reshape(W',R,1);
    datamat = im2col(featuremap',[ker_2 ker_1],'sliding');
    datamat = datamat';
    %D = pdist2(datamat, RuleBunch, 'squaredeuclidean');
    RuleBunch = permute(RuleBunch, [2 1]);
    D = datamat * RuleBunch;
    wt = exp(D);
    wt2 = sum(wt,2) ;
    wt = wt./repmat(wt2,1,R); 
    wt = wt + datamat;
    result = wt * Wt; 
    result = reshape(result,convDim,convDim)';
end