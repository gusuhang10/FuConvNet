function [c,sig]=inimemfun(featuremap,feamap_w,feamap_h,numfuzzypartion)
    m=1/(numfuzzypartion-1);
    c = cell(1,numfuzzypartion);
    sig = cell(1, numfuzzypartion);
    for i=1:numfuzzypartion        
        c{i}=m*(i-1)*ones(feamap_w,feamap_h);
        sig{i}=ones(feamap_w,feamap_h);
    end
end
