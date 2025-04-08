function c=inimemfun_input(featuremap,numfuzzypartition)
    c=zeros(1,numfuzzypartition);
    m = 1/(numfuzzypartition-1);
    for i=1:numfuzzypartition        
        c(i) = m*(i-1);
    end
end