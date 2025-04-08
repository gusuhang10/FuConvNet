%1: Gaussian MF; 2: Sigmf;
function bb=calmemfun(aa,c,sig,numfuzzypartion,type_mf)
    if type_mf == 1
        bb = cell(1,numfuzzypartion);
        for i=1:numfuzzypartion
            bb{i}=exp(-0.5 * ((aa-c{i})./sig{i}).^2);
        end
    elseif type_mf == 2
        bb = cell(1,numfuzzypartion);
        for i=1:numfuzzypartion
            bb{i}=1./(1+exp(-2*(aa-c{i})));           
        end
    end