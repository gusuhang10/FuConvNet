function Wg = convn_fuzzy(activationsP,DeltaC,filterDim) 
    [C1,C2] = size(DeltaC);
    DeltaCC = reshape(DeltaC',C1*C2,1);
    Wt = activationsP'*DeltaCC;
    Wg = reshape(Wt,filterDim,filterDim)';
end