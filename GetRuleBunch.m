function [RuleBunch, RuleList] = GetRuleBunch(c, RuleNum, D, RuleList) 

    RuleList = zeros(RuleNum, D);
    n = length(c); 
    for i = 1:RuleNum
        while true
            OneRuleX = c(randi(n, 1, D));
            if isempty(RuleList)
                Repeated = false;
            else
                Repeated = any(all(RuleList == OneRuleX, 2));
            end
            if ~Repeated
                RuleList(i,:) = OneRuleX;
                break;
            end
        end
    end
    RuleBunch = RuleList(end-RuleNum+1:end, :);
end
