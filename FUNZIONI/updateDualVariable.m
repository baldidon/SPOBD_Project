function [outputArg1] = updateDualVariable(U,W,Z,n_agents)
%UPDATEDUALVARIABLE Summary of this function goes here
%   Detailed explanation goes here
    
    Dual = zeros(height(Z),n_agents);
    
    for i=1:n_agents
        Dual(:,i) = U(:,i) + W(:,i) - Z;
    end

outputArg1 = Dual;
end

