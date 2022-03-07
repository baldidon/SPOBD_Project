function [Z] = updateGlobalVariable(W,U)
%UPDATEGLOBALVARIABLE Summary of this function goes here
%   Detailed explanation goes here
    Z = zeros(height(W),1);
    u_mean = zeros(height(W),1);
    w_mean = zeros(height(W),1);
    
    for i=1:width(W)
        u_mean = u_mean + U(:,i);
        w_mean = w_mean + W(:,i);
    end
    Z = u_mean + w_mean;
    Z = (Z/width(W));
end

