function [outputArg1] = sigmoid(inputArg1)
%SIGMOID compute sigmoid function!
outputArg1 = 1./(1+exp(-inputArg1));
end

