function [outputArg1] = updateLocalVariable(Xtraining,Ytraining,W,Z,U,n_agents,rho)
%UPDATESOLUTION Summary of this function goes here
%   Detailed explanation goes here

    solution = zeros(width(Xtraining),n_agents);
    obs = floor(height(Xtraining)/n_agents);
    step = 5/obs;
    for i=1:n_agents
        X = Xtraining(obs*(i-1)+1:obs*i,:);
        Y = Ytraining(obs*(i-1)+1:obs*i,1);
        n_iter = 500;
        epsilon = 10^-4;
        Wgd = zeros(width(X),n_iter);
        Wgd(:,1) = randn(height(W),1);
        Jgd = zeros(1,n_iter);
        Jgd(:,1) = (-1)*(Y'*log(sigmoid(X*Wgd(:,1))) + (1-Y)'*log(1-sigmoid(X*Wgd(:,1))));
        
        j=2;
        stop=0;
        while((j<=n_iter) && (stop ==0))        
            Wgd(:,j) = Wgd(:,j-1) - (step)*(X'*(sigmoid(X*Wgd(:,j-1))-Y)  + rho*(   Wgd(:,j-1) - Z +U(:,i) )    );
            h = X*Wgd(:,j);
            Jgd(:,j) = (-1)*(Y'*log(sigmoid(h)) + (1-Y)'*log(1-sigmoid(h))) ;
            
            if(abs(Jgd(:,j) - Jgd(:,j-1))< epsilon)
                stop = 1;
            else
                j = j+1;
            end
        end
        
        if(j == (n_iter+1))
            solution(:,i) = Wgd(:,j-1);
        else
            solution(:,i) = Wgd(:,j);
        end
    end

outputArg1 = solution;
end

