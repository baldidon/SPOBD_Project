function [out] = updateLoss(Xtraining,Ytraining,W,n_agents)
%UPDATELOSS calcola l'update delle loss locali dando in pasto soltanto 

    loss = zeros(n_agents,1);
    obs = floor(height(Xtraining)/n_agents);
    for i=1:n_agents
        X = Xtraining(obs*(i-1)+1:obs*i,:);
        Y = Ytraining(obs*(i-1)+1:obs*i,1);
        h = sigmoid(X*W(:,i));
        loss(i) = -1*((Y'*log(h)) + (1-Y)'*log(1-h));
    end

out = loss;
end

