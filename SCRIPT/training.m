% ANDREA BALDINELLI
clear all
close all
clearvars

% PASSIAMO AL LEARNING DEL MODELLO
%%
% selettori in cui si può scegliere quale algoritmo di addestramento
% utilizzare

dataset = readtable('DATASET\dataset_balanced.csv');
show = 1;
gradientDescent = 0;
newtonAlgorithm = 1;
ADMMDistributed = 0;
%% controlliamo che stiamo lavorando con un dataset bilanciato

if show
    figure
    count = groupsummary(dataset,'target');
    bar([0,1],count.GroupCount);
    title("distribuzione nel dataset della variabile target");
    fprintf("\ndiscrepanza tra valori di target:   %d\n",count.GroupCount(1)-count.GroupCount(2)) 
end

%%
% come prima cosa, va creato il dataset di training e di test

matrix = table2array(dataset);
shuffledMatrix = matrix(randperm(size(matrix, 1)), :);

% effettuo lo split 
splitter = floor(height(matrix)*0.2);
Xtest = shuffledMatrix(1:splitter,1:(width(matrix)-1));
Ytest = shuffledMatrix(1:splitter,width(matrix));


% % standardizzo le features, altrimenti ho valori molto grandi
% % uso i dati di training per stimare media e dev std campionaria
Xtraining = shuffledMatrix(splitter+1:height(matrix), 1:(width(matrix)-1));
Ytraining = shuffledMatrix(splitter+1:height(matrix), width(matrix));

sampleMean = mean(Xtraining);
sampleStd = std(Xtraining);


% standard scaling, per evitare pesi molto grandi
Xtraining = (Xtraining - sampleMean)./sampleStd;


% PROVA, introduco la feature dummmy per il parametro 
Xtraining = [ones(height(Xtraining),1),Xtraining];



% VISTO CHE IL DATASET DI TEST VA NORMALIZZATO RISPETTO A MED E DEV STD DEL TRAINING, NON POSSO USARE NORMALIZE
Xtest = (Xtest - sampleMean)./sampleStd;
Xtest = [ones(height(Xtest),1),Xtest];

%%
% CASO 1: AGGIORNAMENTO CON GRADIENT DESCENT
if gradientDescent
n_iter = 1000;
epsilon = 10^-3;

n_parameters = width(Xtraining);
n_obs = height(Xtraining);

WGD = zeros(n_parameters,n_iter);
% matrice in cui vado a salvare le soluzioni di ogni step. per ogni
% algoritmo la nomenclatura delle matrici in gioco sarà: W(nome algoritmo)

step = [10 1 10^-1 10^-2 10^-3];
step = step/n_obs;

JGD = zeros(length(step),n_iter);
% matrice in cui vado a salvare l'andamento della loss function per ogni
% step e per ogni iterazione 

W1GD = randn(n_parameters,1);
WGD(:,1) = W1GD;

k=2;
stop = 0;
WGDopt = zeros(n_parameters,length(step));
% salvo il vettore di parametri ottenuto per ogni step

JGDopt = zeros(1,length(step));
% valore finale loss function, prima dello stop

for t=1:length(step)
    WGD = zeros(n_parameters,n_iter);
    WGD(:,1) = W1GD;
    h = Xtraining*WGD(:,1);
    JGD(:,1) = (-1)*(Ytraining'*log(sigmoid(h)) + (1-Ytraining)'*log(1-sigmoid(h)));
    
    k = 2;
    stop = 0;
    while((k<=n_iter) && (stop==0))
        
        WGD(:,k) = WGD(:,k-1) - (step(t))*(1)*Xtraining'*(sigmoid(Xtraining*WGD(:,k-1))-Ytraining);
        h = Xtraining*WGD(:,k);
        JGD(t,k) = (-1)*(Ytraining'*log(sigmoid(h)) + (1-Ytraining)'*log(1-sigmoid(h))) ;
        if(abs(JGD(t,k) - JGD(t,k-1)) < epsilon)
            stop = 1;
        else
            k = k+1;
        end
    end
    if(k == (n_iter+1))
        fprintf("\nGD:Max iter reached for step-size: %d\n",step(t));
        WGDopt(:,t) = WGD(:,k-1);
        JGDopt(:,t) = JGD(t,k-1);
    else
        fprintf("\nGD:bound reached after %d for step-size: %d\n",k,step(t));
        JGD(t,(k+1:end)) = JGD(t,k);
        WGDopt(:,t) = WGD(:,k);
        JGDopt(:,t) = JGD(t,k);
    end
    fprintf("\nend %d round! let's go\n",t);
end

[JGDbest,best_GD] = min(JGDopt);
wgd_best = WGDopt(:,best_GD);
% seleziono i parametri che mi hanno garantito la minimizzazione della loss
% functon

% non molto rilevante ai fini della classificazione, però si calcola
% l'accuracy
classification_gd_training = sigmoid(Xtraining*wgd_best)>0.5;
accuracy_gd_training = (1.0 /n_obs) * sum(Ytraining == classification_gd_training);


classification_gd_test = sigmoid(Xtest*wgd_best)>0.5;
accuracy_gd_test = (1.0 /height(Xtest)) * sum(Ytest == classification_gd_test);
%% plot andamento loss function
if  show
    figure
    for i=1:length(step)
        plot(1:n_iter,JGD(i,1:n_iter),'-')
        hold on
    end
    ylabel("valore loss function");
    xlabel("iterazioni")
    leg = legend('10','1','0.1','0.01','0.001')
    title(leg,"step-size");
    grid
    title("andamento loss function in funzione di differenti stepisze!");
end
%% andiamo a vedere la convergenza dell'errore (in funzione dello stepsize)
loss_optim = JGD(:,end-1);

error = JGD - JGDbest*ones(height(JGD),1);

if  show
    figure
    for i=1:length(step)
        plot(1:n_iter,error(i,1:n_iter),'-')
        hold on
    end
    ylabel("valore loss function");
    xlabel("iterazioni")
    leg = legend('10','1','0.1','0.01','0.001')
    title(leg,"step-size");
    grid
    title("convergenza errore in funzione di differenti stepisze!");
end

%% proviamo a plottare la RECEIVER OPERATING CHARACTERSITIC RISPETTO TRAINING E TEST

scores_gd_tr = sigmoid(Xtraining*wgd_best);
[Xtr,Ytr,~,AUC_gd_tr] = perfcurve(Ytraining,scores_gd_tr,1);

scores_gd_te = sigmoid(Xtest*wgd_best);
[Xte,Yte,~,AUC_gd_te] = perfcurve(Ytest,scores_gd_te,1);

if show
    figure
    plot(Xtr,Ytr)
    hold on
    plot(Xte,Yte)
    title(["ROC LOGREG x GRADIENT DESCENT";"Si ottiene un AUC score sul TRAINING di "+num2str(AUC_gd_tr)+" e sul TEST di"+num2str(AUC_gd_te)+"!"]);
    legend("ROCxTRAINING","ROCxTEST");
    grid
    ylabel("True positive Rate (detection)");
    xlabel("False positive Rate (false alarm)");
end



end
% 
% 
%
%
% 

%% VERIFICHIAMO LA DIFFERENZA DI VELOCITA' di gd rispetto all'algoritmo di
% newton!

if newtonAlgorithm
n_iter = 1000;
epsilon = 10^-4;
n_parameters = width(Xtraining);
n_obs = height(Xtraining);

step = [2500 1000 500 100 10];
step = step/n_obs;

WN = zeros(n_parameters,n_iter);
JN = zeros(length(step),n_iter);
WN1 = randn(n_parameters,1);
WNopt = zeros(n_parameters,length(step));
JNopt = zeros(1,length(step));


% prima di tutto dobbiamo verificare la derivata seconda della loss
% function sia derivabile
WN(:,1) = WN1;
s = sigmoid(Xtraining*WN(:,1));
diagonal = eye(n_obs).*(s.*(1-s));
Hessian = Xtraining'*(diagonal)*Xtraining;
invH = inv(Hessian);
rank(Hessian)


for t=1:length(step)
    
    WN = zeros(n_parameters,n_iter);
    WN(:,1) = WN1;
    h = Xtraining*WN(:,1);
    JN(t,1) = (-1)*(Ytraining'*log(sigmoid(h)) + (1-Ytraining)'*log(1-sigmoid(h)));
    stop=0;
    k=2;
    
    while((k<=n_iter) && (stop==0))
        
        s = sigmoid(Xtraining*WN(:,k-1));
        diagonal = eye(n_obs).*(s.*(1-s));
        Hessian = Xtraining'*(diagonal)*Xtraining;
        invH = inv(Hessian);
        WN(:,k) = WN(:,k-1) - (step(t))*(invH)*Xtraining'*(s - Ytraining);
        h = sigmoid(Xtraining*WN(:,k));
        JN(t,k) = (-1)*(Ytraining'*log(h) + (1-Ytraining)'*log(1-h));
    
        if(abs(JN(t,k) - JN(t,k-1)) < epsilon)
            stop=1;
        else
            k=k+1;
        end
        
    end
    if(k == (n_iter+1))
        fprintf("\nN:Max iter reached for step-size: %d\n",step(t))
        WNopt(:,t) = WN(:,k-1);
        JNopt(:,t) = JN(t,k-1);
    else
        fprintf("\nN:bound reached after %d, for step-size: %d\n",k,step(t));
        JN(t,(k+1:end)) = JN(t,k);
        WNopt(:,t) = WN(:,k);
        JNopt(:,t) = JN(t,k);
    
    end
    
end
%%
%NEWTON:plot andamento loss function
if  show
    figure
    for i=1:length(step)
        plot(1:n_iter,JN(i,:),'-')
        hold on
    end
    ylabel("loss function");
    xlabel("iterazioni")
    leg = legend('2000','1000','500','100','10')

    title(leg,"step-size");
    grid
    title("NEWTON: andamento loss function in funzione di differenti stepisze!");
end
%% NEWTON valutiamo accuracy 
[JNbest,best_N] = min(JNopt);
wn_best = WNopt(:,best_N);

YNtrainStim = sigmoid(Xtraining*wn_best)>0.5;
accuracyNTraining = (1.0 /n_obs) * sum(Ytraining == YNtrainStim);
YNtestStim = sigmoid(Xtest*wn_best)>0.5;
accuracyNTest = (1.0 /height(Xtest)) * sum(Ytest == YNtestStim);
%%
% VALUTIAMO LA ROC e AUC score

scoresNTR = sigmoid(Xtraining*wn_best);
[XNtr,YNtr,TNtr,AUCNtr] = perfcurve(Ytraining,scoresNTR,1);

scoresNTe = sigmoid(Xtest*wn_best);
[XNte,YNte,TNte,AUCNte] = perfcurve(Ytest,scoresNTe,1);

if show
    figure
    plot(XNtr,YNtr)
    hold on
    title(["ROC LOGREG x NEWTON";"Si ottiene un AUC score TRAINING "+num2str(AUCNtr)+" sul TEST "+num2str(AUCNte)+"!"]);
    plot(XNte,YNte)
    leg = legend('ROCxTRAINING','ROCxTEST');
    grid
    ylabel("True positive Rate (detection)");
    xlabel("False positive Rate (false alarm)");

end

end
%%
% 
%
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
%%
% PROVIAMO A VEDERE I RISULTATI CHE SI OTTENGONO CON UN ADMM Distribuito
% ("CON FUSION CENTER")
% (in cui ho uno splitting rispetto i dati) 

if ADMMDistributed
    n_obs = height(Xtraining);
    n_parameters = width(Xtraining);   
    n_iter = 2000;
    n_agents = 10;
%   questo significa che ogni agente avrà 750 osservazioni con cui lavorare
    
    rho = [0 1 10 50 100];
    n_rho = length(rho);
%   è il parametro di regolarizzazione che mi permette di "convessificare
%   il lagrangiano! Scrivo il tutto in modo da farlo funzionare anche in
%   modalità "multi rho"

    JADMMopt = zeros(n_agents,length(rho));
    WADMMopt = zeros(n_parameters,n_agents,n_rho);
    
    JADMM = zeros(n_agents,n_iter,n_rho);
    WADMM = zeros(n_parameters,n_iter,n_agents);
%   per evitare di lavorare con tensori di dim=4 preferisco azzerarle ad
%   ogni nuovo ciclo del "for dei rho"

    
    
    W1 = randn(n_parameters,1,n_agents);
    U1 = -0.5 + rand(n_parameters,n_agents);
    
    for r=1:n_rho
        WADMM = zeros(n_parameters,n_iter,n_agents);
        WADMM(:,1,:) = W1;
        JADMM(:,1,r) = updateLoss(Xtraining,Ytraining,WADMM(:,1,:),n_agents);
        U = zeros(n_parameters, n_iter,n_agents);
        U(:,1,:) = U1;
        Z = zeros(n_parameters,n_iter);
        Z(:,1) = updateGlobalVariable(WADMM(:,1,:),U(:,1,:));
        
        for k=2:n_iter
            WADMM(:,k,:) = updateLocalVariable(Xtraining,Ytraining,WADMM(:,k-1,:),Z(:,k-1),U(:,k-1,:),n_agents,rho(r));
            Z(:,k) = updateGlobalVariable(WADMM(:,k,:),U(:,k-1,:));
            U(:,k,:) = updateDualVariable(U(:,k-1,:),WADMM(:,k,:),Z(:,k),n_agents);
            JADMM(:,k,r) = updateLoss(Xtraining,Ytraining,WADMM(:,k,:),n_agents);
        end
        
        WADMMopt(:,:,r) = WADMM(:,k,:);
        JADMMopt(:,r) = JADMM(:,k,r);
        fprintf("\nADMM loop completed for rho: %d\n",rho(r));
    end
%%

%%
% plot della loss function totale al variare di rho
if  show
    Jglobal = zeros(n_rho,n_iter);

    figure
    for i=1:n_rho
        Jglobal(i,:) = sum(JADMM(:,:,i));
        plot(1:n_iter,Jglobal(i,:),'-')
        hold on
    end
    ylabel("loss function");
    xlabel("iterazioni")
    leg = legend('0','1','10','50','100')

    title(leg,"step-size");
    grid
    title("ADMM: andamento loss function in funzione di differenti rho!");
end


%%
[JADMMbest,best_admm] = min(sum(JADMMopt));
wadmm_best = WADMMopt(:,best_admm);
classification_admm_training = sigmoid(Xtraining*wadmm_best)>0.5;
accuracy_admm_training = (1.0/n_obs) * sum(Ytraining == classification_admm_training);


classification_admm_test = sigmoid(Xtest*wadmm_best)>0.5;
accuracy_admm_test = (1.0/height(Xtest)) * sum(Ytest == classification_admm_test);
%%
% VALUTIAMO LA ROC e AUC score

scoresADMMTR = sigmoid(Xtraining*wadmm_best);
[XADMMtr,YADMMtr,TADMMtr,AUCADMMtr] = perfcurve(Ytraining,scoresADMMTR,1);

scoresADMMTe = sigmoid(Xtest*wadmm_best);
[XADMMte,YADMMte,TADMMte,AUCADMMte] = perfcurve(Ytest,scoresADMMTe,1);

if show
    
    figure
    plot(XADMMtr,YADMMtr)
    hold on
    title(["ROC LOGREG x ADMM";"Si ottiene un AUC score TRAINING "+num2str(AUCADMMtr)+" sul TEST "+num2str(AUCADMMte)+"!"]);
    plot(XADMMte,YADMMte)
    leg = legend('ROCxTRAINING','ROCxTEST');
    grid
    ylabel("True positive Rate (detection)");
    xlabel("False positive Rate (false alarm)");

end

end
