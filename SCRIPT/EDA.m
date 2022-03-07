% ANDREA BALDINELLI

% la tesina riguarda l'applicazione di svm e logreg split among data and
% features. Valutando le prestazioni
%%
dataset = readtable('DATASET\healthcare-dataset-stroke-data.csv');
show=0;
%%
% come un buon progetto di machine learning  impone, devo fare un po' di
% EDA

% CONTROLLIAMO SE CI SONO VALORI NULLI
nullValues = double(ismissing(dataset));
missingvalues = sum(nullValues, 1);

if (show)
    X = categorical(dataset.Properties.VariableNames);
    X = reordercats(X,dataset.Properties.VariableNames);

    bar(X,missingvalues)
    title("Valori nulli organizzati per feature")
end

% a questo proposito, possiamo andare ad eliminare quei valori,
dataset = rmmissing(dataset);

%% osservando i domini di definizione delle features, alcune sono categoriche. Vanno trasformate in numeri per 
% poter garantire la gestibilità da un algoritmo!

%% DISTRIBUZIONE VARIABILE DI USCITA
if show
    count = groupsummary(dataset,"stroke");
    labels = ["stroke yes","stroke no"];
    pie(count.GroupCount, labels)  
    title("distribuzione variabile di uscita");

end

% Molto sbilanciato questo dataset, è pertanto necessario fare un
% rebalancing finale (con tecniche come SMOTE) per garantire uno stimatore
% con una buona variance di stima.
%%

% EDA gender
if (show)
    X = categorical({'Female','Male','Other'});
    count = groupcounts(dataset,{'gender'});
    bar(X,count.Percent)
    title("Valori feature gender organizzati per (in percentuale)");
end

% visto tale distribuzione, posso trasformare la variabile in is_Female e
% vale 0,1 (è meglio non usare 0,1,2 perchè in quel modo si rischia di dare
% un peso all'essere un genere rispetto ad un altro)

dataset.('isFemale') = strcmp(dataset.gender,'Female');

% possiamo droppare la colonna gender (ne approfitto anche per eliminare
% id)
dataset = removevars(dataset, {'gender','id'});

%% 
% sulla variabile age, per ora meglio non fare nulla, visto che non è detto
% che sia un male averla come variabile numerica (e non divisa per
% categorie di persone, lo vedremo dopo)

% controlliamo solo la presenza di valori astrusi
strangeValues = dataset.age>= 100 & dataset.age <= 1;

dataset.age = ceil(dataset.age);
sum(strangeValues)
% sembrerebbe di no;
if(show)
    histogram(dataset.age)
    title("distribuzione età persone dataset");
end


%%
% possiamo fare un analisi che riguarda la correlazione di ischemia con
% l'età 

dataset.ageCategory=zeros(height(dataset),1);

% definiamo la seguente operazione
% 1:young 2:adult 3:anziani 4: molto anziani

for i=1:height(dataset)
    if dataset.age(i) <= 30
        dataset.ageCategory(i) = 1;
    elseif ((dataset.age(i) >30) && (dataset.age(i)<=60)) 
        dataset.ageCategory(i) = 2;
    elseif ((dataset.age(i) >60) && (dataset.age(i)<=75))
        dataset.ageCategory(i) = 3;
    else
        dataset.ageCategory(i) = 4;
    end             
end


if show
    g = groupsummary(dataset,"ageCategory","sum","stroke");
    X = categorical({'Young','Adult','elder','very_elder'});
    bar(X,g.("sum_stroke"))
    title("distribuzione di ischemie in funzione della categoria di persona");
end


% a questo punto me ne droppo una delle due e posso verificare come
% migliora il tutto


%% HEART_DISEASE

% controlliamo la presenza di valori astrusi
% sembrerebbe di no;
if(show)
    count = groupcounts(dataset,{'heart_disease'});
    bar(count.Percent)
end
%%
% SMOKING STATUS

% osserviamo la distribuzione dei valori di SMOKING STATUS all'interno del
% dataset

% count = groupcounts(dataset, {'smoking_status'});
count = groupsummary(dataset, "smoking_status");

if(show)
    subplot(1,2,1);
    X = categorical({'Unknown';'formerly smoked';'never smoked';'smokes'});
    bar(X,count.GroupCount)
    title("Distribuzione della variabile SMOKING STATUS nel dataset");
end

% rispetto al genere, qui posso dare un significato di "peso" in funzione
% di quanto si "è fumatori". Vediamo la distribuzione della variabile di
% uscita in funzione di SMOKING STATUS

if show
    subplot(1,2,2);
    g = groupsummary(dataset,"smoking_status","sum","stroke");
    X = categorical({'Unknown';'formerly smoked';'never smoked';'smokes'});
    bar(X,g.("sum_stroke"))
    title("distribuzione di ischemie in funzione della categoria di fumatori");
end

% attenzione. Nonostante quello che sia il senso comune, in questo set di
% dati non abbiamo una forte correlazione fra dati di fumatori (o ex
% fumatori) e ischemie


% potrebbe aver senso mostrare la relazione che c'è tra la feature
% "heart_disease" e lo smoking status. In modo tale (eventualmente) di
% lasciarne solo una!

if show
    g = groupsummary(dataset,"smoking_status","sum","heart_disease");
    X = categorical({'Unknown';'formerly smoked';'never smoked';'smokes'});
    bar(X,g.("sum_heart_disease"))
    title("distribuzione di problemi di cuorein funzione della categoria di fumatori");
end

% anche per via del dataset sbilanciato verso il valore "never smoked",
% anche qui non c'è evidenza di "correlazione" con uno dei valori della variabile
% never smoked. 
%%
% faccio una variabile smoke
dataset.ever_smoked = (strcmp(dataset.smoking_status,'formerly smoked') | strcmp(dataset.smoking_status, 'smokes'));


% e quindi per ora droppo la colonna smoking status
dataset = removevars(dataset, {'smoking_status'});
%%

% esploriamo la variabile 'ever_married'.

if show
    count = groupsummary(dataset,"ever_married");
    labels = ["yes","no"];
    pie(count.GroupCount, labels)  
    title("distribuzione variabile ever married");
end

% trasformiamola in una variabile numerica
dataset.ever_married = strcmp(dataset.ever_married,'Yes');
%%
if show
    subplot(1,2,1)
    count = groupsummary(dataset,"ever_married", "sum", "stroke");
    X = categorical({'no','yes'});
    bar(X,count.("sum_stroke"))
    title("distribuzione delle ischemie in funzione dell'essere sposati o meno");
    subplot(1,2,2)
    labels = {"no","yes"};
    pie(count.GroupCount,labels)  
end

% per ora viene lasciata come feature, ma è sicuramente a rischio di
% eliminazione

%% esploriamo WORK TYPE

% osserviamo quanti valori unici ci sono e la loro presenza nel dataset
dist = groupsummary(dataset,"work_type");

% non è molto descrittiva, ad onor del vero
if show
    subplot(1,2,1)
    count = groupsummary(dataset,"work_type", "sum", "stroke");
    X = categorical({'Govt job';'Never worked';'Private';'Self-employed';'children'});
    bar(X,count.("sum_stroke"))
    title("distribuzione delle ischemie in funzione del lavoro");
    subplot(1,2,2)
    labels = {'lavoro pubblico';'Never worked';'Private';'Self-employed';'children'};
    pie(count.GroupCount)  
    legend(labels)
end


% dummmiziamo con una buona riserva sulla sua efficacia
% dataset.private_job = strcmp(dataset.work_type, 'Private');
% dataset.public_job  = strcmp(dataset.work_type, 'Govt_job');
dataset.is_working = strcmp(dataset.work_type, 'Govt_job') | strcmp(dataset.work_type, 'Private') | strcmp(dataset.work_type, 'Self-employed');
% dataset.never_worked = (strcmp(dataset.work_type, 'Never worked') | strcmp(dataset.work_type, 'children'));


% rimuoviamo work-type
dataset = removevars(dataset, {'work_type'});

%% RESIDENCE TYPE

% non è molto descrittiva, ad onor del vero
if show
    subplot(1,2,1)
    count = groupsummary(dataset,"Residence_type", "sum", "stroke");
    X = categorical({'Urban'; 'Rural'});
    bar(X,count.("sum_stroke"))
    title("distribuzione delle ischemie in funzione del tipo di luogo di residenza");
    subplot(1,2,2)
    labels = {'Urban';'Rural'};
    pie(count.GroupCount)  
    legend(labels)
end

% questa se ne va veramente, in quanto è effettivamente (osservando i due
% plot) che è uniformemente distribuita all'interno del dataset, e lo è
% anche in riferimento alla nostra variabile di uscita!


dataset = removevars(dataset,{'Residence_type'});
%%

% osserviamo il livello di glucosio medio
if show
    subplot(1,2,1)
    histogram(dataset.avg_glucose_level)
    title("distribuzione del glucosio medio nel sangue")

    % per quanto ci sono livelli di glicemia abbastanza elevati, anche sopra
    % 200 non sono outlier perchè descrivono una situazione di forte diabete

    % per curiosità, rapportiamo tale dato all'istogramma del indice di massa
    % corporea
    subplot(1,2,2)
    histogram(dataset.bmi)
    title("distribuzione della variabile bmi")
end
%%
close all

if show
    boxplot(dataset.bmi)
    title("distribuzione bmi nel dataset")
end
% visto che già oltre 40 si parla di obesità di 3o livello, sapere che ci
% sono persone che sembrerebbero avere bmi ben oltre 60, mi sembra irreale.
% Anche il boxplot suggerisce che sono outlier. Pertanto mi sembra


% rimuovo le righe con bmi > 60
% toDelete = Tnew.Age < 30;
% Tnew(toDelete,:) = [];

toDelete = dataset.bmi > 60;
dataset(toDelete,:) = [];

height(dataset)
% ho eliminato una decina di righe. Eventualmente le riaggiungo

%%

% sembrerebbe che abbiamo finito! pertanto rimuovo l'ultima colonna che è
% Age category

dataset = removevars(dataset,{'ageCategory'});


%%
% osserviamo la correlazione delle features nel dataset

% prima di tutto riporto in double tutte le colonne binarie che ho creato
% (che altrimenti) sono di tipo logical
dataset.isFemale= double(dataset.isFemale);
dataset.ever_smoked= double(dataset.ever_smoked);
dataset.is_working= double(dataset.is_working);
% dataset.public_job= double(dataset.public_job);
% dataset.never_worked= double(dataset.never_worked);

classification = dataset.stroke;
dataset = removevars(dataset, {'stroke'});
matrix = table2array(dataset);

dataset.stroke = classification;
rho = corrcoef(dataset.Variables);
if show
    h = heatmap(abs(rho),'Colormap',jet,'title','correlazione fra le features')
end


%%

% export della matrice delle osservazioni e della variabile di uscita
% matrix e classification

writetable(dataset,'DATASET/dataset.csv');

fprintf("per continuare, eseguire lo script in python che permette di ribilanciare il dataset!\n");