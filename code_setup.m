function [config] = code_setup(dataset)
%% ======= Read images, labels and other set up
%%% Need changes before running on specific machine


%% =======Specify your paths here ===============
config.database = database;
SAVE_FEATURE_PATH = '../result/';
SAVE_MODEL_PATH = '../result/';

config.save_feature_path = SAVE_FEATURE_PATH; % need add more filename
config.save_feature_path = SAVE_MODEL_PATH; 

%% =======choosing dataset ================
switch dataset
    case 'MSRCV1'
        load('..\MSRCV1.mat');
      
        Ytrn = Y_train;
        Ytst = Y_test;
        
        k = numel(X_train);
        
        for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
        end
        
    case 'NUS_lite'
        
        label_path = '..\NUS_WIDE_lite\NUS-WIDE-SCENE\ground truth\'; 
        train_path = {'..\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_CH.txt',...
                    '..\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_EDH.txt',...
                    '..\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_WT.txt'};
        test_path = {'..\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Test_CH.txt',...
                     '..\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\TEST_EDH.txt',...
                     '..\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Test_WT.txt'};

        data_path = {train_path,test_path};
        k = numel(train_path);

        Ytrn = double(ReadLabel(label_path, 'Train'));

        Ytst = double(ReadLabel(label_path, 'Test'));

        nconcept = size(Ytrn,2);

        for i = 1:k
            X0{i} = [ReadData(train_path{i}),ReadData(test_path{i})];
            kf(i) = size(X0{i},1);
        end
        
end

c = size(Ytrn,2);
ntrn = size(Ytrn,1);
ntst = size(Ytst,1);
n = ntrn+ntst;

for i = 1:k
    X{i} = (X0{i}-repmat(min(X0{i},[],2),1,n))./...
        repmat(max(X0{i},[],2) - min(X0{i},[],2),1,n);
    Xtrn{i} = X{i}(:,1:ntrn);
    Xtst{i} = X{i}(:,ntrn+1:n);
end

config.train_data = Xtrn;
config.test_data = Xtst;

config.train_label = Ytrn;
config.test_label = Ytst;

