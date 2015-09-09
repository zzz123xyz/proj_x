%Input: 
%   l: the number of visual concenpts
%   k: k views (k different kinds of features)
%   kf_i: the number of features in ith view
%       k = numel(kf);
%       m = sum(kf)
%   X{i} : feature matrix of ith view dim: R^{kf_i*n} (kf_i features & n samples)
%   X: feature matrix dim: R^{m*n} (m features & n samples)
%   X_i: feature matrix dim: R^{kf_i*n}
%   Y: labels dim: R^{n*l}
%   W_i: coefficient matrix, dim: R^{kf_i*{t+kf_i}}
%   b_i: the ith bias, dim: R^{{t+kf_i}*1}
%   Z: the shared learned feat dim: R^{n*t}
%   t: the parameters choose from 1 to 3 (it depends)
%   Z_i: the ith learned feat dim: R^{n*kf_i}
%   C: clusters dim: R^{1*(sum(kf)+t)}
%   H: (i-1/n)*Vec1*Vec1'
%Output:
%   W_o: W optimum
%   b_o: 
%   Z0_o: Z optimun
%   Z_o{i}: Zi optimun

clear
clc
%% ------------setting-----------------
addpath(genpath('..\libsvm-3.17\matlab'))
run('..\vlfeat-0.9.18-bin\vlfeat-0.9.18\toolbox\vl_setup.m');
t = 1; % the dims of shared view
gamma = 0.5;  %self setting parameter
train_switch = 0; 
dataset = 'MSRCV1';
coef_savepath = '..\result\';
%% -------------load real data-----------
switch dataset
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
    case 'MSRCV1'
        load('..\MSRCV1.mat');
      
        Ytrn = Y_train;
        Ytst = Y_test;
        
        k = numel(X_train);
        
        for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
        end
end
% %% ------------simulating data-------------
% % m = 500;
% n = 100;
% 
% X0{1} = normrnd(0, 1, 20, n);
% X0{2} = normrnd(0, 2, 30, n);
% X0{3} = normrnd(0, 3, 40, n);
% 
% Y = double(sign(randn(n))>0);
%% ------------normalization-----------
%k = numel(X0);
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

%% -----------training -------------
if train_switch == 1
    [Z0_o, Z_o, W_o, C_o, b_o] = Xtrain(Xtrn, Ytrn, t, k, gamma);
    save([coef_savepath,'_',dataset,'_coef.mat'],'Z0_o','Z_o','W_o','b_o')
else
    load([coef_savepath,'_',dataset,'_coef.mat']);
end

%% -----------testing -------------
Z = [];
for j = 1:ntst
    for i = 1:k
        xtst{i} = Xtst{i}(:,j);
    end
    [z0t, zt] = Xtest(xtst, W_o, b_o, t, k);
    tmp = [z0t, zt];
    Z = [Z; tmp];
    xtst = [];
end


%% ----------predicting -------------
XtrnM = [];
XtstM = [];
ZtrnM = [];
ZtstM = [];

for i = 1:k
    XtrnM = [XtrnM; Xtrn{i}];
end

for i = 1:k
    XtstM = [XtstM; Xtst{i}];
end

ZtrnM = [Z0_o, cell2mat(Z_o)];
ZtstM = Z;

%% -----------svm -------------
switch dataset
    case 'NUS_lite'
        for i = 1:nconcept
            model = svmtrain(Ytrn(:,i), XtrnM');
            [Xlabel(i,:), acc, Xscore(i,:)] = svmpredict...
                (Ytst(:,i), XtstM', model); % test the training data
            Xacc(i) = acc(1);
        end

        for i = 1:nconcept
            model = svmtrain(Ytrn(:,i), ZtrnM);
            [Zlabel(i,:), acc, Zscore(i,:)] = svmpredict...
                (Ytst(:,i), ZtstM, model); % test the training data
            Zacc(i) = acc(1);
        end
        % %% -----------knn -------------
        % 
        % for i = 1: nconcept
        %     Mdl = fitcknn(XtrnM',Ytrn(:,i)');
        %     [Xlabel(i,:),Xscore(i,:)] = predict(Mdl, XtstM');
        % end
        % 
        % for i = 1: nconcept
        %     Mdl = fitcknn(ZtrnM,Ytrn(:,i)');
        %     [Zlabel(i,:),Zscore(i,:)] = predict(Mdl, ZtstM);
        % end
        %% -----------evaluation -----------
        Xap = zeros(1,nconcept);
        Zap = zeros(1,nconcept);
        % for i = 1: nconcept
        %     [XRECALL, XPRECISION, Xinfo{i}] = vl_pr(Ytst(:,i)', Xscore(i,:)); 
        %     Xap(i) = Xinfo{i}.ap;
        % end
        %     Xmap = sum(Xap)/nconcept;
        % 
        % for i = 1: nconcept
        %     [ZRECALL, ZPRECISION, Zinfo{i}] = vl_pr(Ytst(:,i)', Zscore(i,:)); 
        %     Zap(i) = Zinfo{i}.ap;
        % end

        %% ---------------------------------------------
        % for i = 1: nconcept
        %     %Xap(i) = computeAP(Xlabel(i,:), Xscore(i,:), Ytst(:,i)'); 
        %     Xap(i) = computeAP(Ytst(:,i)', Xscore(i,:), 1); 
        % end
        %     Xmap = sum(Xap)/nconcept;
        % 
        % for i = 1: nconcept
        %     Zap(i) = computeAP(Ytst(:,i)', Zscore(i,:), 1); 
        % end
        %     Zmap = sum(Zap)/nconcept;

        %% ------------------------------------------------

        for i = 1: nconcept
            [~,~,~,Xap(i)] = perfcurve(Ytst(:,i)', Xscore(i,:), 1, 'xCrit', 'reca', 'yCrit', 'prec');
        end
            Xmap = sum(Xap)/nconcept;

        for i = 1: nconcept
            [~,~,~,Zap(i)] = perfcurve(Ytst(:,i)', Zscore(i,:), 1, 'xCrit', 'reca', 'yCrit', 'prec');
        end
            Zmap = sum(Zap)/nconcept;    
    
    case 'MSRCV1'
        tmp1 = [1:c]';
        Ytrn_l = Ytrn * tmp1;
        Ytst_l = Ytst * tmp1;
        
            model = svmtrain(Ytrn_l, XtrnM');
            [Xlabel, Xacc, Xscore] = svmpredict...
                (Ytst_l, XtstM', model); % test the training data

            model = svmtrain(Ytrn_l, ZtrnM);
            [Zlabel, Zacc, Zscore] = svmpredict...
                (Ytst_l, ZtstM, model); % test the training data
            
            disp(['original feat acc:',num2str(Xacc(1)),'%'])
            disp(['new feat acc:',num2str(Zacc(1)),'%'])
            

%         %% -----------evaluation -----------
%         [~,~,~,Xap] = perfcurve(Ytst', Xscore, 1, 'xCrit', 'reca', 'yCrit', 'prec');
%         [~,~,~,Zap] = perfcurve(Ytst', Zscore, 1, 'xCrit', 'reca', 'yCrit', 'prec');
end