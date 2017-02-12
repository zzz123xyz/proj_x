
dataset_vec = {'MSRCV1','NUS_lite'};
method_vec = {'Xmethod''RHLM'};

Xpara.t = 1;
Xpara.k
Xpara.gamma =0.5;

for i = 1:numel(dataset_vec)
  dataset = dataset_vec(i);
  config = code_setup(dataset_vec{i});
  
  switch dataset
      case 'RHLM'  
        %% =======RHLM==============
          addpath('..\RHLM_code');
            %U = ********
            semisupervised(config.train_data, config.test_data, U,...
                            config.train_label, config.test_label)

      case 'Xmethod'
        %% =======Method X==========
            if train_switch == 1
                [Z0_o, Z_o, W_o, C_o, b_o] = Xtrain(config.train_data, config.train_label...
                , Xpara);
                save([coef_savepath,'_',dataset,'_coef.mat'],'Z0_o','Z_o','W_o','b_o')
            else
                load([coef_savepath,'_',dataset,'_coef.mat']);
            end

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
      case 'Other'      
  end
  
%% evaluation 
  switch dataset
    %% ==============MAP==================  
    case 'MAP'
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
       
        Xap = zeros(1,nconcept);
        Zap = zeros(1,nconcept);
        

        %% ------------------------------------------------

        for i = 1: nconcept
            [~,~,~,Xap(i)] = perfcurve(Ytst(:,i)', Xscore(i,:), 1, 'xCrit', 'reca', 'yCrit', 'prec');
        end
            Xmap = sum(Xap)/nconcept;

        for i = 1: nconcept
            [~,~,~,Zap(i)] = perfcurve(Ytst(:,i)', Zscore(i,:), 1, 'xCrit', 'reca', 'yCrit', 'prec');
        end
            Zmap = sum(Zap)/nconcept;  
            
     %% =============ACC======================
     case 'ACC'
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
end