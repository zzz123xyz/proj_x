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
%% -------------load real data-----------
data_option = 'Train' ;
label_path = '..\NUS_WIDE_lite\NUS-WIDE-SCENE\ground truth\'; 

data_path = {'..\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_CH.txt'};
          %,'..\NUS_WIDE\Low_Level_Features\Test_Normalized_CH.dat'};
       
Y = double(ReadLabel(label_path, data_option));
        
for i = 1:numel(data_path)
    X0{i} = ReadData(data_path{i});
end
                
load('trained_coef')


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
k = numel(X0);
n = size(Y,1);

for i = 1:k
    X{i} = (X0{i}-repmat(min(X0{i},[],2),1,n))./...
        repmat(max(X0{i},[],2) - min(X0{i},[],2),1,n);
end

%% ------------setting-----------------
t = 2; % the dims of shared view
gamma = 0.5;  %self setting parameter

Zk = [];
Xa = [];
Ms = zeros(n);

I = eye(n);
vec1 = ones(n,1);
H = 1/n*(I-vec1*vec1');

for i = 1:k
    Xa = [Xa;X{i}];
    [kf(i),~] = size(X{i});
end

%% -----------computation --------------
%% compute Z0 and Zi

    %N = I - Y* inv(Y'*Y)* Y';
    N = I - Y / (Y'*Y)* Y';
    for i = 1:k
        %M{i} = H-H*X{i}'*inv((X{i}*H*X{i}'))*X{i}*H;
        M{i} = H-H*X{i}'/(X{i}*H*X{i}')*X{i}*H;
       
        B{i} = M{i} + gamma*N;
        
        [U,S,V]  = svd(B{i});
        nS = sum(diag(S)>0);
        Z_o{i} = U(:,end-kf(i)+1:end);
        
         Ms = Ms + M{i};
    end
    
    A = Ms + gamma * N;
    
    [U,S,V]  = svd(A);
    nS = sum(diag(S)>0);
    Z0_o = U(:,end-t:end);
    
%% compute W

    for i = 1:k
        W_o{i} = (X{i}*H*X{i}')\X{i}*H*[Z0_o,Z_o{i}];
    end

%% compute C
    for i = 1:k
        Zk = [Zk, Z_o{i}];
    end

    C_o = (Y'*Y)\Y'*[Z0_o, Zk];

%% compute b
for i = 1:k
    b_o{i} = 1/n*(W_o{i}'*X{i} - ([Z0_o, Z_o{i}])')*vec1;
end






