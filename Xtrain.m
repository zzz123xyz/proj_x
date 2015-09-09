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
%   W_i: coefficient matrix, dim: R^{kf_i*c}
%   b_i: the ith bias, dim: R^{{c}*1}
%   Z: the shared learned feat dim: R^{n*t}
%   t: the parameters choose from 1 to 3 (it depends)
%   Z_i: the ith learned feat dim: R^{n*{c-t}}
%   C: clusters dim: R^{c*(t+k*(t-c))}
%   H: (i-1/n)*Vec1*Vec1'
%Output:
%   W_o: W optimum
%   b_o: 
%   Z0_o: Z optimun
%   Z_o{i}: Zi optimun



function [Z0_o, Z_o, W_o, C_o, b_o] = Xtrain(Xtrn, Ytrn, t, k, gamma)

%% -------------initialization -----------
c = size(Ytrn, 2);
ntrn = size(Ytrn, 1);
Zk = [];
Xa = [];
Ms = zeros(ntrn);

I = eye(ntrn);
vec1 = ones(ntrn,1);
H = 1/ntrn*(I-vec1*vec1');

for i = 1:k
    Xa = [Xa;Xtrn{i}];
    [kf(i),~] = size(Xtrn{i});
end

%% -----------computation --------------
%% compute Z0 and Zi

    %N = I - Y* inv(Y'*Y)* Y';
    N = I - Ytrn / (Ytrn'*Ytrn)* Ytrn';
    for i = 1:k
        %M{i} = H-H*X{i}'*inv((X{i}*H*X{i}'))*X{i}*H;
        
        %M{i} = H-H*Xtrn{i}'/(Xtrn{i}*H*Xtrn{i}')*Xtrn{i}*H;
        M{i} = H-H*Xtrn{i}'*pinv(Xtrn{i}*H*Xtrn{i}')*Xtrn{i}*H;
        
        B{i} = M{i} + gamma*N;
        
        [U,S,V]  = svd(B{i});
        nS = sum(diag(S)>0);
        %Z_o{i} = U(:,end-kf(i)+1:end);
        Z_o{i} = U(:,end-c+t+1:end);
        
         Ms = Ms + M{i};
    end
    
    A = Ms + gamma * N;
    
    [U,S,V]  = svd(A);
    nS = sum(diag(S)>0);
    Z0_o = U(:,end-t+1:end);
    
%% compute W


    for i = 1:k
        %W_o{i} = (Xtrn{i}*H*Xtrn{i}')\Xtrn{i}*H*[Z0_o,Z_o{i}]; %use pinv
        W_o{i} = pinv(Xtrn{i}*H*Xtrn{i}')*Xtrn{i}*H*[Z0_o,Z_o{i}];
    end

%% compute C
    for i = 1:k
        Zk = [Zk, Z_o{i}];
    end

    C_o = (Ytrn'*Ytrn)\Ytrn'*[Z0_o, Zk];

%% compute b
for i = 1:k
    b_o{i} = 1/ntrn*(W_o{i}'*Xtrn{i} - ([Z0_o, Z_o{i}])')*vec1;
end






