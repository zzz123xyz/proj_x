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
%% ===============================
%   x: cell ,x{i} means ith feature for a sample x



function [z0, z] = Xtest(x, W, b, t, k)

zsum = zeros(1,t);


for i = 1:k
    tmp1 = x{i}'*W{i}(:,1:t)+b{i}(1:t,:)';
    zsum = zsum + tmp1; 
end
    
z0 = 1/k*zsum;

z = [];
for i = 1:k
    tmp1 = x{i}'*W{i}(:,t+1:end)+b{i}(t+1:end,:)'; 
    z = cat(2, z, tmp1);
end


