function [w,b] = trainsvm(train_data, train_label, C)
% Train linear SVM (primal form)
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  C: tradeoff parameter (on slack variable side)
%
% Output:
%  w: feature vector (column vector)
%  b: bias term
%
% CSCI 576 2014 Fall, Homework 3

[m, n] = size(train_data);

H = diag([ones(n,1); zeros(m+1,1)]);
f = [zeros(m+1,1); C*ones(n,1)];


opts = optimoptions('quadprog', 'Algorithm', 'interior-point-convex','Display','off');
partOneA = - repmat([train_label]', n+1, 1)'  .* [train_data ones(m,1)];
partTwoA = - eye(m);
                    
A = [partOneA partTwoA];
b = -[ones(m,1)];
lb = -[inf(m+1,1); zeros(n,1)];
                    
[x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,[],[],lb,[],[],opts);
                    
w = x(1:n);
b = x(n+1);
