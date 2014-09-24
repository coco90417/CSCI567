function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data 
%
% CSCI 576 2014 Fall, Homework 1

% trainmodel
xparameters = [sum(train_data(train_label==1,:))/size(train_label(train_label==1),1); sum(train_data(train_label==2,:))/size(train_label(train_label==2),1); sum(train_data(train_label==3,:))/size(train_label(train_label==3),1); sum(train_data(train_label==4,:))/size(train_label(train_label==4),1)];
xparameters(xparameters==0)=0.1;
yparameters = [size(train_label(train_label==1),1)/size(train_label,1), size(train_label(train_label==2),1)/size(train_label,1), size(train_label(train_label==3),1)/size(train_label,1), size(train_label(train_label==4),1)/size(train_label,1)];
logXparameters = log(xparameters);
logYparameters = log(xparameters);

%  train_accu: accuracy of classifying train_data
logEstimated = (train_data * transpose(logXparameters)+ repmat(logYparameters,size(train_label,1),1));
estimated = exp(logEstimated);
[estimatedMaxVal estimatedMaxInd] = max(estimated,[],2);
size(train_data(estimatedMaxInd==train_label),1);
train_accu = size(train_data(estimatedMaxInd==train_label),1)/size(train_data,1);

%  new_accu: accuracy of classifying new_data
logEstimated = (test_data * transpose(logXparameters)+ repmat(logYparameters,size(test_label,1),1));
estimated = exp(logEstimated);
[estimatedMaxVal estimatedMaxInd] = max(estimated,[],2);
size(test_data(estimatedMaxInd==test_label),1);
new_accu = size(test_data(estimatedMaxInd==test_label),1)/size(test_data,1);