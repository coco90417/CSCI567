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