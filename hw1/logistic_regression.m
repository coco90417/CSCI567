function [new_accu, train_accu] = decision_tree(train_data, train_label, new_data, new_label)

% CSCI 576 2014 Fall, Homework 1

cov = mnrfit(train_data, train_label)
[cov(1:3)'; repmat(cov(4:end),1,3)]