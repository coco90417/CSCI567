function [] = main()

%
% CSCI 567 hw2
% usage: main()
%

[spam_train_data, spam_train_label, spam_test_data, spam_test_label]=loadfile('spam');
[iono_train_data, iono_train_label, iono_test_data, iono_test_label]=loadfile('iono');

% 5.2
loadheader();