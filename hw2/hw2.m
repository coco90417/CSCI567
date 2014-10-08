function [] = main()

%
% CSCI 567 hw2
% usage: main()
%

[spam_train_data, spam_train_label, spam_test_data, spam_test_label]=loadfile('spam');
[iono_train_data, iono_train_label, iono_test_data, iono_test_label]=loadfile('iono');

%%%%%%%%%%%%%%%%%%%% (1) %%%%%%%%%%%%%%%%%%%%%%
disp('5.2')
disp('1')
loadheader(spam_train_data);

%%%%%%%%%%%%%%%%%%%% (3) %%%%%%%%%%%%%%%%%%%%%%
disp('5.3')
disp('2')
disp('figure')
[spam_train_data_cross_entropy, spam_test_data_cross_entropy, spam_l2norm] = gradientdescent(spam_train_data, spam_train_label, spam_test_data, spam_test_label);
[iono_train_data_cross_entropy, iono_test_data_cross_entropy, iono_l2norm] = gradientdescent(iono_train_data, iono_train_label, iono_test_data, iono_test_label);

filename = '3a_spam.pdf';
h = figure;
plot_spam_train_data_cross_entropy = spam_train_data_cross_entropy(1,:,:);
reshaped_plot_spam_train_data_cross_entropy = transpose(reshape(plot_spam_train_data_cross_entropy,  size(stepsize,2), size(T,2)));
plot(reshaped_plot_spam_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for spam data(no regularization)');
xlabel('iterations');
ylabel('cross-entropy');
legend('stepsize = 0.001', 'stepsize = 0.01', 'stepsize = 0.05', 'stepsize = 0.1', 'stepsize = 0.5');
saveas(h, filename);
disp('l2 norm')
show_spam_l2norm = spam_l2norm(1,:,50);
show_spam_l2norm


filename = '3a_iono.pdf';
h = figure;
plot_iono_train_data_cross_entropy = iono_train_data_cross_entropy(1,:,:);
reshaped_plot_iono_train_data_cross_entropy = transpose(reshape(plot_iono_train_data_cross_entropy,  size(stepsize,2), size(T,2)));
plot(reshaped_plot_iono_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for ionosphere data(no regularization)');
xlabel('iterations');
ylabel('cross-entropy');
legend('stepsize = 0.001', 'stepsize = 0.01', 'stepsize = 0.05', 'stepsize = 0.1', 'stepsize = 0.5');
saveas(h, filename);
disp('l2 norm')
show_iono_l2norm = iono_l2norm(1,:,50);
show_iono_l2norm


%%%%%%%%%%%%%%%%%%%% (4) %%%%%%%%%%%%%%%%%%%%%%
disp('5.2')
disp('4')
disp('figure')
filename = '4a_spam.pdf';
h = figure;
plot_spam_train_data_cross_entropy = spam_train_data_cross_entropy(1,:,:);
reshaped_plot_spam_train_data_cross_entropy = transpose(reshape(plot_spam_train_data_cross_entropy,  size(stepsize,2), size(T,2)));
plot(reshaped_plot_spam_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for spam data(lamda=1)');
xlabel('iterations');
ylabel('cross-entropy');
legend('stepsize = 0.001', 'stepsize = 0.01', 'stepsize = 0.05', 'stepsize = 0.1', 'stepsize = 0.5');
saveas(h, filename);
disp('l2 norm')
show_spam_l2norm = spam_l2norm(1,:,50);
show_spam_l2norm


filename = '4a_iono.pdf';
h = figure;
plot_iono_train_data_cross_entropy = iono_train_data_cross_entropy(1,:,:);
reshaped_plot_iono_train_data_cross_entropy = transpose(reshape(plot_iono_train_data_cross_entropy,  size(stepsize,2), size(T,2)));
plot(reshaped_plot_iono_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for ionosphere data(no regularization)');
xlabel('iterations');
ylabel('cross-entropy');
legend('stepsize = 0.001', 'stepsize = 0.01', 'stepsize = 0.05', 'stepsize = 0.1', 'stepsize = 0.5');
saveas(h, filename);
disp('l2 norm')
show_iono_l2norm = iono_l2norm(1,:,50);
show_iono_l2norm


%%%%%%%%%%%%%%%%%%%% (5) %%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%% (6) %%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%% (7) %%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%% (8) %%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%% (9) %%%%%%%%%%%%%%%%%%%%%%