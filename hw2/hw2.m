%
% CSCI 567 hw2
%

[spam_train_data, spam_train_label, spam_test_data, spam_test_label]=loadfile('spam');
[iono_train_data, iono_train_label, iono_test_data, iono_test_label]=loadfile('iono');
lambda = 0:0.05:0.5;
stepsize = [0.001, 0.01, 0.05, 0.1, 0.5];
T = 1:1:50;

%%%%%%%%%%%%%%%%%%%% (1) %%%%%%%%%%%%%%%%%%%%%%
disp('5.2')
disp('1')
loadheader(spam_train_data);

%%%%%%%%%%%%%%%%%%%% (3) %%%%%%%%%%%%%%%%%%%%%%
disp('5.3')
disp('2')

%%%%%%%%%% 3.a
disp('a. figure')
[spam_train_data_cross_entropy, spam_test_data_cross_entropy, spam_l2norm, spam_w_out] = gradientdescent(spam_train_data, spam_train_label, spam_test_data, spam_test_label);
[iono_train_data_cross_entropy, iono_test_data_cross_entropy, iono_l2norm, iono_w_out] = gradientdescent(iono_train_data, iono_train_label, iono_test_data, iono_test_label);

filename = '3a_spam.pdf';
h = figure;
plot_spam_train_data_cross_entropy = spam_train_data_cross_entropy(:,:,1);
reshaped_plot_spam_train_data_cross_entropy = transpose(reshape(plot_spam_train_data_cross_entropy,  5, 50));
plot(reshaped_plot_spam_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for spam data(no regularization)');
xlabel('iterations');
ylabel('cross-entropy');
legend('stepsize = 0.001', 'stepsize = 0.01', 'stepsize = 0.05', 'stepsize = 0.1', 'stepsize = 0.5');
saveas(h, filename);

filename = '3a_iono.pdf';
h = figure;
plot_iono_train_data_cross_entropy = iono_train_data_cross_entropy(1,:,:);
reshaped_plot_iono_train_data_cross_entropy = transpose(reshape(plot_iono_train_data_cross_entropy,   5, 50));
plot(reshaped_plot_iono_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for ionosphere data(no regularization)');
xlabel('iterations');
ylabel('cross-entropy');
legend('stepsize = 0.001', 'stepsize = 0.01', 'stepsize = 0.05', 'stepsize = 0.1', 'stepsize = 0.5');
saveas(h, filename);


%%%%%%%%%% 3.b
disp('b. l2 norm')
disp('spam data')
show_spam_l2norm = spam_l2norm(1,:,50);
show_spam_l2norm
disp('ionophere data')
show_iono_l2norm = iono_l2norm(1,:,50);
show_iono_l2norm


%%%%%%%%%%%%%%%%%%%% (4) %%%%%%%%%%%%%%%%%%%%%%
disp('5.2')
disp('4')

%%%%%%%%%% 4.a
disp('a. figure')
filename = '4a_spam.pdf';
h = figure;
plot_spam_train_data_cross_entropy = spam_train_data_cross_entropy(3,:,:);
reshaped_plot_spam_train_data_cross_entropy = transpose(reshape(plot_spam_train_data_cross_entropy,  5, 50));
plot(reshaped_plot_spam_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for spamsphere data(lamda=0.1)');
xlabel('iterations');
ylabel('cross-entropy');
legend('stepsize = 0.001', 'stepsize = 0.01', 'stepsize = 0.05', 'stepsize = 0.1', 'stepsize = 0.5');
saveas(h, filename);

filename = '4a_iono.pdf';
h = figure;
plot_iono_train_data_cross_entropy = iono_train_data_cross_entropy(3,:,:);
reshaped_plot_iono_train_data_cross_entropy = transpose(reshape(plot_iono_train_data_cross_entropy,  5, 50));
plot(reshaped_plot_iono_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for ionosphere data(lamda=0.1)');
xlabel('iterations');
ylabel('cross-entropy');
legend('stepsize = 0.001', 'stepsize = 0.01', 'stepsize = 0.05', 'stepsize = 0.1', 'stepsize = 0.5');
saveas(h, filename);


%%%%%%%%%% 4.b
disp('b. l2 norm')
disp('spam data')
show_spam_l2norm = spam_l2norm(:,2,50);
show_spam_l2norm
disp('ionophere data')
show_iono_l2norm = iono_l2norm(:,2,50);
show_iono_l2norm


%%%%%%%%%% 4.c
disp('c. figure')
filename = '4c1_spam.pdf';
h = figure;
plot_spam_data_cross_entropy = [spam_train_data_cross_entropy(:,1,50) spam_test_data_cross_entropy(:,1,50)];
reshaped_plot_spam_data_cross_entropy = transpose(reshape(plot_spam_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_spam_data_cross_entropy(1,:), lambda, reshaped_plot_spam_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for spam data(stepsize=0.001)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c2_spam.pdf';
h = figure;
plot_spam_data_cross_entropy = [spam_train_data_cross_entropy(:,2,50) spam_test_data_cross_entropy(:,2,50)];
reshaped_plot_spam_data_cross_entropy = transpose(reshape(plot_spam_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_spam_data_cross_entropy(1,:), lambda, reshaped_plot_spam_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for spam data(stepsize=0.01)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c3_spam.pdf';
h = figure;
plot_spam_data_cross_entropy = [spam_train_data_cross_entropy(:,3,50) spam_test_data_cross_entropy(:,3,50)];
reshaped_plot_spam_data_cross_entropy = transpose(reshape(plot_spam_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_spam_data_cross_entropy(1,:), lambda, reshaped_plot_spam_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for spam data(stepsize=0.05)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c4_spam.pdf';
h = figure;
plot_spam_data_cross_entropy = [spam_train_data_cross_entropy(:,4,50) spam_test_data_cross_entropy(:,4,50)];
reshaped_plot_spam_data_cross_entropy = transpose(reshape(plot_spam_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_spam_data_cross_entropy(1,:), lambda, reshaped_plot_spam_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for spam data(stepsize=0.1)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c5_spam.pdf';
h = figure;
plot_spam_data_cross_entropy = [spam_train_data_cross_entropy(:,5,50) spam_test_data_cross_entropy(:,5,50)];
reshaped_plot_spam_data_cross_entropy = transpose(reshape(plot_spam_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_spam_data_cross_entropy(1,:), lambda, reshaped_plot_spam_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for spam data(stepsize=0.5)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c1_iono.pdf';
h = figure;
plot_iono_data_cross_entropy = [iono_train_data_cross_entropy(:,1,50) iono_test_data_cross_entropy(:,1,50)];
reshaped_plot_iono_data_cross_entropy = transpose(reshape(plot_iono_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_iono_data_cross_entropy(1,:), lambda, reshaped_plot_iono_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for iono data(stepsize=0.001)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c2_iono.pdf';
h = figure;
plot_iono_data_cross_entropy = [iono_train_data_cross_entropy(:,2,50) iono_test_data_cross_entropy(:,2,50)];
reshaped_plot_iono_data_cross_entropy = transpose(reshape(plot_iono_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_iono_data_cross_entropy(1,:), lambda, reshaped_plot_iono_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for iono data(stepsize=0.01)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c3_iono.pdf';
h = figure;
plot_iono_data_cross_entropy = [iono_train_data_cross_entropy(:,3,50) iono_test_data_cross_entropy(:,3,50)];
reshaped_plot_iono_data_cross_entropy = transpose(reshape(plot_iono_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_iono_data_cross_entropy(1,:), lambda, reshaped_plot_iono_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for iono data(stepsize=0.05)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c4_iono.pdf';
h = figure;
plot_iono_data_cross_entropy = [iono_train_data_cross_entropy(:,4,50) iono_test_data_cross_entropy(:,4,50)];
reshaped_plot_iono_data_cross_entropy = transpose(reshape(plot_iono_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_iono_data_cross_entropy(1,:), lambda, reshaped_plot_iono_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for iono data(stepsize=0.1)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);

filename = '4c5_iono.pdf';
h = figure;
plot_iono_data_cross_entropy = [iono_train_data_cross_entropy(:,5,50) iono_test_data_cross_entropy(:,5,50)];
reshaped_plot_iono_data_cross_entropy = transpose(reshape(plot_iono_data_cross_entropy,  11, 2));
plot(lambda, reshaped_plot_iono_data_cross_entropy(1,:), lambda, reshaped_plot_iono_data_cross_entropy(2,:));
title('cross-entropy with lambda at 50th iteration for iono data(stepsize=0.5)');
xlabel('lambda');
ylabel('cross-entropy');
legend('training', 'testing');
saveas(h, filename);


%%%%%%%%%%%%%%%%%%%% (6) %%%%%%%%%%%%%%%%%%%%%%
[nt_spam_train_data_cross_entropy, nt_spam_test_data_cross_entropy, nt_spam_l2norm] = newtonmethod(spam_train_data, spam_train_label, spam_test_data, spam_test_label, spam_w_out);
[nt_iono_train_data_cross_entropy, nt_iono_test_data_cross_entropy, nt_iono_l2norm] = newtonmethod(iono_train_data, iono_train_label, iono_test_data, iono_test_label, iono_w_out);

%%%%%%%%%% 6.a
disp('a. fiure')
filename = '6a_spam.pdf';
h = figure;
plot_spam_train_data_cross_entropy = nt_spam_train_data_cross_entropy;
reshaped_plot_spam_train_data_cross_entropy = transpose(reshape(plot_spam_train_data_cross_entropy,  11, 50));
plot(reshaped_plot_spam_train_data_cross_entropy(:,1));
title('cross-entropy from 1-50 iterations for training data for spamsphere data');
xlabel('iterations');
ylabel('cross-entropy');
legend('lambda = 0');
saveas(h, filename);



filename = '6a_iono.pdf';
h = figure;
plot_iono_train_data_cross_entropy = nt_iono_train_data_cross_entropy;
reshaped_plot_iono_train_data_cross_entropy = transpose(reshape(plot_iono_train_data_cross_entropy,  11, 50));
plot(reshaped_plot_iono_train_data_cross_entropy(:,1));
title('cross-entropy from 1-50 iterations for training data for ionosphere data');
xlabel('iterations');
ylabel('cross-entropy');
legend('lambda = 0');
saveas(h, filename);

%%%%%%%%%% 6.b
disp('b. l2 norm')
disp('spam data')
show_spam_l2norm = nt_spam_l2norm(:,50);
show_spam_l2norm
disp('ionophere data')
show_iono_l2norm = nt_iono_l2norm(:,50);
show_iono_l2norm


%%%%%%%%%%% 6.c
disp('c. cross entropy')
disp('spam data')
show_spam_cross = nt_spam_test_data_cross_entropy(:,50);
show_spam_cross
disp('ionosphere data')
show_iono_cross = nt_iono_test_data_cross_entropy(:,50);
show_iono_cross




%%%%%%%%%%%%%%%%%%%% (7) %%%%%%%%%%%%%%%%%%%%%%


filename = '7_spam.pdf';
h = figure;
plot_spam_train_data_cross_entropy = nt_spam_train_data_cross_entropy;
reshaped_plot_spam_train_data_cross_entropy = transpose(reshape(plot_spam_train_data_cross_entropy,  11, 50));
plot(reshaped_plot_spam_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for spamsphere data');
xlabel('iterations');
ylabel('cross-entropy');
legend('lambda = 0', 'lambda = 0.05', 'lambda = 0.1', 'lambda = 0.15', 'lambda = 0.2', 'lambda = 0.25', 'lambda = 0.3', 'lambda = 0.35', 'lambda = 0.4', 'lambda = 0.45', 'lambda = 0.5');
saveas(h, filename);

filename = '7_iono.pdf';
h = figure;
plot_iono_train_data_cross_entropy = nt_iono_train_data_cross_entropy;
reshaped_plot_iono_train_data_cross_entropy = transpose(reshape(plot_iono_train_data_cross_entropy,  11, 50));
plot(reshaped_plot_iono_train_data_cross_entropy);
title('cross-entropy from 1-50 iterations for training data for ionosphere data');
xlabel('iterations');
ylabel('cross-entropy');
legend('lambda = 0', 'lambda = 0.05', 'lambda = 0.1', 'lambda = 0.15', 'lambda = 0.2', 'lambda = 0.25', 'lambda = 0.3', 'lambda = 0.35', 'lambda = 0.4', 'lambda = 0.45', 'lambda = 0.5');
saveas(h, filename);

