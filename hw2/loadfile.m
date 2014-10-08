function[train_data, train_label, test_data, test_label] = loadfile(k)

% load files

if k = 'spam'
train_data_file_TP = 'hw2_data/spam/train/spam/train_spam.final';
train_data_TP = importdata(train_data_file_TP);
train_data_file_TN = 'hw2_data/spam/train/ham/train_ham.final';
train_data_TN = importdata(train_data_file_TN);
train_data = [train_data_TP; train_data_TN];
train_label = [ones(size(train_data_TP, 1),1); zeros(size(train_data_TN, 1),1)];
test_data_file_TP = 'hw2_data/spam/test/spam/test_spam.final';
test_data_TP = importdata(test_data_file_TP);
test_data_file_TN = 'hw2_data/spam/test/ham/test_ham.final';
test_data_TN = importdata(test_data_file_TN);
test_data = [test_data_TP; test_data_TN];
test_label = [ones(size(test_data_TP, 1),1); zeros(size(test_data_TN, 1),1)];

else
train_data_file = 'hw2_data/ionosphere/ionosphere_train.dat.final';
train_data_raw = importdata(train_data_file);
train_data = train_data_raw(:,1:34);
train_data_label = train_data_raw(:,35);
test_data_file = 'hw2_data/ionosphere/ionosphere_test.dat.final';
test_data_raw = importdata(test_data_file);
test_data = test_data_raw(:,1:34);
test_data_label = test_data_raw(:,35);

end