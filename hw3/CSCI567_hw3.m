[train_data, train_label, test_data, test_label] = loadfile()

% load files
train_data_file = 'splice_test.mat';
train_data = importdata(train_data_file);

test_data_file = 'splice_train.mat';
test_data = importdata(test_data_file);

