function[train_data, train_label, test_data, test_label] = loadfile()

% load files
train_data_file = 'splice_test.mat';
train_data_struc = importdata(train_data_file);
train_data = train_data_struc.data;
train_label = train_data_struc.label;

test_data_file = 'splice_train.mat';
test_data_struc = importdata(test_data_file);
test_data = test_data_struc.data;
test_label = test_data_struc.label;

