function[train_data, train_label, test_data, test_label] = loadfile()

% load files
train_data_file = 'splice_train.mat';
train_data_struc = importdata(train_data_file);
train_data = train_data_struc.data;
train_label = train_data_struc.label;

test_data_file = 'splice_test.mat';
test_data_struc = importdata(test_data_file);
test_data = test_data_struc.data;
test_label = test_data_struc.label;

train_mean_matrix = repmat(mean(train_data),size(train_data,1),1);
train_std_matrix = repmat(std(train_data),size(train_data,1),1);
test_mean_matrix = repmat(mean(train_data),size(test_data,1),1);
test_std_matrix = repmat(std(train_data),size(test_data,1),1);

train_data = (train_data - train_mean_matrix)./train_std_matrix;
test_data = (test_data - test_mean_matrix)./test_std_matrix;
