function[train_data, train_label] = loadfile()

% load files
train_data_file = '2DGaussian.csv';
train_data=csvread(train_data_file, 1);
train_label = train_data(:,1);
train_data = train_data(:,2:3);

