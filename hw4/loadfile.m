function[train_data] = loadfile()

% load files
train_data_file = '2DGaussian.csv';
train_data=csvread(train_data_file, 1);

