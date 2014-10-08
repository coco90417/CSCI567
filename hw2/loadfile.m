function[train_data, train_label, test_data, test_label] = loadfile(k)

% load files

if k = 'ionosphere'
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%*s%*[\n]';
train_filename = './hw2_data/ionosphere/ionosphere_train.dat';
fid=fopen(train_filename, 'r');
train_data = textscan(fid, formatSpec, 'delimiter', ',');
fclose(fid);
train_data=train_data(:,1:34);
train_label = train_data(:,35);
test_filename = './hw2_data/ionosphere/ionosphere_test.dat';
test_data = importdata(test_filename, ',');
test_data= test_data(:,1:34);
test_label = test_data(:,35);


elsif k = 'spam'
% load vocab
vocab = textread('./hw2_data/spam/vocab.dat');
list = dir('./hw2_data/spam/train/spam');

end