[train_data, train_label, test_data, test_label] = loadfile();

% load files

model = svmtrain(train_label, train_data, '-t 0 -c 4^(−6) -v 5');

