function[train_data, train_label, test_data, test_label] = loadfile(k)

% load files

if k = 'spam'
train_data_file_TP = 'hw2_data/spam/train/spam/train_spam.final';
train_data_TP = importdata(train_data_file_TP);
train_data_file_TN = 'hw2_data/spam/train/ham/train_ham.final';
train_data_TN = importdata(train_data_file_TN);
test_data_file_TP = 'hw2_data/spam/test/spam/test_spam.final';
test_data_TP = importdata(test_data_file_TP);
test_data_file_TN = 'hw2_data/spam/test/ham/test_ham.final';
test_data_TN = importdata(test_data_file_TN);



my $email_train_spam = "hw2_data/spam/train/spam/train_spam";
my $email_train_ham = "hw2_data/spam/train/ham/train_ham";
my $email_test_spam = "hw2_data/spam/test/spam/test_spam";
my $email_test_ham = "hw2_data/spam/test/ham/test_ham";

else
train_data_file = 'hw2_data/ionosphere/ionosphere_train.dat.final';
train_data_label =
train_data_file = 'hw2_data/ionosphere/ionosphere_test.dat.final';
train_data_label =



train_data = importdata(train_data_file);
train_label = importdata(train_label_file);
test_data = importdata(test_data_file);
test_label = importdata(test_label_file);
valid_data = importdata(validation_data_file);
valid_label = importdata(validation_label_file);