function [train_data, train_label, test_data, test_label, validation_data, validation_label] = loadfile()

% load files

train_data_file = 'car_train.data.out.x'
train_label_file = 'car_train.data.out.y'
test_data_file = 'car_test.data.out.x'
test_label_file = 'car_test.data.out.y'
validation_data_file = 'car_valid.data.out.x'
validation_label_file = 'car_valid.data.out.y'


train_data = importdata(train_data_file)
train_label = importdata(train_label_file)
test_data = importdata(test_data_file)
test_label = importdata(test_label_file)
validation_data = importdata(validation_data_file)
validation_label = importdata(validation_label_file)

