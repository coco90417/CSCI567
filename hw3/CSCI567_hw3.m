% load files
[train_data, train_label, test_data, test_label] = loadfile();


% add path, required for libsvm
addpath('/home/cocodong/usr/matlab/lib/libsvm-3.18/matlab');

[m n] = size(train_data);
[test_m test_n] = size(test_data);


crosssize = round(m/5);
sub_test_data(1) = train_data(1:crosssize, :);
sub_test_label(1) = train_label(1:crosssize, :);
sub_train_data(1) = set_diff(train_data, sub_test_data(1), 'row');
sub_train_label(1) = set_diff(train_label, sub_train_label(1), 'row');

sub_test_data(2) = train_data(crosssize+1:2*crosssize, :);
sub_test_label(2) = train_label(crosssize+1:2*crosssize, :);
sub_train_data(2) = set_diff(train_data, sub_test_data(2), 'row');
sub_train_label(2) = set_diff(train_label, sub_train_label(2), 'row');

sub_test_data(3) = train_data(2*crosssize+1:3*crosssize, :);
sub_test_label(3) = train_label(2*crosssize+1:3*crosssize, :);
sub_train_data(3) = set_diff(train_data, sub_test_data(3), 'row');
sub_train_label(3) = set_diff(train_label, sub_train_label(3), 'row');

sub_test_data(4) = train_data(3*crosssize+1:4*crosssize, :);
sub_test_label(4) = train_label(3*crosssize+1:4*crosssize, :);
sub_train_data(4) = set_diff(train_data, sub_test_data(4), 'row');
sub_train_label(4) = set_diff(train_label, sub_train_label(4), 'row');

sub_test_data(5) = train_data(4*crosssize:m, :);
sub_test_label(5) = train_label(5*crosssize:m, :);
sub_train_data(5) = set_diff(train_data, sub_test_data(5), 'row');
sub_train_label(5) = set_diff(train_label, sub_train_label(5), 'row');

% implement linear SVM
cost = [4^(-6) 4^(-5) 4^(-4) 4^(-3) 4^(-2) 4^(-1) 1 4 4^2];

for i=1:9
tempclock = clock();
starttime = tempclock(5) * 60 + tempclock(6);
temp_accuracy = zeros(5,1);
for j=1:5
[temp_w,temp_b] = trainsvm(sub_train_data(j), sub_train_label(j), cost(i));
temp_accuracy(j) = testsvm(sub_test_data(j), sub_test_label(j), temp_w, temp_b);
end
tempclock = clock();
endtime  = tempclock(5) * 60 + tempclock(6);
second = endtime  - starttime ;
average_time(i) = second/5;
accuracy = sum(temp_accuracy)/5;
end
 
% Use linear SVM in LIBSVM
model = zeros(9,1);
tempclock = clock();
starttime = tempclock(5) * 60 + tempclock(6);
model(1) = svmtrain(train_label, train_data, '-t 0 -c 4^(−6) -v 5');
model(2) = svmtrain(train_label, train_data, '-t 0 -c 4^(−5) -v 5');
model(3) = svmtrain(train_label, train_data, '-t 0 -c 4^(−4) -v 5');
model(4) = svmtrain(train_label, train_data, '-t 0 -c 4^(−3) -v 5');
model(5) = svmtrain(train_label, train_data, '-t 0 -c 4^(−2) -v 5');
model(6) = svmtrain(train_label, train_data, '-t 0 -c 4^(−1) -v 5');
model(7) = svmtrain(train_label, train_data, '-t 0 -c 4^(0) -v 5');
model(8) = svmtrain(train_label, train_data, '-t 0 -c 4^(1) -v 5');
model(9) = svmtrain(train_label, train_data, '-t 0 -c 4^(2) -v 5');
tempclock = clock();
endtime  = tempclock(5) * 60 + tempclock(6);
second = endtime  - starttime ;

 
% Use poly SVM in LIBSVM
 
model = zeros(11,1);
tempclock = clock();
starttime = tempclock(5) * 60 + tempclock(6);
model(1) = svmtrain(train_label, train_data, '-t 1 -c 4^(−3) -d 1 -v 5');
model(2) = svmtrain(train_label, train_data, '-t 1 -c 4^(−2) -d 1 -v 5');
model(3) = svmtrain(train_label, train_data, '-t 1 -c 4^(−1) -d 1 -v 5');
model(4) = svmtrain(train_label, train_data, '-t 1 -c 4^(0) -d 1 -v 5');
model(5) = svmtrain(train_label, train_data, '-t 1 -c 4^(1) -d 1 -v 5');
model(6) = svmtrain(train_label, train_data, '-t 1 -c 4^(2) -d 1 -v 5');
model(7) = svmtrain(train_label, train_data, '-t 1 -c 4^(3) -d 1 -v 5');
model(8) = svmtrain(train_label, train_data, '-t 1 -c 4^(4) -d 1 -v 5');
model(9) = svmtrain(train_label, train_data, '-t 1 -c 4^(5) -d 1 -v 5');
model(10) = svmtrain(train_label, train_data, '-t 1 -c 4^(6) -d 1 -v 5');
model(11) = svmtrain(train_label, train_data, '-t 1 -c 4^(7) -d 1 -v 5');
tempclock = clock();
endtime  = tempclock(5) * 60 + tempclock(6);
second = endtime  - starttime ;

model = zeros(11,1);
tempclock = clock();
starttime = tempclock(5) * 60 + tempclock(6);
model(1) = svmtrain(train_label, train_data, '-t 1 -c 4^(−3) -d 2 -v 5');
model(2) = svmtrain(train_label, train_data, '-t 1 -c 4^(−2) -d 2 -v 5');
model(3) = svmtrain(train_label, train_data, '-t 1 -c 4^(−1) -d 2 -v 5');
model(4) = svmtrain(train_label, train_data, '-t 1 -c 4^(0) -d 2 -v 5');
model(5) = svmtrain(train_label, train_data, '-t 1 -c 4^(1) -d 2 -v 5');
model(6) = svmtrain(train_label, train_data, '-t 1 -c 4^(2) -d 2 -v 5');
model(7) = svmtrain(train_label, train_data, '-t 1 -c 4^(3) -d 2 -v 5');
model(8) = svmtrain(train_label, train_data, '-t 1 -c 4^(4) -d 2 -v 5');
model(9) = svmtrain(train_label, train_data, '-t 1 -c 4^(5) -d 2 -v 5');
model(10) = svmtrain(train_label, train_data, '-t 1 -c 4^(6) -d 2 -v 5');
model(11) = svmtrain(train_label, train_data, '-t 1 -c 4^(7) -d 2 -v 5');
tempclock = clock();
endtime  = tempclock(5) * 60 + tempclock(6);
second = endtime  - starttime ;

model = zeros(11,1);
tempclock = clock();
starttime = tempclock(5) * 60 + tempclock(6);
model(1) = svmtrain(train_label, train_data, '-t 1 -c 4^(−3) -d 3 -v 5');
model(2) = svmtrain(train_label, train_data, '-t 1 -c 4^(−2) -d 3 -v 5');
model(3) = svmtrain(train_label, train_data, '-t 1 -c 4^(−1) -d 3 -v 5');
model(4) = svmtrain(train_label, train_data, '-t 1 -c 4^(0) -d 3 -v 5');
model(5) = svmtrain(train_label, train_data, '-t 1 -c 4^(1) -d 3 -v 5');
model(6) = svmtrain(train_label, train_data, '-t 1 -c 4^(2) -d 3 -v 5');
model(7) = svmtrain(train_label, train_data, '-t 1 -c 4^(3) -d 3 -v 5');
model(8) = svmtrain(train_label, train_data, '-t 1 -c 4^(4) -d 3 -v 5');
model(9) = svmtrain(train_label, train_data, '-t 1 -c 4^(5) -d 3 -v 5');
model(10) = svmtrain(train_label, train_data, '-t 1 -c 4^(6) -d 3 -v 5');
model(11) = svmtrain(train_label, train_data, '-t 1 -c 4^(7) -d 3 -v 5');
tempclock = clock();
endtime  = tempclock(5) * 60 + tempclock(6);
second = endtime  - starttime ;
 
 

 
 
 % Use kernel SVM in LIBSVM
 
 model = zeros(11,1);
 tempclock = clock();
 starttime = tempclock(5) * 60 + tempclock(6);
 model(1) = svmtrain(train_label, train_data, '-t 2 -c 4^(−3) -g 4^(−7) -v 5');
 model(2) = svmtrain(train_label, train_data, '-t 2 -c 4^(−2) -g 4^(−7) -v 5');
 model(3) = svmtrain(train_label, train_data, '-t 2 -c 4^(−1) -g 4^(−7) -v 5');
 model(4) = svmtrain(train_label, train_data, '-t 2 -c 4^(0) -g 4^(−7) -v 5');
 model(5) = svmtrain(train_label, train_data, '-t 2 -c 4^(1) -g 4^(−7) -v 5');
 model(6) = svmtrain(train_label, train_data, '-t 2 -c 4^(2) -g 4^(−7) -v 5');
 model(7) = svmtrain(train_label, train_data, '-t 2 -c 4^(3) -g 4^(−7) -v 5');
 model(8) = svmtrain(train_label, train_data, '-t 2 -c 4^(4) -g 4^(−7) -v 5');
 model(9) = svmtrain(train_label, train_data, '-t 2 -c 4^(5) -g 4^(−7) -v 5');
 model(10) = svmtrain(train_label, train_data, '-t 2 -c 4^(6) -g 4^(−7) -v 5');
 model(11) = svmtrain(train_label, train_data, '-t 2 -c 4^(7) -g 4^(−7) -v 5');
 tempclock = clock();
 endtime  = tempclock(5) * 60 + tempclock(6);
 second = endtime  - starttime ;

 model = zeros(11,1);
 tempclock = clock();
 starttime = tempclock(5) * 60 + tempclock(6);
 model(1) = svmtrain(train_label, train_data, '-t 2 -c 4^(−3) -g 4^(−6) -v 5');
 model(2) = svmtrain(train_label, train_data, '-t 2 -c 4^(−2) -g 4^(−6) -v 5');
 model(3) = svmtrain(train_label, train_data, '-t 2 -c 4^(−1) -g 4^(−6) -v 5');
 model(4) = svmtrain(train_label, train_data, '-t 2 -c 4^(0) -g 4^(−6) -v 5');
 model(5) = svmtrain(train_label, train_data, '-t 2 -c 4^(1) -g 4^(−6) -v 5');
 model(6) = svmtrain(train_label, train_data, '-t 2 -c 4^(2) -g 4^(−6) -v 5');
 model(7) = svmtrain(train_label, train_data, '-t 2 -c 4^(3) -g 4^(−6) -v 5');
 model(8) = svmtrain(train_label, train_data, '-t 2 -c 4^(4) -g 4^(−6) -v 5');
 model(9) = svmtrain(train_label, train_data, '-t 2 -c 4^(5) -g 4^(−6) -v 5');
 model(10) = svmtrain(train_label, train_data, '-t 2 -c 4^(6) -g 4^(−6) -v 5');
 model(11) = svmtrain(train_label, train_data, '-t 2 -c 4^(7) -g 4^(−6) -v 5');
 tempclock = clock();
 endtime  = tempclock(5) * 60 + tempclock(6);
 second = endtime  - starttime ;

 model = zeros(11,1);
 tempclock = clock();
 starttime = tempclock(5) * 60 + tempclock(6);
 model(1) = svmtrain(train_label, train_data, '-t 2 -c 4^(−3) -g 4^(−5) -v 5');
 model(2) = svmtrain(train_label, train_data, '-t 2 -c 4^(−2) -g 4^(−5) -v 5');
 model(3) = svmtrain(train_label, train_data, '-t 2 -c 4^(−1) -g 4^(−5) -v 5');
 model(4) = svmtrain(train_label, train_data, '-t 2 -c 4^(0) -g 4^(−5) -v 5');
 model(5) = svmtrain(train_label, train_data, '-t 2 -c 4^(1) -g 4^(−5) -v 5');
 model(6) = svmtrain(train_label, train_data, '-t 2 -c 4^(2) -g 4^(−5) -v 5');
 model(7) = svmtrain(train_label, train_data, '-t 2 -c 4^(3) -g 4^(−5) -v 5');
 model(8) = svmtrain(train_label, train_data, '-t 2 -c 4^(4) -g 4^(−5) -v 5');
 model(9) = svmtrain(train_label, train_data, '-t 2 -c 4^(5) -g 4^(−5) -v 5');
 model(10) = svmtrain(train_label, train_data, '-t 2 -c 4^(6) -g 4^(−5) -v 5');
 model(11) = svmtrain(train_label, train_data, '-t 2 -c 4^(7) -g 4^(−5) -v 5');
 tempclock = clock();
 endtime  = tempclock(5) * 60 + tempclock(6);
 second = endtime  - starttime ;

 model = zeros(11,1);
 tempclock = clock();
 starttime = tempclock(5) * 60 + tempclock(6);
 model(1) = svmtrain(train_label, train_data, '-t 2 -c 4^(−3) -g 4^(−4) -v 5');
 model(2) = svmtrain(train_label, train_data, '-t 2 -c 4^(−2) -g 4^(−4) -v 5');
 model(3) = svmtrain(train_label, train_data, '-t 2 -c 4^(−1) -g 4^(−4) -v 5');
 model(4) = svmtrain(train_label, train_data, '-t 2 -c 4^(0) -g 4^(−4) -v 5');
 model(5) = svmtrain(train_label, train_data, '-t 2 -c 4^(1) -g 4^(−4) -v 5');
 model(6) = svmtrain(train_label, train_data, '-t 2 -c 4^(2) -g 4^(−4) -v 5');
 model(7) = svmtrain(train_label, train_data, '-t 2 -c 4^(3) -g 4^(−4) -v 5');
 model(8) = svmtrain(train_label, train_data, '-t 2 -c 4^(4) -g 4^(−4) -v 5');
 model(9) = svmtrain(train_label, train_data, '-t 2 -c 4^(5) -g 4^(−4) -v 5');
 model(10) = svmtrain(train_label, train_data, '-t 2 -c 4^(6) -g 4^(−4) -v 5');
 model(11) = svmtrain(train_label, train_data, '-t 2 -c 4^(7) -g 4^(−4) -v 5');
 tempclock = clock();
 endtime  = tempclock(5) * 60 + tempclock(6);
 second = endtime  - starttime ;

 model = zeros(11,1);
 tempclock = clock();
 starttime = tempclock(5) * 60 + tempclock(6);
 model(1) = svmtrain(train_label, train_data, '-t 2 -c 4^(−3) -g 4^(−3) -v 5');
 model(2) = svmtrain(train_label, train_data, '-t 2 -c 4^(−2) -g 4^(−3) -v 5');
 model(3) = svmtrain(train_label, train_data, '-t 2 -c 4^(−1) -g 4^(−3) -v 5');
 model(4) = svmtrain(train_label, train_data, '-t 2 -c 4^(0) -g 4^(−3) -v 5');
 model(5) = svmtrain(train_label, train_data, '-t 2 -c 4^(1) -g 4^(−3) -v 5');
 model(6) = svmtrain(train_label, train_data, '-t 2 -c 4^(2) -g 4^(−3) -v 5');
 model(7) = svmtrain(train_label, train_data, '-t 2 -c 4^(3) -g 4^(−3) -v 5');
 model(8) = svmtrain(train_label, train_data, '-t 2 -c 4^(4) -g 4^(−3) -v 5');
 model(9) = svmtrain(train_label, train_data, '-t 2 -c 4^(5) -g 4^(−3) -v 5');
 model(10) = svmtrain(train_label, train_data, '-t 2 -c 4^(6) -g 4^(−3) -v 5');
 model(11) = svmtrain(train_label, train_data, '-t 2 -c 4^(7) -g 4^(−3) -v 5');
 tempclock = clock();
 endtime  = tempclock(5) * 60 + tempclock(6);
 second = endtime  - starttime ;

 model = zeros(11,1);
 tempclock = clock();
 starttime = tempclock(5) * 60 + tempclock(6);
 model(1) = svmtrain(train_label, train_data, '-t 2 -c 4^(−3) -g 4^(−2) -v 5');
 model(2) = svmtrain(train_label, train_data, '-t 2 -c 4^(−2) -g 4^(−2) -v 5');
 model(3) = svmtrain(train_label, train_data, '-t 2 -c 4^(−1) -g 4^(−2) -v 5');
 model(4) = svmtrain(train_label, train_data, '-t 2 -c 4^(0) -g 4^(−2) -v 5');
 model(5) = svmtrain(train_label, train_data, '-t 2 -c 4^(1) -g 4^(−2) -v 5');
 model(6) = svmtrain(train_label, train_data, '-t 2 -c 4^(2) -g 4^(−2) -v 5');
 model(7) = svmtrain(train_label, train_data, '-t 2 -c 4^(3) -g 4^(−2) -v 5');
 model(8) = svmtrain(train_label, train_data, '-t 2 -c 4^(4) -g 4^(−2) -v 5');
 model(9) = svmtrain(train_label, train_data, '-t 2 -c 4^(5) -g 4^(−2) -v 5');
 model(10) = svmtrain(train_label, train_data, '-t 2 -c 4^(6) -g 4^(−2) -v 5');
 model(11) = svmtrain(train_label, train_data, '-t 2 -c 4^(7) -g 4^(−2) -v 5');
 tempclock = clock();
 endtime  = tempclock(5) * 60 + tempclock(6);
 second = endtime  - starttime ;

 model = zeros(11,1);
 tempclock = clock();
 starttime = tempclock(5) * 60 + tempclock(6);
 model(1) = svmtrain(train_label, train_data, '-t 2 -c 4^(−3) -g 4^(−1) -v 5');
 model(2) = svmtrain(train_label, train_data, '-t 2 -c 4^(−2) -g 4^(−1) -v 5');
 model(3) = svmtrain(train_label, train_data, '-t 2 -c 4^(−1) -g 4^(−1) -v 5');
 model(4) = svmtrain(train_label, train_data, '-t 2 -c 4^(0) -g 4^(−1) -v 5');
 model(5) = svmtrain(train_label, train_data, '-t 2 -c 4^(1) -g 4^(−1) -v 5');
 model(6) = svmtrain(train_label, train_data, '-t 2 -c 4^(2) -g 4^(−1) -v 5');
 model(7) = svmtrain(train_label, train_data, '-t 2 -c 4^(3) -g 4^(−1) -v 5');
 model(8) = svmtrain(train_label, train_data, '-t 2 -c 4^(4) -g 4^(−1) -v 5');
 model(9) = svmtrain(train_label, train_data, '-t 2 -c 4^(5) -g 4^(−1) -v 5');
 model(10) = svmtrain(train_label, train_data, '-t 2 -c 4^(6) -g 4^(−1) -v 5');
 model(11) = svmtrain(train_label, train_data, '-t 2 -c 4^(7) -g 4^(−1) -v 5');
 tempclock = clock();
 endtime  = tempclock(5) * 60 + tempclock(6);
 second = endtime  - starttime ;
 
 
