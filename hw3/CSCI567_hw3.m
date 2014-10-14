% load files
[train_data, train_label, test_data, test_label] = loadfile();


% add path, required for libsvm
addpath('/home/cocodong/usr/matlab/lib/libsvm-3.18/matlab');

% implement linear SVM
cost = [4^(-6) 4^(-5) 4^(-4) 4^(-3) 4^(-2) 4^(-1) 1 4 4^2];

for i=1:9
tempclock = clock();
starttime = tempclock(5) * 60 + tempclock(6);
temp_accuracy = zeros(5,1);

indices = crossvalind('Kfold',train_label,5);
for j = 1:5
test = (indices == j); train = ~test;
temp_train_data = train_data(train,:);
temp_train_label = train_label(train);
temp_test_data = train_data(test,:);
temp_test_label = train_label(test);
[temp_w temp_b] = trainsvm(temp_train_data,temp_train_label, cost(i));
temp_accuracy(j) = testsvm(temp_test_data,temp_test_label, temp_w, temp_b);
end

tempclock = clock();
endtime  = tempclock(5) * 60 + tempclock(6);
second = endtime  - starttime ;
average_time(i) = second/5;
accuracy(i) = sum(temp_accuracy)/5;
end
 
% Use linear SVM in LIBSVM
model = zeros(9,1);
tempclock = clock();
starttime = tempclock(5) * 60 + tempclock(6);
model(1) = svmtrain(train_label, train_data, '-t 0 -c 4^(−6) -v 5');
model_two = svmtrain(train_label, train_data, '-t 0 -c 4^(−5) -v 5');
model(3) = svmtrain(train_label, train_data, '-t 0 -c 4^(−4) -v 5');
model_four = svmtrain(train_label, train_data, '-t 0 -c 4^(−3) -v 5');
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
model_five = svmtrain(train_label, train_data, '-t 1 -c 4^(1) -d 1 -v 5');
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
 
 
