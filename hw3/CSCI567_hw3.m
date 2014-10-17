% load files
[train_data, train_label, test_data, test_label] = loadfile();


% add path, required for libsvm
addpath('/home/cocodong/usr/matlab/lib/libsvm-3.18/matlab');

% implement linear SVM
cost = [4^(-6) 4^(-5) 4^(-4) 4^(-3) 4^(-2) 4^(-1) 1 4 4^2];

for i=1:9
tempclock = clock();
starttime = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
temp_accuracy = zeros(5,1);
indices = crossvalind('Kfold',train_label,5);
for j = 1:5
test = (indices == j); train = ~test;
temp_train_data = train_data(train,:);
temp_train_label = train_label(train);
temp_test_data = train_data(test,:);
temp_test_label = train_label(test);
[temp_w temp_b] = trainsvm(temp_train_data,temp_train_label, cost(i));
temp_accuracy(j) = testsvm(temp_test_data,temp_test_label, temp_w, temp_b)
end
tempclock = clock();
endtime  = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
second = endtime  - starttime ;
average_time(i) = second/5
accuracy(i) = sum(temp_accuracy)/5
end

disp('linear SVM implementation')
disp('accuracy')
accuracy
disp('time')
average_time

% Usr linear SVM with c = 4^(-2)
[final_w, final_b] = trainsvm(train_data,train_label, cost(5));
disp('linear SVM implementation testing accuracy')
final_accuracy = testsvm(test_data,test_label, final_w, final_b)



% Use linear SVM in LIBSVM
cost = [4^(-6) 4^(-5) 4^(-4) 4^(-3) 4^(-2) 4^(-1) 1 4 4^2];
model = zeros(length(cost),1);
second = zeros(length(cost),1);
for i=1:length(cost)
tempclock = clock();
starttime = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
f = strcat({'-t 0 -c '}, num2str(cost(i)), {' -v 5'});
model(i) = svmtrain(train_label, train_data, f{1});
tempclock = clock();
endtime  = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
second(i) = endtime  - starttime ;
end

disp('linear SVM in LIBSVM')
disp('accuracy')
model
disp('time')
second

%{
model =

51.7000
78.5000
79.9000
80.7000
79.5000
79.5000
79.3000
79.1000
79.1000


second =

0.5222
0.4781
0.3810
0.3373
0.3355
0.5010
1.4413
4.9881
20.9647
%}


% Use poly SVM in LIBSVM
cost = [4^(-3) 4^(-2) 4^(-1) 1 4 4^2 4^3 4^4 4^5 4^6 4^7];
degree = [1 2 3];
model = zeros(length(cost),length(degree));
second = zeros(length(cost),length(degree));
for i=1:length(cost)
    for j=1:length(degree)
        tempclock = clock();
        starttime = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
        f = strcat({'-t 1 -c '}, num2str(cost(i)), {' -d '}, num2str(degree(j)), {' -v 5'});
        model(i,j) = svmtrain(train_label, train_data, f{1})
        tempclock = clock();
        endtime  = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
        second(i,j) = endtime  - starttime
    end
end

disp('poly SVM in LIBSVM')
disp('accuracy')
model
disp('time')
second

%{
model =

51.7000   51.7000   51.7000
78.3000   51.7000   51.7000
80.0000   64.5000   62.0000
80.8000   72.9000   80.3000
79.3000   71.4000   80.6000
79.5000   71.3000   80.6000
79.1000   71.3000   80.6000
79.0000   71.3000   80.6000
79.1000   71.3000   80.6000
78.9000   71.3000   80.6000
79.0000   71.3000   80.6000



second =

0.5278    0.5402    0.5440
0.4776    0.5415    0.5449
0.3837    0.5503    0.5486
0.3424    0.5373    0.5778
0.3450    0.6255    0.5873
0.5180    0.6308    0.5917
1.5454    0.6296    0.5893
4.5673    0.6258    0.5865
20.2202    0.6282    0.5889
72.1096    0.6291    0.5921
116.7945    0.6253    0.5864

%}



 % Use kernel SVM in LIBSVM

cost = [4^(-3) 4^(-2) 4^(-1) 1 4 4^2 4^3 4^4 4^5 4^6 4^7];
gamma = [4^(-7) 4^(-6) 4^(-5) 4^(-4) 4^(-3) 4^(-2) 4^(-1)];
model = zeros(length(cost),length(gamma));
second = zeros(length(cost),length(gamma));
for i=1:length(cost)
    for j=1:length(gamma)
        tempclock = clock();
        starttime = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
        f = strcat({'-t 2 -c '}, num2str(cost(i)), {' -g '}, num2str(gamma(j)), {' -v 5'});
        model(i,j) = svmtrain(train_label, train_data, f{1});
        tempclock = clock();
        endtime  = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
        second(i,j) = endtime  - starttime ;
    end
end


disp('radial SVM in LIBSVM')
disp('accuracy')
model
disp('time')
second

%{
model =

51.7000   51.7000   51.7000   51.7000   51.7000   51.7000   51.7000
51.7000   51.7000   51.7000   54.3000   61.6000   51.7000   51.7000
51.7000   51.7000   73.9000   80.2000   83.8000   51.7000   51.7000
51.7000   74.9000   79.4000   82.9000   85.8000   77.0000   57.7000
75.8000   79.9000   81.0000   84.4000   86.5000   78.4000   58.1000
79.6000   80.7000   82.2000   86.4000   86.5000   78.4000   58.1000
80.5000   80.5000   84.3000   86.4000   86.5000   78.4000   58.1000
79.8000   81.6000   86.0000   86.4000   86.5000   78.4000   58.1000
80.3000   84.2000   86.3000   86.4000   86.5000   78.4000   58.1000
81.3000   85.7000   86.3000   86.4000   86.5000   78.4000   58.1000
84.1000   86.4000   86.3000   86.4000   86.5000   78.4000   58.1000


second =

0.6214    0.6162    0.6184    0.6229    0.6278    0.6532    0.6613
0.6207    0.6195    0.6204    0.6251    0.6313    0.6588    0.6676
0.6161    0.6173    0.6182    0.5546    0.5552    0.6590    0.6682
0.6167    0.6162    0.5217    0.4412    0.4989    0.6817    0.6874
0.6178    0.5126    0.4263    0.4017    0.6152    0.6924    0.6795
0.5093    0.4212    0.3814    0.4676    0.6158    0.6869    0.6792
0.4187    0.3835    0.4532    0.6434    0.6165    0.6872    0.6796
0.3811    0.4418    0.7851    0.6411    0.6162    0.6877    0.6789
0.4466    0.7662    1.0052    0.6407    0.6156    0.6866    0.6795
0.7371    2.0017    1.0034    0.6410    0.6160    0.6864    0.6786
1.8288    2.5924    1.0041    0.6414    0.6153    0.6863    0.6791
%}

% final implementation
final_model = svmtrain(train_label, train_data, '-t 2 -c 4 -g 0.015625');
[predict_label, libsvm_accuracy, dec_values] = svmpredict(test_label, test_data, final_model);
    
disp('radial SVM testing accuracy')
libsvm_accuracy
% Accuracy = 90.4368% (1967/2175) (classification)
 
