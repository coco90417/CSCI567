% run CSCI567_hw4

% load files
[train_data] = loadfile();

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
test = (indices == j);
train = ~test;
temp_train_data = train_data(train,:);
temp_train_label = train_label(train);
temp_test_data = train_data(test,:);
temp_test_label = train_label(test);
[temp_w temp_b] = trainsvm(temp_train_data,temp_train_label, cost(i));
temp_accuracy(j) = testsvm(temp_test_data,temp_test_label, temp_w, temp_b);
end
tempclock = clock();
endtime  = tempclock(4) * 3600 + tempclock(5) * 60 + tempclock(6);
second = endtime  - starttime ;
average_time(i) = second/5;
accuracy(i) = sum(temp_accuracy)/5;
end

%{
accuracy =

0.5300    0.7890    0.8080    0.8181    0.8019    0.8020    0.8059    0.7979    0.7990
    
average_time =
    
0.4414    0.2862    0.3634    0.3444    0.4247    0.4212    0.5379    0.5138    0.5651
    
    %}

disp('linear SVM implementation')
disp('accuracy')
accuracy
disp('time')
average_time

% Usr linear SVM with c = 4^(-3)
[final_w, final_b] = trainsvm(train_data,train_label, cost(5));
disp('linear SVM implementation testing accuracy')
final_accuracy = testsvm(test_data,test_label, final_w, final_b)

%{
    final_accuracy =

    0.8451
%}

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
second(i) = (endtime  - starttime)/5 ;
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
    
    0.1036
    0.0939
    0.0750
    0.0664
    0.0661
    0.0997
    0.2885
    1.0019
    4.2508
%}


% Use poly SVM in LIBSVM
cost = [4^(-4) 4^(-3) 4^(-2) 4^(-1) 1 4 4^2 4^3 4^4 4^5 4^6 4^7];
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
    second(i,j) = (endtime  - starttime)/5;
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
    
    0.1039    0.1070    0.1075
    0.1044    0.1071    0.1079
    0.0946    0.1073    0.1080
    0.0757    0.1073    0.1082
    0.0677    0.1060    0.1134
    0.0677    0.1230    0.1157
    0.1018    0.1235    0.1161
    0.3057    0.1235    0.1160
    0.9039    0.1234    0.1161
    4.0074    0.1235    0.1161
    14.2799    0.1237    0.1161
    23.2817    0.1229    0.1149

%}



 % Use kernel SVM in LIBSVM

cost = [4^(-4) 4^(-3) 4^(-2) 4^(-1) 1 4 4^2 4^3 4^4 4^5 4^6 4^7];
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
        second(i,j) = (endtime  - starttime)/5 ;
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
    
    0.1234    0.1220    0.1222    0.1233    0.1241    0.1281    0.1280
    0.1219    0.1221    0.1224    0.1232    0.1245    0.1293    0.1306
    0.1220    0.1222    0.1225    0.1233    0.1244    0.1304    0.1313
    0.1221    0.1223    0.1225    0.1099    0.1099    0.1306    0.1326
    0.1227    0.1223    0.1032    0.0873    0.0988    0.1351    0.1361
    0.1224    0.1019    0.0844    0.0795    0.1218    0.1363    0.1347
    0.1009    0.0834    0.0756    0.0928    0.1220    0.1362    0.1347
    0.0829    0.0761    0.0899    0.1271    0.1220    0.1364    0.1347
    0.0756    0.0875    0.1550    0.1271    0.1220    0.1363    0.1349
    0.0886    0.1522    0.1995    0.1271    0.1221    0.1363    0.1347
    0.1466    0.3982    0.1995    0.1271    0.1220    0.1364    0.1349
    0.3636    0.5152    0.1995    0.1272    0.1221    0.1363    0.1347
%}

% final implementation
final_model = svmtrain(train_label, train_data, '-t 2 -c 4 -g 0.015625');
[predict_label, libsvm_accuracy, dec_values] = svmpredict(test_label, test_data, final_model);
    
disp('radial SVM testing accuracy')
libsvm_accuracy
% Accuracy = 90.4368% (1967/2175) (classification)
 
