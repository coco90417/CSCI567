%%%%%%%%%%%%%%%%%%%% boundary %%%%%%%%%%%%%%%%%%%%%%%
function [] = decision_tree(filename, k)

% CSCI 576 2014 Fall, Homework 1

filename = 'testing_20.pdf';
k = 20;

boundary = matfile('boundary.mat');


train_data = boundary.features;
train_label = boundary.labels;

for j = 1:100
temp_matrix = ones(100,2)* 0.01 * j-0.005;
temp_matrix(1:100,2) = [1:100]* 0.01-0.005;
x(((j-1)*100+1):(j*100)) = temp_matrix(1:100,1);
y(((j-1)*100+1):(j*100)) = temp_matrix(1:100,2);
s(((j-1)*100+1):(j*100)) = ones(100,1) * 20;
new_data = temp_matrix;
total_data = [train_data; new_data];
distance = pdist(total_data);
distanceMatrix = squareform(distance);
distanceMatrix = distanceMatrix((size(train_data,1)+1):size(total_data,1), 1:size(train_data,1));
sortedDistance = sort(distanceMatrix, 2);
for i=1:size(new_data,1)
tempIndex = find(distanceMatrix(i,:)<=sortedDistance(i,k));
finalIndexTest(i,1:k)= tempIndex(1:k);
end
finalIndexTest=finalIndexTest(:,1:k);
estimatedClass=mode(train_label(finalIndexTest),2);
plot_class(((j-1)*100+1):(j*100)) = estimatedClass;


end

h = figure;
c=plot_class*1000-2000;
scatter(x,y,s,c,'fill')
saveas(h, filename);
                              


%%%%%%%%%%%%%%%%%%%%%% decision_tree %%%%%%%%%%%%%%%%%%
function [new_accu_1, train_accu_1, new_accu_2, train_accu_2] = decision_tree(train_data, train_label, new_data, new_label, i)

% CSCI 576 2014 Fall, Homework 1

tree1 = ClassificationTree.fit(train_data, train_label, 'MinLeaf', i, 'SplitCriterion', 'gdi', 'Prune', 'off');
tree2 = ClassificationTree.fit(train_data, train_label, 'MinLeaf', i, 'SplitCriterion', 'deviance', 'Prune', 'off');

pred_tree1_train = tree1.predict(train_data);
pred_tree2_train = tree2.predict(train_data);

pred_tree1_new = tree1.predict(new_data);
pred_tree2_new = tree2.predict(new_data);

train_accu_1 = sum(pred_tree1_train == train_label)/size(train_data,1);
train_accu_2 = sum(pred_tree2_train == train_label)/size(train_data,1);

new_accu_1 = sum(pred_tree1_new == new_label)/size(new_data,1);
new_accu_2 = sum(pred_tree2_new == new_label)/size(new_data,1);




%%%%%%%%%%%%%%%%%% knn_classify %%%%%%%%%%%%%%%%%%%%%%

function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, k)
% modeling
distance = pdist(train_data);
distanceMatrix = squareform(distance);
distanceMatrix(distanceMatrix==0)=inf;
sortedDistance = sort(distanceMatrix, 2);

for i=1:size(train_data,1)
tempIndex = find(distanceMatrix(i,:)<=sortedDistance(i,k));
finalIndex(i,1:k)= tempIndex(1:k);
end
finalIndex=finalIndex(:,1:k);

estimatedClass=mode(train_label(finalIndex),2);
train_accu = size(train_data(estimatedClass==train_label),1)/size(train_data,1);

% testing
total_data = [train_data; new_data];
distance = pdist(total_data);
distanceMatrix = squareform(distance);
distanceMatrix = distanceMatrix((size(train_data,1)+1):size(total_data,1), 1:size(train_data,1));
sortedDistance = sort(distanceMatrix, 2);

for i=1:size(new_data,1)
tempIndex = find(distanceMatrix(i,:)<=sortedDistance(i,k));
finalIndexTest(i,1:k)= tempIndex(1:k);
end
finalIndexTest=finalIndexTest(:,1:k);

estimatedClass=mode(train_label(finalIndexTest),2);
new_accu = size(new_data(estimatedClass==new_label),1)/size(new_data,1);


%%%%%%%%%%%%%%%%% logistic_regression %%%%%%%%%%%%%%%


B=mnrfit(train_data,train_label);
tpihat=mnrval(B,train_data);
train_pred=zeros(size((train_label),1),1);
taccuracy=0;
for i=1:size((train_label),1)
train_pred(i)=find(tpihat(i,:)==max(tpihat(i,:)));
if train_pred(i)==train_label(i);
taccuracy=taccuracy+1;
end
end

train_accu=taccuracy/size((train_label),1);

npihat=mnrval(B,new_data);
new_pred=zeros(size((new_label),1),1);
naccuracy=0;
for i=1:size((new_label),1)
new_pred(i)=find(npihat(i,:)==max(npihat(i,:)));
if new_pred(i)==new_label(i);
naccuracy=naccuracy+1;
end
end

new_accu=naccuracy/size((new_label),1);

vpihat=mnrval(B,valid_data);
valid_pred=zeros(size((valid_label),1),1);
vaccuracy=0;
for i=1:size((valid_label),1)
valid_pred(i)=find(vpihat(i,:)==max(vpihat(i,:)));
if valid_pred(i)==valid_label(i);
vaccuracy=vaccuracy+1;
end
end

valid_accu=vaccuracy/size((valid_label),1);


%%%%%%%%%%%%%%%%%%% naive_bayes %%%%%%%%%%%%%%%%%%%%%

function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)

% trainmodel
xparameters = [sum(train_data(train_label==1,:))/size(train_label(train_label==1),1); sum(train_data(train_label==2,:))/size(train_label(train_label==2),1); sum(train_data(train_label==3,:))/size(train_label(train_label==3),1); sum(train_data(train_label==4,:))/size(train_label(train_label==4),1)];
xparameters(xparameters==0)=0.1;
yparameters = [size(train_label(train_label==1),1)/size(train_label,1), size(train_label(train_label==2),1)/size(train_label,1), size(train_label(train_label==3),1)/size(train_label,1), size(train_label(train_label==4),1)/size(train_label,1)];
logXparameters = log(xparameters);
logYparameters = log(yparameters);

%  train_accu: accuracy of classifying train_data
logEstimated = (train_data * transpose(logXparameters)+ repmat(logYparameters,size(train_label,1),1));
estimated = exp(logEstimated);
[estimatedMaxVal estimatedMaxInd] = max(estimated,[],2);
train_accu = size(train_data(estimatedMaxInd==train_label),1)/size(train_data,1);

%  new_accu: accuracy of classifying new_data
logEstimated = (new_data * transpose(logXparameters)+ repmat(logYparameters,size(new_label,1),1));
estimated = exp(logEstimated);
[estimatedMaxVal estimatedMaxInd] = max(estimated,[],2);
new_accu = size(new_data(estimatedMaxInd==new_label),1)/size(new_data,1);



