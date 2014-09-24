function [] = decision_tree(filename, k)

% CSCI 576 2014 Fall, Homework 1

boundary = matfile('boundary.mat');


train_data = boundary.features;
train_label = boundary.labels;
new_data = (1:100, 1:100);
new_data = new_data * 0.01-0.005;

for j = 1:100
temp_matrix = ones(2,100);


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



h = figure;
s=0.1;
c=boundary
scatter(x,y,s,c,'fill')
saveas(h, filename);
                              
