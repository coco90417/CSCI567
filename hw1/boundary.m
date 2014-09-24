function [] = decision_tree(filename, k)

% CSCI 576 2014 Fall, Homework 1

boundary = matfile('boundary.mat');


train_data = boundary.features;
train_label = boundary.labels;

for j = 1:100
temp_matrix = ones(100,2)* 0.01 * j-0.005;
temp_matrix(1:100,2) = [1:100]* 0.01-0.005;
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
plot_class(j:(j+99)) = estimatedClass;
x(j:(j+99)) = ones(100,1)* 0.01 * j-0.005;
y(j:(j+99)) = [1:100]* 0.01-0.005;
s(j:(j+99)) = ones(100,1) * 20;

end

h = figure;
c=plot_class*1000-2000;
scatter(x,y,s,c,'fill')
saveas(h, filename);
                              
