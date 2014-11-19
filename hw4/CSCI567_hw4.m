% run CSCI567_hw4

% load files
[train_data] = loadfile();

% k = 2, 3, 5
k_vector = [2,3,5];
for i=1:3
[train_data, cost_vector] = kmeans(train_data, k_vector(i));
name = strcat('scatter plot of k means clustering with k= ', num2str(k_vector(i)));
filename = strcat('question1_', num2str(k_vector(i)), '.pdf');
plotcluster(train_data, name, filename);
end