% run CSCI567_hw4

% load files
[train_data] = loadfile();

% k = 2, 3, 5
k = [2,3,5];
for i=1:3
[train_data, cost_vector] = kmeans(train_data, k(i));
name = strcat('scatter plot of k means clustering with k= ', num2str(k(i)));
filename = strcat('qustion1_', num2str(k(i)), '.pdf');
plot(train_data, name, filename);
end