% run CSCI567_hw4

% load files
[train_data, train_label] = loadfile();

% 5.2
% k = 2, 3, 5
disp('5.2 (a)')
k_vector = [2,3,5];
maxiteration = 10000000000;
for i=1:3
[output_label, cost_vector] = kmeans(train_data, train_label, k_vector(i), maxiteration);
name = strcat('scatter plot of k means clustering with k= ', num2str(k_vector(i)));
filename = strcat('question1_', num2str(k_vector(i)), '.pdf');
plotcluster(train_data, output_label, name, filename);
end



% k = 4, run 5 times
disp('5.2 (b)')
k = 4;
maxiteration = 50;
for i=1:5
[output_label, cost_vector] = kmeans(train_data, train_label, k, maxiteration);
name = strcat('cost of k means clustering with k=5 time=', num2str(i));
filename = strcat('question2_', num2str(i), '.pdf');
plotcost(cost_vector);

