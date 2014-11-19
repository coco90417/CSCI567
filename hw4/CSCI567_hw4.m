% run CSCI567_hw4

% load files
[train_data, train_label] = loadfile();

% 5.2
% k = 2, 3, 5
disp('5.2 (a)')
k_vector = [2,3,5];
maxiteration = 200;
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
cost(i,1:50) = cost_vector;
end
filename = 'question2.pdf';
h=figure;
plot(cost);
title('cost of k means clustering with k=5 for five random initiations');
xlabel='iteration';
ylabel='cost';
legend('1st time','2nd time','3rd time','4th time','5th time');
save(h, filename);

