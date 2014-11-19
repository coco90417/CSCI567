% run CSCI567_hw4

% load files
[train_data, train_label] = loadfile();

% 5.2
% k = 2, 3, 5
disp('5.2 (a)')
k_vector = [2,3,5];
maxiteration = 500;
mode = 1
for i=1:3
[output_label, cost_vector] = kmeans(train_data, train_label, k_vector(i), maxiteration, mode);
name = strcat('scatter plot of k means clustering with k= ', num2str(k_vector(i)));
filename = strcat('question1_', num2str(k_vector(i)), '.pdf');
plotcluster(train_data, output_label, name, filename);
end



% k = 4, run 5 times
disp('5.2 (b)')
k = 4;
maxiteration = 50;
mode = 2
for i=1:5
[output_label, cost_vector] = kmeans(train_data, train_label, k, maxiteration, mode);
cost(i,1:50) = cost_vector;
end
filename = 'question2.pdf';
h=figure;
plot(cost');
title('cost of k means clustering with k=5 for five random initiations');
xlabel='iteration';
ylabel='cost';
legend('1st time','2nd time','3rd time','4th time','5th time');
saveas(h, filename);

     
     
% image reconstruction
disp('5.3 (d)')
k_vector = [3 8 15];
image = 'hw4.jpg';
data = double(imread(image));
[a b c] = size(data);
train_data = reshape(data, a*b, c);
output_data = train_data;
train_label = ones(size(train_data,1),1);
maxiteration = 500;
mode = 1;
for i = 1:3
[output_label, cost_vector] = kmeans(train_data, train_label, k_vector(i), maxiteration, mode);
for j = 1:k_vector(i)
mu(j,:) = sum(train_data(output_label==j, :))/sum(output_label==j);
output_data(output_label==j,:) = mu(j,:);
end
reshaped_output_data = reshape(output_data, a, b, c);
h=figure;
imshow(reshaped_output_data);
name = strcat('picture of k means clustering with k= ', num2str(k_vector(i)));
filename = strcat('question3_', num2str(k_vector(i)), '.pdf');
saveas(h,filename);
end
     

