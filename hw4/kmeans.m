function [train_data, cost_vector] = kmeans(train_data, k)

[m n] = size(train_data);
mu = rand(k,n-1);
new_mu = zeros(k,n-1);
iteration=0;

while(new_mu ~= mu)
iteration=iteration+1;
cost_vector(iteration) = 0;
if iteration > 1
mu = new_mu;
end
for i = 1:m
observation = train_data(i, 2:3);
copy_observation = repmat(observation, k, 1);
distance = dot((copy_observation-mu)',(copy_observation-mu)');
[min_distance, index] = min(distance);
train_data(i, 1) = index;
end

for i = 1:k
new_mu(i,1:2) = sum(train_data(train_data(:,1)==i, 2:3))/sum(train_data(train_data(:,1)==i));
end

for i = 1:m
observation = train_data(i, 2:3);
index = train_data(i, 1);
cost_vector(iteration) = cost_vector(iteration) + dot((observation-new_mu(index,:))', (observation-new_mu(index,:))');
end

end

