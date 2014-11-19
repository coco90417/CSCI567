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
three_observation = repmat(observation, k, 1);
distance = dot((three_observation-mu)',(three_observation-mu)');
[min_distance, index] = min(distance);
train_data(i, 1) = index;
end
[new_mu] = grpstats(train_data(:,2:3), train_data(:,1));

for i = 1:m
observation = train_data(i, 2:3);
index = train_data(i, 1);
cost_vector(iteration) = cost_vector(iteration) + dot((observation-new_mu(index,:))', (observation-new_mu(index,:))');
end

end

