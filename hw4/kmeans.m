function [output_label, cost_vector] = kmeans(train_data, train_label, k)

[m n] = size(train_data);
mu = rand(k,n-1);
iteration=0;
output_label = zeros(m, 1);
class = train_label;

while(isequaln(class, output_label))
iteration=iteration+1;
cost_vector(iteration) = 0;
class = output_label;
for i = 1:m
observation = train_data(i, :);
copy_observation = repmat(observation, k, 1);
distance = sum((copy_observation-mu).^2, 2);
[min_distance, index] = min(distance);
output_label = index;
end

for i = 1:k
mu(i,:) = sum(train_data(output_label==i, :))/sum(output_label==i));
end

for i = 1:m
observation = train_data(i, :);
index = output_label(i);
cost_vector(iteration) = cost_vector(iteration) + sum((observation-mu(index,:)).^2, 2);
end

end

