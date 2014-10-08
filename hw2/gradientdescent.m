function[train_data_cross_entropy, test_data_cross_entropy, l2norm, w_out] = gradientdescent(train_data, train_label, test_data, test_label)

% train_data_cross_entropy, test_data_cross_entropy: 3d matrix (lambda, stepsize, T)
% l2norm: 3d matrix


% >> train_data = iono_train_data;
% >> test_data = iono_test_data;
% >> train_label = iono_train_label;
% >> test_label = iono_test_label;

% initialization
b = 0.1;
w = zeros(size(train_data,2), 1);
new_train_data = [ones(size(train_data, 1),1) train_data];
new_test_data = [ones(size(test_data, 1),1) test_data];
lambda = 0:0.05:0.5;
stepsize = [0.001, 0.01, 0.05, 0.1, 0.5];
T = 1:1:50;
sigma = 1./(1+ exp(-(new_train_data * [b; w])));
gradient = zeros(size(train_data,2), 1);
train_data_cross_entropy = zeros(size(lambda,2),size(stepsize,2),size(T,2));
test_data_cross_entropy = zeros(size(lambda,2),size(stepsize,2),size(T,2));
l2norm = zeros(size(lambda,2),size(stepsize,2),size(T,2));

% run
for i = 1:size(lambda,2)
for j = 1:size(stepsize,2)
for k = 1:size(T,2)
gradient = transpose(new_train_data) * (sigma - train_label) + 2*lambda(i)*[0; w];
temp_w = [b; w];
temp_w = temp_w - stepsize(j) * gradient;
b = temp_w(1);
w = temp_w(2:size(temp_w,1));
sigma = 1./(1+ exp(-(new_train_data * [b; w])));
sigma_test = 1./(1 + exp(-(new_test_data * [b; w])));
sigma(sigma>1-10^(-16)) = 1-10^(-16);
sigma_test(sigma_test>1-10^(-16)) = 1-10^(-16);
sigma(sigma<10^(-16)) = 10^(-16);
sigma_test(sigma_test<10^(-16)) = 10^(-16);
sigma_train = sigma;
train_data_cross_entropy(i, j, k) = -(transpose(train_label) * log(sigma_train) + transpose(1-train_label) * log(1-sigma_train)) + lambda(i)*norm(w)^2
test_data_cross_entropy(i, j, k) = -(transpose(test_label) * log(sigma_test) + transpose(ones(size(test_label,1),1)-test_label) * log(ones(size(test_label,1),1)-sigma_test)) + lambda(i)*norm(w)^2;
l2norm(i, j, k) = norm(w);
end

if i == 2 & j == 2
w_out = [b; w];
end

end
end


