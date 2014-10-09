function[train_data_cross_entropy, test_data_cross_entropy, l2norm, w_out] = gradientdescent(train_data, train_label, test_data, test_label)

% train_data_cross_entropy, test_data_cross_entropy: 3d matrix (lambda, stepsize, T)
% l2norm: 3d matrix

% initialization

lambda = 0:0.05:0.5;
stepsize = [0.001, 0.01, 0.05, 0.1, 0.5];
T = 1:1:50;
train_data_cross_entropy = zeros(size(stepsize,2),size(T,2),size(lambda,2));
test_data_cross_entropy = zeros(size(stepsize,2),size(T,2),size(lambda,2));
l2norm = zeros(size(stepsize,2),size(T,2),size(lambda,2));

% run
for i = 1:size(lambda,2)
for j = 1:size(stepsize,2)
for k = 1:size(T,2)

if k == 1
b = 0.1;
w = zeros(size(train_data,2), 1);
sigma = 1./(1+ exp(-(train_data * w + b)));
gradient = zeros(size(train_data,2), 1);
end

b = b - stepsize(j) *(sum(sigma-train_label));
w = w - stepsize(j) *(train_data' * (sigma - train_label) + 2 * lambda(i) * w);
sigma = 1./(1+ exp(-(train_data * w + b)));
sigma_test = 1./(1+ exp(-(test_data * w + b)));
sigma(sigma<10^(-16)) = 10^(-16);
sigma(sigma>1-10^(-16)) = 1-10^(-16);
sigma_test(sigma_test<10^(-16)) = 10^(-16);
sigma_test(sigma_test>1-10^(-16)) = 1-10^(-16);
sigma_train = sigma;
temp_train_data_cross_entropy(j, k) = -(train_label' * log(sigma_train) + (1-train_label)' * log(1-sigma_train)) + lambda(i)*norm(w,2)^2;
temp_test_data_cross_entropy(j, k) = -(test_label' * log(sigma_test) + (1-test_label)' * log(1-sigma_test)) + lambda(i)*norm(w,2)^2;
temp_l2norm(j, k) = norm(w,2);
if i == 2 & j == 2 & k == 5
w_out = [b; w];
end
end
end
train_data_cross_entropy(:, :, i) = temp_train_data_cross_entropy;
test_data_cross_entropy(:,:,i) = temp_test_data_cross_entropy;
l2norm(:,:,i) = temp_l2norm;
clear b w sigma temp_l2norm temp_test_data_cross_entropy temp_train_data_cross_entropy;
end

                    

                      

                      
                   
