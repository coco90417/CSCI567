function[train_data_cross_entropy, test_data_cross_entropy, l2norm] = newtonmethod(train_data, train_label, test_data, test_label, w_in)

% train_data_cross_entropy, test_data_cross_entropy: 3d matrix (lambda, stepsize, T)
% l2norm: 3d matrix

% initialization
b = w_in(1);
w = w_in(2:size(w_in,1));
new_train_data = [ones(size(train_data, 1),1) train_data];
new_test_data = [ones(size(test_data, 1),1) test_data];
lambda = 0:0.05:0.5;
T = 1:1:50;
sigma = 1./(ones(size(train_label,1),1)+ exp(-(new_train_data * [b; w])));


% run
for i = 1:size(lambda,2)
for k = 1:size(T,2)
    temp_w = [b; w];
    gradient = transpose(new_train_data) * (sigma - train_label) + 2*lambda(i)*[0; w];
    hessian = - transpose(new_train_data) * diag(sigma .* (ones(size(train_label,1),1)-sigma)) * new_train_data + 2*ones(size(temp_w,1));
    temp_w = temp_w - pinv(hessian) * gradient;
    b = temp_w(1);
    w = temp_w(2:size(temp_w,1));
    sigma = 1./(ones(size(train_label,1),1)+ exp(-(new_train_data * [b; w])));
    sigma_train = 1./(ones(size(train_label,1),1) + exp(-(new_train_data * [b; w])));
    sigma_test = 1./(ones(size(test_label,1),1) + exp(-(new_test_data * [b; w])));
    sigma(sigma>1-10^(-16)) = 1-10^(-16);
    sigma(sigma<10^(-16)) = 10^(-16);
    sigma_train(sigma_train>1-10^(-16)) = 1-10^(-16);
    sigma_test(sigma_test>1-10^(-16)) = 1-10^(-16);
    sigma_train(sigma_train<10^(-16)) = 10^(-16);
    sigma_test(sigma_test<10^(-16)) = 10^(-16);
    train_data_cross_entropy(i, k) = -(transpose(train_label) * log(sigma_train) + transpose(ones(size(train_label,1),1)-train_label) * log(ones(size(train_label,2),1)-sigma_train)) + lambda(i)*norm(w)^2;
    test_data_cross_entropy(i, k) = transpose(test_label) * log(sigma_test) + transpose(ones(size(test_label,1),1)-test_label) * log(ones(size(train_label,2),1)-sigma_test) + lambda(i)*norm(w)^2;
    l2norm(i, k) = norm(w);
end
end


