function[nt_train_data_cross_entropy, nt_test_data_cross_entropy, nt_l2norm] = newtonmethod(train_data, train_label, test_data, test_label, w_in)

% train_data_cross_entropy, test_data_cross_entropy: 3d matrix (lambda, stepsize, T)
% l2norm: 3d matrix

% initialization
new_train_data = [ones(size(train_data, 1),1) train_data];
new_test_data = [ones(size(test_data, 1),1) test_data];
lambda = 0:0.05:0.5;
T = 1:1:50;
nt_train_data_cross_entropy = zeros(size(lambda,2),size(T,2));
nt_test_data_cross_entropy = zeros(size(lambda,2),size(T,2));
nt_l2norm = zeros(size(lambda,2),size(T,2));

% run
for i = 1:size(lambda,2)
for k = 1:size(T,2)
    if k == 1
    b = w_in(1);
    w = w_in(2:size(w_in,1));
    sigma = 1./(1+ exp(-(train_data * w + b)));
    temp_w = [b; w];
    end

    gradient = transpose(new_train_data) * (sigma - train_label) + 2*lambda(i)*[0; w];
    hessian =  transpose(new_train_data) * diag(sigma .* (ones(size(train_label,1),1)-sigma)) * new_train_data + 2*lambda(i)*diag(ones(size(temp_w,1),1));

    temp_w = temp_w - pinv(hessian) * gradient;
    b = temp_w(1);
    w = temp_w(2:size(temp_w,1));
    sigma = 1./(1+ exp(-(train_data * w + b)));
    sigma_test = 1./(1+ exp(-(test_data * w + b)));
    sigma(sigma<10^(-16)) = 10^(-16);
    sigma_test(sigma_test<10^(-16)) = 10^(-16);
    sigma(sigma>1-10^(-16)) = 1-10^(-16);
    sigma_test(sigma_test>1-10^(-16)) = 1-10^(-16);
    sigma_train = sigma;
nt_train_data_cross_entropy(i, k) = -(transpose(train_label) * log(sigma_train) + transpose(ones(size(train_label,1),1)-train_label) * log(ones(size(train_label,2),1)-sigma_train)) + lambda(i)*norm(w)^2;
nt_test_data_cross_entropy(i, k) = -(transpose(test_label) * log(sigma_test) + transpose(ones(size(test_label,1),1)-test_label) * log(ones(size(test_label,2),1)-sigma_test)) + lambda(i)*norm(w)^2;
nt_l2norm(i, k) = norm(w);
end
end
clear nt_train_data_cross_entropy nt_test_data_cross_entropy nt_l2norm;


