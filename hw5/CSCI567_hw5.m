% run CSCI567_hw5

% add path, required for libsvm
addpath('/home/cocodong/usr/matlab/lib/libsvm-3.18/matlab');

% load files
load('face_data.mat');
load('hmm_data.mat');

% 4.c
disp('4.c')
for i = 1:640
[m n] = size(image{i});
image_data(i, :) = reshape(image{i}, [], 1);
end
d = 5;
image_eigenvecs = pca_fun(image_data, d);

for temp_d = 1:d
filename = strcat('question4c_', num2str(temp_d), '.jpg');
h = figure;
one_eigenvec = image_eigenvecs(:,temp_d);
final_data = reshape(one_eigenvec, m, n);
imshow(final_data,[]);
saveas(h, filename);
end


% 4.d
disp('4.d')
dimension = [20 50 100 200];
cost = [4^(-6) 4^(-5) 4^(-4) 4^(-3) 4^(-2) 4^(-1) 1 4 4^2];
gamma = [4^(-7) 4^(-6) 4^(-5) 4^(-4) 4^(-3) 4^(-2) 4^(-1)];



% d=1
d = 1
image_eigenvecs = pca_fun(image_data, dimension(d));
new_image_data = image_data * image_eigenvecs;
new_label = personID';
for i=1:length(cost)
linear_f = strcat({'-t 0 -c '}, num2str(cost(i)), {' -v 5'});
linear_acc(i) = svmtrain(double(new_label), double(new_image_data), linear_f{1});
for j=1:length(gamma)
radial_f = strcat({'-t 2 -c '}, num2str(cost(i)), {' -g '}, num2str(gamma(j)), {' -v 5'});
radial_acc(i,j) = svmtrain(double(new_label), double(new_image_data), radial_f{1});
end
end

%{
    linear
    25.9375   34.5312   62.0312   79.0625   86.4062   90.6250   93.7500   92.9688   89.6875
    
    radial
    26.0938   27.0312   30.9375   35.3125   47.3438   40.7812   24.0625
    26.0938   27.0312   30.9375   35.3125   47.3438   40.7812   24.0625
    26.0938   27.0312   30.9375   35.3125   47.3438   40.7812   24.0625
    26.0938   27.0312   30.9375   35.3125   47.3438   40.7812   24.0625
    26.0938   27.0312   30.9375   35.3125   47.3438   40.7812   24.0625
    26.0938   27.0312   30.9375   41.5625   51.4062   41.7188   24.0625
    26.0938   29.2188   48.2812   63.9062   70.4688   62.8125   41.0938
    29.0625   49.6875   69.2188   81.8750   82.9688   71.2500   43.1250
    49.8438   70.1562   82.1875   89.0625   87.9688   71.5625   43.5938
    
}


% d=2
d = 2
image_eigenvecs = pca_fun(image_data, dimension(d));
new_image_data = image_data * image_eigenvecs;
new_label = personID';
for i=1:length(cost)
linear_f = strcat({'-t 0 -c '}, num2str(cost(i)), {' -v 5'});
linear_acc(i) = svmtrain(double(new_label), double(new_image_data), linear_f{1});
for j=1:length(gamma)
radial_f = strcat({'-t 2 -c '}, num2str(cost(i)), {' -g '}, num2str(gamma(j)), {' -v 5'});
radial_acc(i,j) = svmtrain(double(new_label), double(new_image_data), radial_f{1});
end
end



%{
    linear
    29.0625   39.2188   71.2500   87.8125   92.9688   94.2188   95.4688   93.5938   92.5000
    
    radial
    29.2188   30.3125   35.3125   42.5000   55.7812   43.5938   24.3750
    29.2188   30.3125   35.3125   42.5000   55.7812   43.5938   24.3750
    29.2188   30.3125   35.3125   42.5000   55.7812   43.5938   24.3750
    29.2188   30.3125   35.3125   42.5000   55.7812   43.5938   24.3750
    29.2188   30.3125   35.3125   42.5000   55.7812   43.5938   24.3750
    29.2188   30.3125   35.4688   49.8438   61.8750   44.5312   24.3750
    29.2188   32.9688   55.1562   77.5000   84.6875   76.8750   42.8125
    32.3438   55.1562   79.0625   90.4688   90.4688   79.2188   45.3125
    55.3125   79.6875   90.3125   93.7500   93.2812   79.5312   45.4688
}


% d=3
d = 3
image_eigenvecs = pca_fun(image_data, dimension(d));
new_image_data = image_data * image_eigenvecs;
new_label = personID';
for i=1:length(cost)
linear_f = strcat({'-t 0 -c '}, num2str(cost(i)), {' -v 5'});
linear_acc(i) = svmtrain(double(new_label), double(new_image_data), linear_f{1});
for j=1:length(gamma)
radial_f = strcat({'-t 2 -c '}, num2str(cost(i)), {' -g '}, num2str(gamma(j)), {' -v 5'});
radial_acc(i,j) = svmtrain(double(new_label), double(new_image_data), radial_f{1});
end
end

%{
    linear
    29.0625   39.8438   73.2812   91.2500   94.5312   95.1562   95.3125   94.6875   93.5938
    
    radial
    29.2188   30.0000   35.6250   43.1250   56.0938   40.6250   24.0625
    29.2188   30.0000   35.6250   43.1250   56.0938   40.6250   24.0625
    29.2188   30.0000   35.6250   43.1250   56.0938   40.6250   24.0625
    29.2188   30.0000   35.6250   43.1250   56.0938   40.6250   24.0625
    29.2188   30.0000   35.6250   43.1250   56.0938   40.6250   24.0625
    29.2188   30.0000   35.7812   50.1562   63.1250   40.6250   24.0625
    29.2188   32.8125   56.2500   79.3750   86.5625   77.1875   41.2500
    32.3438   56.5625   82.6562   92.9688   92.1875   79.2188   43.7500
    55.7812   83.4375   93.1250   94.5312   93.4375   79.0625   43.9062
    
}


% d=4
d = 4
image_eigenvecs = pca_fun(image_data, dimension(d));
new_image_data = image_data * image_eigenvecs;
new_label = personID';
for i=1:length(cost)
linear_f = strcat({'-t 0 -c '}, num2str(cost(i)), {' -v 5'});
linear_acc(i) = svmtrain(double(new_label), double(new_image_data), linear_f{1});
for j=1:length(gamma)
radial_f = strcat({'-t 2 -c '}, num2str(cost(i)), {' -g '}, num2str(gamma(j)), {' -v 5'});
radial_acc(i,j) = svmtrain(double(new_label), double(new_image_data), radial_f{1});
end
end


%{
    linear
    
    28.7500   39.6875   73.2812   92.0312   95.0000   96.2500   95.4688   95.3125   94.6875
    
    
    radial
    28.7500   30.0000   35.4688   43.4375   55.7812   39.6875   24.6875
    28.7500   30.0000   35.4688   43.4375   55.7812   39.6875   24.6875
    28.7500   30.0000   35.4688   43.4375   55.7812   39.6875   24.6875
    28.7500   30.0000   35.4688   43.4375   55.7812   39.6875   24.6875
    28.7500   30.0000   35.4688   43.4375   55.7812   39.6875   24.6875
    28.7500   30.0000   35.6250   50.4688   62.8125   39.6875   24.6875
    28.7500   32.8125   56.2500   79.2188   86.8750   75.6250   39.3750
    31.8750   56.2500   82.9688   92.9688   92.5000   77.8125   42.3438
    55.7812   83.7500   93.2812   95.1562   93.4375   78.4375   42.3438
}



%{
me:
    A_estimate
    0.9171    0.0829
    0.0729    0.9271
    
    
    B_estimate
    0.0979    0.4303    0.3924    0.0794
    0.3868    0.1023    0.1146    0.3963
hmm:
    
    
}



