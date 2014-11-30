% run CSCI567_hw5

% load files
loadfile();

% 4.c
disp("4.c")
eigenvecs = pca_fun(X, d);
filename = strcat('question4c_', num2str(i), '.jpg');
imwrite(final_data,filename,'jpg')
