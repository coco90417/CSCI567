% run CSCI567_hw5

% load files
[face_data, hmm_data] = loadfile();

% 4.c
disp("4.c")
eigenvecs = pca_fun(X, d);
filename = strcat('question4c_', num2str(i), '.jpg');
imwrite(final_data,filename,'jpg')
