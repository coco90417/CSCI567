function eigenvecs = pca_fun(X, d)

% Implementation of PCA
% input:
%   X - N*D data matrix, each row as a data sample
%   d - target dimensionality, d <= D
% output:
%   eigenvecs: D*d matrix
%
% usage:
%   eigenvecs = pca_fun(X, d);
%   projection = X*eigenvecs;
%
% CSCI 576 2014 Fall, Homework 5

[m,n] = size(X);
centeredX = X - repmat(mean(X,1),m, 1);
sigma = 1/m * centeredX' * centeredX;
[v,D] = eig(sigma);
eigenvecs = v(:,1:d);
