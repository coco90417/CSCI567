function [A_estimate, E_estimate] = baumwelch(data, A_guess, E_guess, N_iter)
%
% Train Hidden Markov Model using the Baum-Welch algorithm (expectation maximization)
% Input:
%  data: N*T matrix, N data samples of length T
%  A_guess: K*K matrix, where K is the number hidden states [initial guess for the transition matrix]
%  E_guess: K*E matrix, where E is the number of emissions [initial guess for the emission matrix]
%
% Output:
%  A_estimate: estimate for the transition matrix after N_iter iterations of expectation-maximization 
%  E_estimate: estimate for the emission matrix after N_iter iterations of expectation-maximization
%
% CSCI 576 2014 Fall, Homework 5