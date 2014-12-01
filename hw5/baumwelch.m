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


A_guess = [0.7 0.3;0.3 0.7];
B_guess = [0.25 0.25 0.25 0.25;0.25 0.25 0.25 0.25];
pi = [0.8 0.2];
my_N_iter = 1;

while( my_N_iter <= N_iter)

% alpha(time, state, sample)
% initialization
for n = 1:1000
for i = 1:2
    alpha(1,i,n) = pi(i)*B_guess(i,data(n,1));
end
    c(1,n) = 1/sum(alpha(1,:,n));
    for i = 1:2
    alpha(1,i,n) = c(1,n)*alpha(1,i,n);
end

% update
for t=2:6
for n=1:1000
for i=1:2
alpha(t,i,n) = sum(alpha(t-1,:,n).*A_guess(:,i).*B_guess(i,data(n,t)));
end
    c(t,n) = 1/sum(alpha(t,:,n));
for i = 1:2
    alpha(t,i,n) = c(t,n)*alpha(t,i,n);
end

% beta(time, state, sample)
% initialization
for n = 1:1000
for i = 1:2
beta(6,i,n) = 1;
beta(6,i,n) = c(6,n)*beta(6,i,n);
end
end

% update
for t=1:5
for n=1:1000
for i=1:2
beta_t = 6-t;
beta(beta_t,i,n) = sum(A_guess(i,:).*B_guess(:,data(n,beta_t+1)));
beta(beta_t,i,n) = c(beta_t,n)*beta(beta_t,i,n);
end
end
end

% parameter

A_estimate = zeros(2,2);
for i = 1:2
for j = 1:2
A_estimate_temp_up = 0;
A_estimate_temp_down = 0;
for m = 1:1000
A_estimate_temp_up(m) = sum(alpha(1:5,i,m).*A_guess(i,j).*B_guess(j,data(m,2:6)).*beta(2:6,j,m));
A_estimate_temp_down(m) = sum(alpha(1:5,i,m).*beta(i,j,m));
end
A_estimate(i,j) = sum(A_estimate_temp_up)/sum(A_estimate_temp_down);
end
end

B_estimate = zeros(2,4);
for i = 1:2
for m = 1:size(data(:,k)==k,1)
temp_alpha_one = alpha(data(m,:)==1,i,m);
temp_beta_one = beta(data(m,:)==1,i,m);
temp_alpha_two = alpha(data(m,:)==2,i,m);
temp_beta_two = beta(data(m,:)==2,i,m);
temp_alpha_three = alpha(data(m,:)==3,i,m);
temp_beta_three = beta(data(m,:)==3,i,m);
temp_alpha_four = alpha(data(m,:)==4,i,m);
temp_beta_four = beta(data(m,:)==4,i,m);
B_estimate_temp_up_one(m) = sum(temp_alpha_one(:,i,m).*temp_beta_one(:,i,m));
B_estimate_temp_up_two(m) = sum(temp_alpha_two(:,i,m).*temp_beta_two(:,i,m));
B_estimate_temp_up_three(m) = sum(temp_alpha_three(:,i,m).*temp_beta_three(:,i,m));
B_estimate_temp_up_four(m) = sum(temp_alpha_four(:,i,m).*temp_beta_four(:,i,m));
B_estimate_temp_down_one(m) = sum(alpha_one(:,i,m).*beta_one(:,i,m));
B_estimate_temp_down_two(m) = sum(alpha_one(:,i,m).*beta_one(:,i,m));
B_estimate_temp_down_three(m) = sum(alpha_one(:,i,m).*beta_one(:,i,m));
B_estimate_temp_down_four(m) = sum(alpha_one(:,i,m).*beta_one(:,i,m));
end
B_estimate(i,1) = sum(B_estimate_temp_up_one)/sum(B_estimate_temp_down_one);
B_estimate(i,2) = sum(B_estimate_temp_up_two)/sum(B_estimate_temp_down_two);
B_estimate(i,3) = sum(B_estimate_temp_up_three)/sum(B_estimate_temp_down_three);
B_estimate(i,4) = sum(B_estimate_temp_up_four)/sum(B_estimate_temp_down_four);
end


A_guess = A_estimate;
B_guess = B_estimate;
end






