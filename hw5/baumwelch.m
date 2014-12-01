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



pi = [0.1 0.9];
alpha = zeros(6, 2, size(data,1));
beta = zeros(6, 2, size(data,1));


my_N_iter = 1;

while( my_N_iter <= N_iter)

% alpha(time, state, sample)
% initialization
for n = 1:size(data, 1)
    for i = 1:2
    alpha(1,i,n) = pi(i)*E_guess(i,data(n,1));
    end
    c(1,n) = 1/sum(alpha(1,:,n));
    for i = 1:2
    alpha(1,i,n) = c(1,n)* alpha(1,i,n);
    end
end

% update
for t=2:6
for n=1:size(data, 1)
for i=1:2
alpha(t,i,n) = sum(alpha(t-1,:,n)'.* A_guess(:,i).* E_guess(i,data(n,t)));
end
c(t,n) = 1/sum(alpha(t,:,n));
for i = 1:2
alpha(t,i,n) = c(t,n)*alpha(t,i,n);
end
end
end

% beta(time, state, sample)
% initialization
for n = 1:size(data, 1)
for i = 1:2
beta(6,i,n) = c(6,n);
end
end

% update
for t=1:5
for n=1:size(data, 1)
for i=1:2
beta_t = 6-t;
beta(beta_t,i,n) = sum(A_guess(i,:)'.* E_guess(:,data(n,beta_t+1)).* beta(beta_t+1,:,n)');
beta(beta_t,i,n) = c(beta_t,n) * beta(beta_t,i,n);
end
end
end

% parameter

A_estimate = zeros(2,2);
E_estimate = zeros(2,4);
for i = 1:2
for j = 1:2
A_estimate_temp_up = 0;
A_estimate_temp_down = 0;
for m = 1:size(data, 1)
A_estimate_temp_up(m) = sum(alpha(1:5,i,m)*A_guess(i,j).*E_guess(j,data(m,2:6))'.*beta(2:6,j,m));
A_estimate_temp_down(m) = sum(alpha(1:5,i,m).*beta(1:5,i,m)./c(1:5,m));
end
A_estimate(i,j) = sum(A_estimate_temp_up)/sum(A_estimate_temp_down);
end
end

E_estimate = zeros(2,4);
E_estimate_temp_down_one = zeros(size(data, 1),1);

for i = 1:2
for m = 1:size(data, 1)
temp_alpha_one = alpha(data(m,:)==1,i,m);
temp_beta_one = beta(data(m,:)==1,i,m);
c_one = c(data(m,:)==1,m);
temp_alpha_two = alpha(data(m,:)==2,i,m);
temp_beta_two = beta(data(m,:)==2,i,m);
c_two = c(data(m,:)==2,m);
temp_alpha_three = alpha(data(m,:)==3,i,m);
temp_beta_three = beta(data(m,:)==3,i,m);
c_three = c(data(m,:)==3,m);
temp_alpha_four = alpha(data(m,:)==4,i,m);
temp_beta_four = beta(data(m,:)==4,i,m);
c_four = c(data(m,:)==4,m);
E_estimate_temp_up_one(m) = sum(temp_alpha_one.*temp_beta_one ./c_one);
E_estimate_temp_up_two(m) = sum(temp_alpha_two.*temp_beta_two ./c_two);
E_estimate_temp_up_three(m) = sum(temp_alpha_three.*temp_beta_three ./c_three);
E_estimate_temp_up_four(m) = sum(temp_alpha_four.*temp_beta_four ./c_four);
E_estimate_temp_down_one(m) = sum(alpha(:,i,m).*beta(:,i,m)./c(:,m));
E_estimate_temp_down_two(m) = sum(alpha(:,i,m).*beta(:,i,m)./c(:,m));
E_estimate_temp_down_three(m) = sum(alpha(:,i,m).*beta(:,i,m)./c(:,m));
E_estimate_temp_down_four(m) = sum(alpha(:,i,m).*beta(:,i,m)./c(:,m));
end
E_estimate(i,1) = sum(E_estimate_temp_up_one)/sum(E_estimate_temp_down_one);
E_estimate(i,2) = sum(E_estimate_temp_up_two)/sum(E_estimate_temp_down_two);
E_estimate(i,3) = sum(E_estimate_temp_up_three)/sum(E_estimate_temp_down_three);
E_estimate(i,4) = sum(E_estimate_temp_up_four)/sum(E_estimate_temp_down_four);
end


A_guess = A_estimate;
E_guess = E_estimate;
my_N_iter = my_N_iter + 1;
end
                            
                            






