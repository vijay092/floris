addpath('./tools/')
% Companion program for "Multi-Agent Reinforcement Learning via Double
% Averaging Primal-Dual Optimization", 
% H.-T. Wai, Z. Yang, Z. Wang, M. Hong, in NeurIPS 2018.
% Last updated on 12.24.2018

% Some of the functions used are shared by Simon Du (CMU) from the paper
% "Stochastic Variance Reduction Methods for Policy Evaluation", ICML 2017.

% read data (it's actually Sarsa/Mountaincar data)
data.phi = csvread('./data/state_wf_300.csv')';
data.phi_next = csvread('./data/nextstate_wf_300.csv')';
data.rewards = csvread('./data/reward_wf_300.csv');
data.gamma = 0.95;

%Training
[d,n] = size(data.phi);
n_training = 5000;
n_per_run = n_training; %should > n_training
n_runs_used = 1;

data_tr = data;

M = 70; %total epochs
N = 3; % no of agents

rho = 0.01;

%string contains all algorithm name
algorithms = ['DistIAG'];

%empirical optimal
theta_optimal = zeros(d,n_runs_used);

%setup 
%DistSAGA algorithm
if ~isempty(strfind(algorithms,'DistIAG'))
    M0_saga = 2;
    theta_init_dist = rand(d,N)/sqrt(d);
    w_init_dist = rand(d,N)/sqrt(d);
    %i.i.d sampling
    theta_saga_wt = zeros(d,M,n_runs_used);
    obj_saga_wt = zeros(M, n_runs_used);
end

%run algo
for i = 1:n_runs_used
    %generate data for this run
    indices = (i-1)*n_per_run + 1 : (i-1)*n_per_run + n_training;
    data_tr.phi = data.phi(:,indices);
    data_tr.phi_next = data.phi_next(:,indices);
    data_tr.rewards = data.rewards(indices);
    [A,b,C] = compute_AbC(data_tr); % To simplify the process of tuning the step size
    
    %% PD-Distributed IAG
    if ~isempty(strfind(algorithms,'DistIAG'))
        sigma_theta_distiag = 0.005/eigs(A,1);

        % the below dual step size is larger than that in the paper, but 
        % it converges and works better
        sigma_w_distiag = 0.005; 
        %i.i.d sampling
        [theta_distiag, obj_distiag ] = DistIAG( data_tr, theta_init_dist, ...
            w_init_dist, rho, M, sigma_theta_distiag, sigma_w_distiag, 'A_N10');
    end
    
end

plot(mean(obj_distiag, 2))