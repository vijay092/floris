function [ phi,phi_gamma_phi_next, reward_phi ] = algo_preprocess( data )
%This function outputs some common vectors, matrices needed for all
%algorithm based on training data

% This function is written by Simon Du (CMU)
if isfield(data, 'actions')
    n = length(data.actions);
    [d,~] = size(data.features);
else
    [d,n] = size(data.phi);
end
phi = zeros(d,n);
phi_next = zeros(d,n);
%pre compute phi
if isfield(data,'phi')
    phi = data.phi;
else 
    for t = 1:n
        phi(:,t) = data.features(:,data.states(t));
    end
end
%pre compute phi_next
if isfield(data,'phi_next')
    phi_next = data.phi_next;
else 
    for t = 1:n
        phi_next(:,t) = data.features(:,data.states_next(t));
    end
end
%pre compute phi_gamma_phi_next
phi_gamma_phi_next = zeros(d,n);
for t = 1:n 
    phi_gamma_phi_next(:,t) = phi(:,t)- data.gamma*phi_next(:,t);
end
%precompute reward phi
reward_phi = zeros(d,1);
for t = 1:n
   reward_phi = reward_phi + data.rewards(t)*phi(:,t); 
end

end

