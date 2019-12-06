function [ A,b,C ] = compute_AbC( data )
% This function is written by Simon Du (CMU)

if isfield(data, 'actions')
    n = length(data.actions);
    [d,~] = size(data.features);
else
    [n,d] = size(data.phi);
end 
[ phi, phi_gamma_phi_next, reward_phi ] = algo_preprocess( data );


A = phi*phi_gamma_phi_next'/n;
b = reward_phi/n;
C = phi*phi'/n;


end

