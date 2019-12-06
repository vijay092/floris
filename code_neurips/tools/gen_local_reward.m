function b_i = gen_local_reward( data, N , mode )
% split the reward into the local ones - using the same data struct as
% before
% written by H-T. Wai 

n = length( data.rewards );
b_i = zeros( n, N );
num_turbines = 3;
%for nn = 1 : n 
for nn = 1 : N
%     % for each time index, let's assume that the reward is distributed
%     % according a p.m.f.
%     if strcmp(mode, 'std')
%         simpl_vec = rand(N,1); simpl_vec = simpl_vec / sum( simpl_vec );
%         b_i(nn,:) = N*data.rewards(nn)*simpl_vec;
%     else % more extreme
%         simpl_vec = 10*randn(N,1); mean_simpl_vec = mean(simpl_vec);
%         simpl_vec = simpl_vec - mean_simpl_vec + data.rewards(nn);
%         b_i(nn,:) = simpl_vec;
%     end
    for i = 1: n
        b_i(i,nn) = data.rewards(i) * (num_turbines-nn) * (1/6);
    end
end