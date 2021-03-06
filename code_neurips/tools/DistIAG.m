function [ theta_N, obj ] = DistIAG( data, theta, w, rho, T, sigma_theta, sigma_w, graph_dat )
%This algorithm implement a distributed IAG algorithm for minimizing EM-MSPBE. At every
%iteration we use 1 sample (minibatch = 1).
%Input:
%data: a struct consist of
%   phi: n x 1 vector  
%   phi_next: n x 1 vector
%   rewards: n x 1 vector
%   rhos: n x 1 vector, ratio of stationary probabilities 
%       between target polict and data collection policy.
%   features: d by m matrix, features(:,i): feature vector of i-th state
%   gamma: discount factor
%theta: d x 1 vector, initialization of theta
%w: d x 1 vector, initialization of w
%T: scalar, number of outler loop
%sigma_theta: step size for theta (primal)
%sigma_w: step size for w (dual)
%Output:
%theta: d x 1 vector, linear approximation coefficient

% written by H-T. Wai 
% We use $n$ as the number of samples (instead of $M$ in the paper)
[d,n] = size(data.phi);
% thetas = zeros(d,M);

[ phi, phi_gamma_phi_next, reward_phi ] = algo_preprocess( data );

A = phi*phi_gamma_phi_next'/n;
b = reward_phi/n;
C = phi*phi'/n;
C_inv = pinv(C);

B  = rho*eye(d)+A'*pinv(C)*A;

theta_opt = pinv(B)*(A'*pinv(C)*b);
obj_opt = (A*theta_opt - b)'*pinv(C)*(A*theta_opt-b) + rho*norm(theta_opt)^2;

%eval(['load ' graph_dat]); 
W_G = [ 2/3 1/3 0 ; 1/3 1/3 1/3 ; 0 1/3 2/3];
N = size(W_G,1); 

% "split" the reward into the individual reward
reward_N = gen_local_reward( data , N , 'std' );

theta_N = theta; w_N = w;


obj = zeros(T,N);

% allocating memory for the gradients
grd_theta_old = zeros(d,N,n);
grd_w_old = zeros(d,N,n);

% variables for gradient tracking
s_theta_avg = sum(grd_theta_old,3)/n;
d_w_avg = sum(grd_w_old,3)/n;

% begin DistIAG
for i = 1 : T
    % we're in the $i$th epoch
    % Using a random shuffling approach -- i.e., sampling w/o
    % replacement
    indices = randperm( n );
    for k = 1 : n % we use $k$ for the sample index (instead of $p$)
        index = indices(k);
        % compute the gradient (note w is actually the negated gradient)
        grd_theta_cur = zeros(d,N); grd_w_cur = zeros(d,N);

        for nn = 1 : N
            grd_theta_cur(:,nn) = rho*theta_N(:,nn) - ...
                phi_gamma_phi_next(:,index)*(phi(:,index)'*w_N(:,nn));
            grd_w_cur(:,nn) = phi(:,index)*(-phi_gamma_phi_next(:,index)'*theta_N(:,nn) ...
                + reward_N(index,nn) - (phi(:,index)'*w_N(:,nn))); 
        end
        % Update gradient surrogate (Eq. (13) and (14))
        s_theta_avg = s_theta_avg*W_G + grd_theta_cur/n - grd_theta_old(:,:,index)/n;
        
        d_w_avg = d_w_avg + grd_w_cur/n - grd_w_old(:,:,index)/n;

        % save the current gradients into the memory
        grd_theta_old(:,:,index) = grd_theta_cur; grd_w_old(:,:,index) = grd_w_cur;
        
        % Update theta and w (Eq. (15))
        theta_N = theta_N*W_G - sigma_theta * s_theta_avg;
        
        w_N = w_N + sigma_w * d_w_avg;
    end
    
    % Evaluate the optimality gap and print it out
    for nn = 1 : N
        obj(i,nn) = (A*theta_N(:,nn) - b)'*C_inv*(A*theta_N(:,nn)-b) + rho*norm(theta_N(:,nn))^2 - obj_opt;
        if isnan(obj(i,nn))
            obj(i,nn) = (A*theta_N(:,nn) - b)'*(A*theta_N(:,nn)-b) + rho*norm(theta_N(:,nn))^2;
        end
    end
    if mod(i,10) == 0
        disp('DistIAG iid outer loop:'); disp(i);
        disp('(worst) obj:'); disp(max(obj(i,:)));
        disp('(avg) obj:'); disp(mean(obj(i,:)));
    end
end
