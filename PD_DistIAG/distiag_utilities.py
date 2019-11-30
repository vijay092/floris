import numpy as np
from numpy import matmul, transpose

class Data():
    def __init__(self, states, next_states, rewards, gamma):
        self.states = states
        self.next_states = next_states
        self.rewards = rewards
        self.gamma = gamma

def algo_preprocess(data):
    dims = np.shape(data.states)

    d = dims[0]
    n = dims[1]

    phi = data.states
    phi_next = data.next_states

    phi_gamma_phi_next = np.zeros((d,n))

    for i in range(n):
        phi_gamma_phi_next[:,i] = phi[:,i] - data.gamma*phi_next[:,i]

    reward_phi = np.zeros((d,1))
    
    for i in range(n):
        reward_phi = reward_phi + np.reshape(data.rewards[i]*phi[:,i], (d,1))
        

    return [phi, phi_gamma_phi_next, reward_phi]

def split_reward(data, N, agent_dict):
    # number of samples
    n = len(data.rewards)

    reward_N = np.zeros((n,N))

    # hard-coded for 3 turbine linear wind farm
    num_turbines = 3
    for i in range(n):
        for nn in range(N):
            reward_N[i,nn] = data.rewards[i] * (num_turbines-nn) * (1/6)

    #W_G = np.ones((num_turbines, num_turbines))
    W_G = np.identity(num_turbines)

    return [reward_N, W_G]

def run_pd_distiag(training_data, initial_theta, initial_w, rho, M, sigma_theta, sigma_w, agent_dict):
    data = training_data

    dims = np.shape(data.states)

    d = dims[0]
    n = dims[1]

    outputs = algo_preprocess(data)

    phi = outputs[0]
    phi_gamma_phi_next = outputs[1]
    reward_phi = outputs[2]

    A = matmul(phi, phi_gamma_phi_next.T) / n
    b = reward_phi/n
    C = matmul(phi, phi.T) / n
    C_inv = np.linalg.pinv(C)

    B = rho*np.identity(d) + matmul(A.T, matmul(C_inv, b) )

    theta_opt = matmul(np.linalg.pinv(B), matmul(A.T, matmul(C_inv,b) ) )
    obj_opt = matmul( matmul( (matmul(A,theta_opt) - b).T, C_inv), (matmul(A, theta_opt)-b) ) + rho*np.linalg.norm(theta_opt)**2
    
    N = len(agent_dict)

    [reward_N, W_G] = split_reward(data, N, agent_dict)
    
    theta_N = initial_theta
    w_N = initial_w

    # TODO: still don't know what M is, equivalent of T in [1]
    obj = np.zeros((M,N)) 

    theta_grad_old = np.zeros((d,N,n))
    w_grad_old = np.zeros((d,N,n))

    s_theta_avg = np.sum(theta_grad_old, axis=2) / n
    d_w_avg = np.sum(w_grad_old, axis=2) / n

    avg_obj = []

    for i in range(M):
        print(i)
        indices = np.random.permutation(n)
        for k in indices:
            theta_grad_cur = np.zeros((d,N))
            w_grad_cur = np.zeros((d,N))

            for nn in range(N):
                theta_grad_cur[:,nn] = rho*theta_N[:,nn] - matmul(np.outer(phi_gamma_phi_next[:,k], phi[:,k]), w_N[:,nn])
                w_grad_cur[:,nn] = matmul(np.outer(phi[:,k], -phi_gamma_phi_next[:,k]), theta_N[:,nn]) + reward_N[k,nn] - matmul(phi[:,k], w_N[:,nn])

            s_theta_avg = matmul(s_theta_avg, W_G) + theta_grad_cur/n - theta_grad_old[:,:,k]/n
            d_w_avg = d_w_avg + w_grad_cur/n - w_grad_old[:,:,k]/n

            theta_grad_old[:,:,k] = theta_grad_cur
            w_grad_old[:,:,k] = w_grad_cur

            theta_N = matmul(theta_N, W_G) - sigma_theta*s_theta_avg
            w_N = w_N + sigma_w*d_w_avg

        for nn in range(N):
            theta_N_col = np.reshape(theta_N[:,nn], (d,1))
            obj[i,nn] = matmul( matmul( (matmul(A, theta_N_col) - b).T, C_inv), (matmul(A, theta_N_col) - b) ) + rho*np.linalg.norm(theta_N_col)**2 - obj_opt
        
        avg_obj.append(np.mean(obj[i,:]))

    return [np.array(avg_obj)]