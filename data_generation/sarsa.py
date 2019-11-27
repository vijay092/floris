import numpy as np
import utilities as utils

def select_action(q_row, method, epsilon=0.5):
    """
    Method taken from Lecture Exercise 22: Reinforcement Learning
    """
    if method not in ["random", "epsilon"]:
        raise NameError("Undefined method.")
    
    action_arr = list(range(len(q_row)))

    if method=="random":
      #return np.random.randint(low=0,high=len(q_row))
      return np.random.choice(np.array(action_arr))
    elif method=="epsilon":
      rand_num = np.random.random()
      if rand_num < epsilon:
        #return np.random.randint(low=0,high=len(q_row))
        return np.random.choice(np.array(action_arr))
      else:
        return np.argmax(q_row)

def create_q_table(env):
    """
    Create a Q-table with the correct dimensions
    Method based on Lecture Exercise 22: Reinforcement Learning
    """
    dims = []

    for state_space in env.observation_space:
        dims.append(state_space.n)

    dims.append(env.action_space.n)

    return np.zeros(tuple(dims))

def sarsa_update(q_table, state_indices, action, reward, next_state_indices, alpha, gamma, epsilon):
    """
    Method based on Lecture Exercise 22: Reinforcement Learning
    """
    Q_s_a = q_table[state_indices, action]

    # Sarsa is on-policy, and does not use greedy value maximization to select the next action
    next_action = select_action(q_table[next_state_indices], 'epsilon', epsilon)

    Q_s_1_a = q_table[next_state_indices, next_action]

    return (1-alpha)*Q_s_a + alpha*(reward + gamma * Q_s_1_a)

def generate_trajectory(env, params, tiling):
    """
    Create a trajectory of state, next_state, and reward in np arrays
    """

    q_table = create_q_table(env)

    state = env.reset()

    done = False
    
    method = params['method']
    epsilon = params['epsilon']
    alpha = params['alpha']
    gamma = params['gamma']

    # output array for diagnostic purposes
    power = []
    states = []
    rewards = []
    next_states = []

    counter = 0
    while not done:
        action = action = select_action(q_table[state], method=method, epsilon=epsilon)

        [next_state, reward, done, misc] = env.step(action)

        q_table[state, action] = sarsa_update(q_table, state, action, reward, next_state, alpha, gamma, epsilon)

        power.append( sum([turbine.power for turbine in env.fi.floris.farm.turbines]) )

        states.append( utils.encode_state(tiling, state) )
        rewards.append( reward )
        next_states.append( utils.encode_state(tiling, next_state) )

        state = next_state

        if (counter + 1) % 100 == 0:
            print("Simulation iteration:", counter)

    return [states, rewards, next_states, power]
