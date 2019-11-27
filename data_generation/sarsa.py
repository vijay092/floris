import numpy as np

def select_action(q_row, method, epsilon=0.5):
    """
    Method based on Lecture Exercise 22: Reinforcement Learning
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
    """
    dims = []

    for state_space in env.observation_space:
        dims.append(state_space.n)

    dims.append(env.action_space.n)

    return np.zeros(tuple(dims))

