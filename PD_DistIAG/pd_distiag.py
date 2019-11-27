import numpy as np
import os
import pickle

# path to parent directory
dir_path = os.path.dirname(os.getcwd())

# path to data_generation foler
data_path = dir_path + "\\data_generation"

# load the three arrays and dictionary that are necessary for the PD_DistIAG algorithm
states_path = data_path + "\\states.npy"
states = np.load(states_path)

next_states_path = data_path + "\\next_states.npy"
next_states = np.load(next_states_path)

rewards_path = data_path + "\\rewards.npy"
rewards = np.load(rewards_path)

agents_path = data_path + "\\agents.pickle"
with open(agents_path, 'rb') as handle:
    agent_dict = pickle.load(handle)

# begin algorithm
