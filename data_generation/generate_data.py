import gym
import marl_env

import floris
import floris.tools as wfct

import utilities as utils
import sarsa
import numpy as np

fi = wfct.floris_utilities.FlorisInterface("example_input.json")

D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]

# create FlorisUtilities object with the correct layout of turbines
fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])
fi.calculate_wake()
# create a list of agents that is assembled using the FlorisUtilities object
agents = utils.initialize_agent_list(fi, [layout_x, layout_y])

# lower and upper yaw limits
limits = [-30, 29]

# create an environment
env = gym.make('marl_env:marl-farm-v0', fi=fi, agents=agents, limits=limits)

lower_bounds = [-30, -30, -30]
upper_bounds = [30, 30, 30]
bins = (100, 100, 100)
bins_list = [bins, bins, bins]
offsets_list = [(-5, -5, -5), (0,0,0), (5, 5, 5)]

test_tiling = utils.create_tiling(lower_bounds, upper_bounds, bins_list, offsets_list)

test_state = (-25, -25, -25)

test_encoding = utils.encode_state(test_tiling, test_state)

q_table = sarsa.create_q_table(env)

state = env.reset()

action = sarsa.select_action(q_table[state], 'epsilon', epsilon=0.8)

outputs = env.step(action)
print("Initial state:", state)
print("Action:", action)
print("Next state:", outputs[0])

print(np.shape(sarsa.create_q_table(env)))