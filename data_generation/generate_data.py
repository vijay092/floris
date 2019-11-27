import gym
import marl_env

import floris
import floris.tools as wfct
import utilities as utils
import sarsa

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

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

iterations = 100#85000

# create an environment
env = gym.make('marl_env:marl-farm-v0', fi=fi, agents=agents, limits=limits, iterations=iterations)

lower_bounds = [-30, -30, -30]
upper_bounds = [30, 30, 30]
bins = (50, 50, 50)
bins_list = [bins, bins, bins]
offsets_list = [(-5, -5, -5), (0,0,0), (5, 5, 5)]

tiling = utils.create_tiling(lower_bounds, upper_bounds, bins_list, offsets_list)

params = {'method':'epsilon', 
          'epsilon':0.3,
          'alpha':0.1,
          'gamma':0.5}

outputs = sarsa.generate_trajectory(env, params, tiling)

plt.plot(outputs[-1])

plt.figure()
for i in range(len(agents)):
    plt.plot( [state[i] for state in outputs[-2]] )

dir_path = os.path.dirname(os.getcwd())

np.save('states', outputs[0])
np.save('rewards', outputs[1])
np.save('next_states', outputs[2])

agent_dict = {agent.alias: agent.coordinates for agent in agents}

with open('agents.pickle', 'wb') as handle:
    pickle.dump(agent_dict, handle)

np.save('agents', agent_dict, allow_pickle=True)

plt.show()