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

#layout_x = [0, 7*D, 14*D, 21*D, 28*D, 0, 7*D, 14*D, 21*D, 28*D]
#layout_y = [0, 0, 0, 0, 0, 5*D, 5*D, 5*D, 5*D, 5*D]

# create FlorisUtilities object with the correct layout of turbines
fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])
fi.calculate_wake()
# create a list of agents that is assembled using the FlorisUtilities object
agents = utils.initialize_agent_list(fi, [layout_x, layout_y])

# lower and upper yaw limits
limits = [-30, 29]

iterations = 40000

# create an environment
env = gym.make('marl_env:marl-farm-v0', fi=fi, agents=agents, limits=limits, iterations=iterations)

lower_bounds = [-30, -30, -30]
upper_bounds = [30, 30, 30]
bins = (33, 33, 33)
bins_list = [bins, bins, bins]
offsets_list = [(-0.5, -0.5, -0.5), (0,0,0), (0.5, 0.5, 0.5)]

tiling = utils.create_tiling(lower_bounds, upper_bounds, bins_list, offsets_list)

params = {'method':'epsilon', 
          'epsilon':0.3,
          'alpha':0.1,
          'gamma':0.5}

print("Training wind farm...")
[q_table, power, unencoded_states] = sarsa.train_farm(env, params)

np.save("q_table_300_new", q_table)
#q_table = np.load("q_table.npy")

# power and yaw angle training plots
plt.plot(power)
plt.xlabel("Simulation Iteration")
plt.ylabel("Power (W)")
plt.title("Power vs. Simulation Iteration for Three Turbine Wind Farm")
plt.show()

plt.figure()

for i in range(len(agents)):
    plt.plot( [state[i] for state in unencoded_states] )
plt.xlabel("Simulation Iteration")
plt.ylabel("Yaw Angle (deg)")
plt.title("Yaw Angle vs Simulation Iteration")

np.save('power_300', power)
np.save('yaw_angles_300', unencoded_states)


print("Generating trajectory...")
outputs = sarsa.generate_trajectory(env, params, tiling, q_table)

# save numpy arrays and agent dict pickle
dir_path = os.path.dirname(os.getcwd())

np.save('states_300', outputs[0])
np.save('rewards_300', outputs[1])
np.save('next_states_300', outputs[2])

agent_dict = {agent.alias: agent.coordinates for agent in agents}

with open('agents.pickle', 'wb') as handle:
    pickle.dump(agent_dict, handle)

# save data as text files
np.savetxt("state_wf_300.csv", outputs[0], delimiter=',')
np.savetxt("nextstate_wf_300.csv", outputs[2], delimiter=',')
np.savetxt("reward_wf_300.csv", outputs[1], delimiter=',')

plt.show()