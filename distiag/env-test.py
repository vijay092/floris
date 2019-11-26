import gym
import marl_env

import floris
import floris.tools as wfct

import distiag_utilities as du

fi = wfct.floris_utilities.FlorisInterface("example_input.json")

D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]

# create FlorisUtilities object with the correct layout of turbines
fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])
fi.calculate_wake()
# create a list of agents that is assembled using the FlorisUtilities object
agents = du.initialize_agent_list(fi, [layout_x, layout_y])

# create an environment
env = gym.make('marl_env:marl-farm-v0', fi=fi, agents=agents)

