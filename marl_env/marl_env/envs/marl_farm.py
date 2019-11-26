import gym
from gym import error, spaces, utils
from gym.utils import seeding

import floris
import floris.tools as wfct

import matplotlib.pyplot as plt 
import numpy as np
import math

class FarmMARL(gym.Env):
    """
    This is a custom OpenAI gym environment that is designed to create a multiagent environment in which agents
    representing wind turbines can interact.
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, fi, agents):
        self.fi = fi
        self.agents = agents
        self.yaw_angles = [0 for agent in agents]
        self.values = [agent.turbine.power for agent in agents]

        # there are three actions: decrease 1 deg, stay, and increase 1 deg
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Discrete(50)

    def step(self, action):
        """
        action: list of turbine yaw angle deltas (one for each turbine in the wind farm)
        """
        # updated yaw angles: action_value will be 0, 1, or 2
        new_yaw_angles = [yaw_angle + action_value - 1 for (yaw_angle, action_value) in zip(self.yaw_angles, action)]
        # updated value function outputs
        new_values = self._calculate_values()

        # calculate the wakes due to the new yaw angles
        self.fi.calculate_wake(yaw_angles=new_yaw_angles)

        # update the wind farm yaw angles
        self.yaw_angles = new_yaw_angles
        
        # calculate reward based on the change in the value function
        rewards = [new_value - value for (new_value, value) in zip(new_values, self.values)]

        # reset the internally tracked value function outputs
        self.values = new_values

        done = False

        return [self.yaw_angles, rewards, done, None]

    def _calculate_values(self):
        """
        Returns a list of values corresponding to each turbine in the wind farm.
        """
        values = []
        for agent in self.agents:
            downwind_agent = self._find_downwind_neighbor(agent)
            if downwind_agent is None:
                # this condition means there are no downwind turbines and the reward is 
                values.append(agent.turbine.power)
            else:
                # add the power values from the current turbine and the downwind turbine
                values.append(agent.turbine.power + downwind_agent.turbine.power)
        return values

    def _find_downwind_neighbor(self, agent):
        """
        Determine the most immediate downwind turbine to calculate the reward function
        """
        downwind_agents = []
        for neighbor_agent in self.agents:
            if neighbor_agent.coordinates[0] - agent.coordinates[0] > 0:
                # if difference in x coordinates is positive, neighbor_agent is downwind since wind is at 270 deg
                downwind_agents.append(neighbor_agent)

        # return None if there are no downwind agents
        if len(downwind_agents) == 0:
            return None

        # difference in both the x and y direction
        x_diffs = [downwind_agent.coordinates[0] - agent.coordinates[0] for downwind_agent in downwind_agents]
        y_diffs = [downwind_agent.coordinates[1] - agent.coordinates[1] for downwind_agent in downwind_agents]

        # get the argument of the turbine with the smallest Euclidean distance downwind from the current turbine
        temp_index = np.argmin([math.sqrt(x**2 + y**2) for (x,y) in zip(x_diffs, y_diffs)])

        return downwind_agents[temp_index]

    def reset(self):
        self.yaw_angles = [0 for agent in self.agents]
        self.fi.calculate_wake(self.yaw_angles)
        self.values = [agent.turbine.power for agent in self.agents]
        return

    def render(self, mode='human', close=False):
        return