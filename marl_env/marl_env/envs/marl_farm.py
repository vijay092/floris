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
    def __init__(self, fi, agents, limits):
        self.fi = fi
        self.agents = agents
        self.yaw_angles = [0 for agent in agents]

        self.yaw_lower_limit = limits[0]
        self.yaw_upper_limit = limits[1]

        self.farm_power = self._calculate_farm_power()

        self.turbine_rated_power = 5

        self.N = len(agents)

        # there are three actions: decrease 1 deg, stay, and increase 1 deg for each turbine, so 3^N
        # possible action choices
        self.action_space = gym.spaces.Discrete(3**self.N)
        
        # technically this should range from -30 to 30, but only positive numbers are allowed
        turbine_obs_space = gym.spaces.Discrete((limits[1]-limits[0]+1))

        self.observation_space = gym.spaces.Tuple((turbine_obs_space, turbine_obs_space, turbine_obs_space))

    def step(self, action):
        """
        action: list of turbine yaw angle deltas (one for each turbine in the wind farm)
        """
        deltas = [None for agent in self.agents]
        x = action
        for i,_ in enumerate(self.agents):
            deltas[i] = (x % 3**(self.N - (i+1)) ) - 1
            x = x % 3**(self.N - (i+1))

        # updated yaw angles: delta will be -1, 0, or 1
        new_yaw_angles = [yaw_angle + delta for (yaw_angle, delta) in zip(self.yaw_angles, deltas)]

        # make sure yaw angles do not exceed limits
        new_yaw_angles = [max(self.yaw_lower_limit, yaw_angle) for yaw_angle in new_yaw_angles]
        new_yaw_angles = [min(self.yaw_upper_limit, yaw_angle) for yaw_angle in new_yaw_angles]

        # calculate the wakes due to the new yaw angles (this consitutes the actual "step")
        self.fi.calculate_wake(yaw_angles=new_yaw_angles)

        # updated value function outputs
        new_farm_power = self._calculate_farm_power()

        # update the wind farm yaw angles
        self.yaw_angles = new_yaw_angles
        
        # calculate reward based on the change in the value function
        reward = new_farm_power - self.farm_power

        # threshold is 1% of the rated power of the entire wind farm 
        threshold = self.N * self.turbine_rated_power * 0.01

        if new_farm_power - self.farm_power < threshold:
            done = True
        else:
            done = False

        # reset the internally tracked farm power level
        self.farm_power = new_farm_power

        # the state of the wind farm is obtained by considering the turbines in aggregate
        return [self.yaw_angles, reward, done, None]

    def _calculate_farm_power(self):
        return sum([agent.turbine.power for agent in self.agents])

    def _calculate_farm_values(self):
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