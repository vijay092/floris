import gym
from gym import error, spaces, utils
from gym.utils import seeding

import floris
import floris.tools as wfct

import matplotlib.pyplot as plt 
import numpy as np

class FarmMARL(gym.Env):
    """
    This is a custom OpenAI gym environment that is designed to create a multiagent environment in which agents
    representing wind turbines can interact.
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, fi, agents):
        self.fi = fi
        self.agents = agents
    def step(self, action):
        return
    def reset(self):
        return
    def render(self, mode='human', close=False):
        return