import floris
import floris.tools as wfct

import numpy as np

# Agent object that encapsulates necessary turbine information
class Agent():
    def __init__(self, turbine, alias, coordinates):
        self.turbine = turbine
        self.alias = alias
        self.coordinates = coordinates
        return

# creates a list of agents that will be used by the environment to calculate reward
def initialize_agent_list(fi, layout_array):
    agents = []
    layout_x = layout_array[0]
    layout_y = layout_array[1]

    aliases = ["turbine_" + str(i) for i in range(len(fi.floris.farm.turbines))]
    farm_turbines = {alias: (x,y) for alias,x,y in zip(aliases,layout_x,layout_y)}
    for i,turbine in enumerate(fi.floris.farm.turbines):
        agent = Agent(turbine, aliases[i], farm_turbines[aliases[i]])
        agents.append(agent)

    return agents

def create_grid(lower_bounds, upper_bounds, bins, offsets):
    """
    Create a grid to be used with a tiling algorithm.

    Returns a numpy array of numpy arrays, each one representing a tiling along each dimension
    """
    if len(lower_bounds) != len(upper_bounds) or len(bins) != len(offsets) or len(bins) != len(lower_bounds):
        raise ValueError('Grid dimensions do not agree')

    grid = []

    for lower_bound,upper_bound,bin_num,offset in zip(lower_bounds, upper_bounds, bins, offsets):
        row = np.linspace(lower_bound, upper_bound, num=bin_num-1) - offset
        grid.append(row)

    return np.array(grid)