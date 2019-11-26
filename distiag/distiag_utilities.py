import floris
import floris.tools as wfct

import numpy as np

class Agent():
    """
    Agent object that encapsulates necessary turbine information 
    """
    def __init__(self, turbine, alias, coordinates):
        self.turbine = turbine
        self.alias = alias
        self.coordinates = coordinates
        return

def initialize_agent_list(fi, layout_array):
    """
    Creates a list of agents that will be used by the environment to calculate reward
    """
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

def create_tiling(lower_bounds, upper_bounds, bins_lists, offsets_lists):
    """
    Create an array of grids that can be used for tile coding. bins_lists is an array of iterables that
    specify the bin numbers for each of the dimensions of each grid. offsets_lists is an array of 
    iterables that specify the offsets for each of the dimensions of each grid.

    Returns a numpy array of grids
    """
    if len(lower_bounds) != len(upper_bounds) or len(bins_lists) != len(offsets_lists) or len(bins_lists) != len(lower_bounds):
        raise ValueError('Tiling dimensions do not agree')

    tiling = []

    for lower_bound,upper_bound,bins,offsets in zip(lower_bounds, upper_bounds, bins_lists, offsets_lists):
        grid = create_grid(lower_bound, upper_bound, bins, offsets)
        tiling.append(grid)

    return np.array(tiling)

    def encode_state(tiling, state):
        """
        Takes a multi-dimensional state observation and enocdes it using a given tiling.
        """
        return