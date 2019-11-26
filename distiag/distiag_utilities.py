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
        row = np.linspace(lower_bound, upper_bound, num=bin_num) + offset
        grid.append(row)

    return np.array(grid)

def create_tiling(lower_bounds, upper_bounds, bins_list, offsets_list):
    """
    Create an array of grids that can be used for tile coding. bins_list is an array of iterables that
    specify the bin numbers for each of the dimensions of each grid. offsets_list is an array of 
    iterables that specify the offsets for each of the dimensions of each grid.

    Returns a numpy array of grids
    """
    if len(lower_bounds) != len(upper_bounds) or len(bins_list) != len(offsets_list) or len(bins_list) != len(lower_bounds):
        raise ValueError('Tiling dimensions do not agree')

    tiling = []

    for bins,offsets in zip(bins_list, offsets_list):
        grid = create_grid(lower_bounds, upper_bounds, bins, offsets)
        tiling.append(grid)

    return np.array(tiling)

def vectorize_digital(bin, dim):
    """
    Write a digital bin number as a boolean array

    bin: bin number
    dim: length of vectorized output
    """
    vector_output = []
    for i in range(dim):
        if i == bin:
            vector_output.append(1)
        else:
            vector_output.append(0)
    return vector_output

def encode_state(tiling, state):
    """
    Takes a multi-dimensional state observation and enocdes it using a given tiling.
    """
    if len(tiling[0]) != len(state):
        raise ValueError("State observation dimension does not match tiling dimension.")

    tiling_encoding = []

    for grid in tiling:
        grid_encoding = []
        for i,axis in enumerate(grid):
            coding = np.digitize(state[i], axis)
            grid_encoding.append(vectorize_digital(coding, len(axis)))
        tiling_encoding.append(grid_encoding)

    return np.array(tiling_encoding)