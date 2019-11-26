import floris
import floris.tools as wfct

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
    """
    return