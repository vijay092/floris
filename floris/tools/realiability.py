

def get_surface(wind_speed, TI, T1, T2, component):
    """
    This function gets the respective surrogate surface for the input 
    parameters.
    
    Args:
        wind_speed ([type]): [description]
        TI ([type]): [description]
        T1 ([type]): [description]
        T2 ([type]): [description]
        component ([type]): [description]
    """

    # pickle.load(the_right_file)

# this would live in turbine.py
def calculate_turbine_reliablity(wind_speed, TI, T1, T2, component):
    surf = get_surface(wind_speed, TI, T1, T2, component)

    reliability = calc_reliability(x, y, surf, component)

    self.component_reliability = reliability

# this would run after calculate_wake, using the flow field 

from itertools import combinations

def calculate_farm_reliability(wind_speed, TI, component):

    Turbine_L10s = np.zeros(self.nturbs)
    Turbine_PowerVals = np.zeros(self.nturbs)

    # turbs = []
    # theta = math.radians(wind_direction) #rotate grid 90 deg
    # for ii in list(zip(layout_x, layout_y)):
    #     turbs.append((self.rotate_array(ii, theta)))
    
    # layout_x = [ii[0] for ii in turbs]
    # layout_y = [ii[1] for ii in turbs]

    rotated_map = self.turbine_map.rotated(
            self.wind_direction, center_of_rotation)

    # sort the turbine map
    sorted_map = rotated_map.sorted_in_x_as_list()

    layout_x = [coord.x1 for coord, turbine in sorted_map]
    layout_y = [coord.x2 for coord, turbine in sorted_map]
    
    turb_label = range(0, len(layout_x))    

    # 2) Assign L10 values to all turbines
    
    L10_matrix = np.full( (len(turb_label), len(turb_label)), float(L10_T1_surface[0]) )
    # Power_matrix = np.full( (len(turb_label), len(turb_label)), float(Power_T1_surface[0]) )

    turbs_to_analyse = []

    x_threshold = 1260
    y_threshold = 252
    
    #Delete pairs that fall outside space constraints
    for ii in list(combinations(turb_label, 2)):
        turb1 = ii[0] 
        turb2 = ii[1]
        if abs(layout_x[turb2] - layout_x[turb1]) <= x_threshold and abs(layout_y[turb2] - layout_y[turb1]) <= y_threshold:
            turbs_to_analyse.append(ii)
            
    # Assign L10 and power values; turbs are ordered in proximity to the incoming wind direction
    for ii in turbs_to_analyse:
        turb1 = ii[0]
        turb2 = ii[1]
        L10_matrix[turb1][turb2] = L10_T2_surface(layout_x[turb2],layout_y[turb2])
        # Power_matrix[turb1][turb2] = Power_T2_surface(layout_x[turb2],layout_y[turb2]) #kW/hr

    #Find the minimum of each row (this is the minimum )
    for column in L10_matrix.T:
        Turbine_L10s = [min(column) for column in L10_matrix.T]
        Turbine_PowerVals = [Power_matrix.T[ii][np.argmin(column)] for ii, column in enumerate(L10_matrix.T)]