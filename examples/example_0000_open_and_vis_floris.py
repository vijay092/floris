# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

# This example code was modified to print out the wind farm layout that I will be using for the project.

import matplotlib.pyplot as plt
import floris.tools as wfct
import os
import numpy as np

# Initialize the FLORIS interface fi
fi = wfct.floris_utilities.FlorisInterface("example_input.json")

D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]

fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])

dir_path = os.path.dirname( os.getcwd() )

yaw_angles_path = dir_path + "\\data_generation\\yaw_angles_300_new.npy"
yaw_angles = np.load(yaw_angles_path) - 30

# Calculate wake
fi.calculate_wake(yaw_angles=yaw_angles[-1])

# Initialize the horizontal cut
hor_plane = wfct.cut_plane.HorPlane(
    fi.get_flow_data(),
    fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
plt.show()
