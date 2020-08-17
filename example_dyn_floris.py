# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt
import floris
import floris.tools as wfct
import numpy as np

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("./example_input.json")

# Calculate wake
fi.calculate_wake()

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

powers = []
true_powers = []
total_time = 30

fi.reinitialize_flow_field(wind_speed=10)
fi.calculate_wake()
true_power = fi.get_farm_power()/1e6

fi.reinitialize_flow_field(wind_speed=8)
fi.calculate_wake()
vel_1 =[]
vel_2 =[]

tspan = np.linspace(0,total_time,100);
for sim_time in tspan:

    if sim_time == 0:
        fi.reinitialize_flow_field(wind_speed=10, sim_time=0)

    fi.calculate_wake(sim_time=sim_time)
    vel_1.append(fi.floris.farm.turbines[0].average_velocity)
    vel_2.append(fi.floris.farm.turbines[1].average_velocity)



# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
 
plt.figure()

plt.plot(tspan, vel_1, label="First Row")
plt.plot(tspan, vel_2, label="Second Row")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

plt.show()
