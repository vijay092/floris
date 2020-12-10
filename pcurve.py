# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:19:53 2020

@author: sanja
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


df_wind = pd.read_csv("swift_201908_processed_mean.csv")
    
    
power = [0.414,0.426,0.426,0.427,0.427,0.438,0.439,0.427,0.377,0.297,0.238,0.193,0.159,0.133,0.112,0.095,0.081,0.07,
         0.061,0.054,0.047,0.042,0.037,0.033]
speed = [4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,
         26.0,27.0]
f = interpolate.interp1d(speed, power, fill_value='extrapolate')
date = np.array(df_wind.iloc[0::, 0])
test_speed = np.array(df_wind.iloc[0::, 3])
np.array(df_wind.iloc[0::, 3])
# turbine parameters
D = 93
rho = 1.225
Area = np.pi * (D/2)**2
cut_in = 3.0
cut_out = 25.0
rated_power = 2300000 # each machine can produce 2.3MW
# record power
out_power = np.zeros(np.shape(test_speed))
for i in range(len(test_speed)):
    Cp = f(test_speed[i])
    if test_speed[i] < 4.0:
        out_power[i] = 0.0
    elif test_speed[i] > cut_out:
        out_power[i] = 0.0
    else:
        out_power[i] = 0.5 * rho * Area * Cp * test_speed[i]**3
    # rated power
    if out_power[i] > rated_power:
        out_power[i] = rated_power
    if 10 < test_speed[i] > 4:
        out_power[i] = 0.85* out_power[i]



plt.plot(out_power/(10**7)*48)
plt.plot(test_speed)
plt.plot()
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power (MW)')
plt.grid()
plt.show()


import csv
with open('pcurve.csv', 'w',newline='') as csvfile:
    fieldnames = ['Date/time (CDT)','POWER (MW)','WIND SP (m/s)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(date)):
        writer.writerow({'Date/time (CDT)':date[i],\
                         'POWER (MW)':out_power[i]*48*1e-6,\
                         'WIND SP (m/s)':test_speed[i]})





