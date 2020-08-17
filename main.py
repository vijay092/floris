# this file runs an example of loss of communication in a wind farm while attempting to follow an AGC signal

# V1: no dynamics
# TODO: V2 - add turbine dynamics - mass spring damper
# TODO: V3 - add wake dynamics - time delay

# Questions:
# 1. how many turbines can go down before AGC is no longer achieveable?
# 2.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resilience import ResilientWind

# floris
import floris.tools as wfct

# verbose and show plots
plot = True
verbose = True

# load floris
fi = wfct.floris_interface.FlorisInterface("resilient_wind.json")

layout_x = [0,600,1200]
layout_y = [0,0,0]
nTurbs = len(layout_x)
fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))

# lose communication with these turbines
loss_comms = [0]
loss_comms2 = [0,1]

# generate a wind profile for one hour
Nt = 1 * 60 # seconds
t = np.linspace(0,4*(Nt-1),Nt)
ws = 8.0 * np.ones(Nt)
ws[20:40] = 6 * np.ones(40-20)

if plot:
    plt.figure()
    plt.plot(t,ws)
    plt.grid()
    plt.ylabel('Wind Speed (m/s)')
    plt.xlabel('Time (s)')

# compute the amount of power that FLORIS can produce without control
power_init = np.zeros(Nt)
power_agc = np.zeros(Nt)
noise = 0.1
for i in  range(Nt):
    # reinitialize floris
    fi.reinitialize_flow_field(wind_speed=ws[i])

    # Calculate wake
    fi.calculate_wake()
    # power calculation
    power_init[i] = fi.get_farm_power()

    # generate a signal that can be  achieved by turbines
    # TODO: not realistic because not including time delays from wakes/turbine
    per = np.random.randn(1)  # ask turbines to produce power between 0 and 100% of available power
    if i > 0:
        power_agc[i] = np.min([noise*per*power_init[i-1] + power_init[i-1], power_init[i]])
    else:
        power_agc[i] = power_init[i]

power_agc = 0.8*power_agc

if plot:
    plt.figure()
    plt.plot(t, power_init / (10 ** 6), label='Available')
    plt.plot(t, power_agc / (10 ** 6), label='AGC')
    plt.grid()
    plt.ylabel('Power (MW)')
    plt.xlabel('Time (s)')
    plt.legend()

# Resiliency of wind
lam = 1.0
res_wind = ResilientWind(fi,power_agc,lam)

# optimize floris for the AGC signal - follow the AGC signal, but try to 1) minimize difference between biggest and
# smallest power, 2) loads, 3) ???
power_default = np.zeros(Nt)
power_init_default = np.zeros(Nt)
power_loss_comms = np.zeros((round(nTurbs/2),Nt))
power_init_loss_comms = np.zeros((round(nTurbs/2),Nt))
power_loss_comms_est = np.zeros((round(nTurbs/2),Nt))
power_init_loss_comms_est = np.zeros((round(nTurbs/2),Nt))
power_shutdown = np.zeros((round(nTurbs/2),Nt))
power_init_shutdown = np.zeros((round(nTurbs/2),Nt))

for i in range(Nt):
# for i in range(1,10):
    print('Time ', i, 'out of ', Nt)
    res_wind.optimize(ws[i],power_agc[i],opt_type='default',loss_comms=[])
    power_default[i] = res_wind.power_opt
    power_init_default[i] = res_wind.power_initial

    # for j in range(0,round(nTurbs/2)):
    loss_comms = []
    for j in range(0, round(nTurbs / 2)):
    # for j in range(0,1):

        loss_comms.append(j)

        # loss of communications but you still know the power output of the whole farm
        res_wind.optimize(ws[i],power_agc[i],opt_type='loss_comms',loss_comms=loss_comms)
        power_loss_comms[j,i] = res_wind.power_opt
        power_init_loss_comms[j,i] = res_wind.power_initial

        # loss of communications - estimate the power output of the turbine - currently assumes the average of the turbines that are in communication
        res_wind.optimize(ws[i], power_agc[i], opt_type='loss_comms_est', loss_comms=loss_comms)
        power_loss_comms_est[j,i] = res_wind.power_opt
        power_init_loss_comms_est[j,i] = res_wind.power_initial

        # loss of communication and you shut down the problem turbine
        res_wind.optimize(ws[i],power_agc[i],opt_type='shutdown',loss_comms=loss_comms)
        power_shutdown[j,i] = res_wind.power_opt
        power_init_shutdown[j,i] = res_wind.power_initial

    # TODO: add noise to turbines that have lost comms - estimate will not be exact

plt.figure()
plt.plot(t,power_init / (10**6), label='Power available')
plt.plot(t,power_agc / (10**6), label='AGC signal')
plt.plot(t,power_default / (10**6), '--', label='All turbines')
loss_comms = []
for i in range(round(nTurbs/2)):
    loss_comms.append(i)
    strTurb = 'Lost comms = ' + str(loss_comms)
    strShut = 'Shutdown = ' + str(loss_comms)
    plt.plot(t,power_loss_comms[i,:] / (10**6), '--', label=strTurb)
    plt.plot(t,power_shutdown[i,:] / (10**6), '--', label=strShut)
plt.grid()
plt.legend()
plt.xlabel('Time (s)',fontsize=15)
plt.ylabel('Power (MW)')

plt.figure()
plt.plot(t,power_init / (10**6), label='Power available')
plt.plot(t,power_agc / (10**6), label='AGC signal')
plt.plot(t,power_default / (10**6), '--', label='All turbines')
loss_comms = []
for i in range(round(nTurbs/2)):
    loss_comms.append(i)
    strTurb = 'Lost comms = ' + str(loss_comms)
    plt.plot(t,power_loss_comms_est[i,:] / (10**6), '--', label=strTurb)
plt.grid()
plt.legend()
plt.title('Estimate Turbine Power from Lost Comms')
plt.xlabel('Time (s)',fontsize=15)
plt.ylabel('Power (MW)')

plt.show()







