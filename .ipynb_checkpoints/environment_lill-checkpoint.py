# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:32:30 2019

@author: sanja
"""
import numpy as np
# Environment for the wind farm system. The goal is to maximize power
# Action: The control inputs are pitch angle and Generator Power
# Reward: The power itself
# x_next: Dictated by the ODE. (have to discretize and use it)
from numpy import linspace, zeros, exp
import matplotlib.pyplot as plt
import floris
import floris.tools as wfct
import random
from scipy.stats import norm,multivariate_normal




def reset(fi,no_turb,wnd_init):
    #rndWnd = random.randint(5,10)
    fi.reinitialize_flow_field(wind_speed=[wnd_init],sim_time = 0)
    Wnd = np.zeros(no_turb,)
    fi.calculate_wake(yaw_angles=np.zeros(48,))
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    
    return Wnd, fi



def step(a,x,fi,sim_time,pwr_4s, wnd_4s):
    
    '''
    This function contains the dyanmics of the wake
    
    a :: index of the action to be chosen
    x :: state (wind speed)
    fi :: floris interface object
    sim_time :: time of simulation
    Actions :: Action space
    
    '''
    # Given the index, select the action:
    no_turb = 48; idx = 10;
    a1 = 0*np.ones(no_turb)
    a1[idx :idx +9] = a; 

    fi.reinitialize_flow_field(wind_speed=[wnd_4s],sim_time = sim_time)
    fi.calculate_wake(yaw_angles=a1)
    pow_array = fi.get_turbine_power()
    power = np.sum(pow_array)
    # State 

    Wnd = np.zeros(no_turb,)
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    x_next = Wnd[idx:idx+9];
    

    
    # Rewards for the system
    rewards = 0;
    low_pow = 0.95*1e6*pwr_4s
    up_pow = 1.05*1e6*pwr_4s
    
    # If power generated is greater than the 
    if power > up_pow:
        pow_asc = np.sort(pow_array);
        pow_asc_idx = np.argsort(pow_asc)
        
        cum_pow = np.cumsum(pow_asc)
        idx = np.abs(pwr_4s*1e6 - cum_pow).argmin()
        power = cum_pow[idx]
        rewards = 1;
        
    elif power > low_pow and power < up_pow:
        rewards = 1;

    
    done = bool(0)
    return x_next, rewards, done, fi, power



