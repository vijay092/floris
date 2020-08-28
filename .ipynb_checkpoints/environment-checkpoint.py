# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:32:30 2019

@author: sanja
"""
import numpy as np
# Environment for the wind turbine system. The goal is to maximize power
# Action: The control inputs are pitch angle and Generator Power
# Reward: The power itself
# x_next: Dictated by the ODE. (have to discretize and use it)
from numpy import linspace, zeros, exp
import matplotlib.pyplot as plt
import floris
import floris.tools as wfct
import random



def reset(fi,no_turb):
    rndWnd = random.randint(3, 11)
    fi.reinitialize_flow_field(wind_speed=[rndWnd,rndWnd])
    
    Wnd = np.zeros(no_turb,)
    fi.calculate_wake(yaw_angles=[0,0])
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    
    return Wnd, fi

def step(a,x,fi,sim_time):
    
    '''
    This function contains the dyanmics of the wake
    
    a :: index of the action to be chosen
    x :: state (wind speed)
    fi :: floris interface object
    sim_time :: time of simulation
    Actions :: Action space
    
    '''
    # Given the index, select the action:
    #fi.reinitialize_flow_field(wind_speed=random.randint(3, 11))
    fi.calculate_wake(yaw_angles=a)
    power = np.sum(fi.get_turbine_power())
    no_turb = 2; 
    Wnd = np.zeros(no_turb,)
    
    
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    x_next = Wnd;
    
    # Rewards for the system
    power_ref = 5e6;
    #print(power)
    if np.sum(power) > 7e6:
        rewards = 1;
    else:
        rewards = -1;
        
    done = bool(0)
    return x_next, rewards, done, fi 

