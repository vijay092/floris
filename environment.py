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
def reset(nTurb):
    x = np.random.uniform(0.2,1,nTurb)
    return x

def step(a,x,fi):
    
    # simulate to get next step using euler
    #dt = 1e-2
    #x_next = x + dt*ODETurb.Turbine(0,x,a)
    fi.calculate_wake(yaw_angles=[a,a])
    power = fi.get_turbine_power()
    x_next = power;
    # Rewards for the system
    rewards = np.sum(power);
    done =  x[0] > 1e10
    done = bool(done)
    
    return x_next, rewards, done 

