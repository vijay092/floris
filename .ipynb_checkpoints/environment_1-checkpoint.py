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
    fi.calculate_wake(yaw_angles=[0,0,0,0,0,0,0,0,0])
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    
    return Wnd, fi

def pow_dev(fi,mean_ws,std_ws):
    
    n_sim = 100; 
    ws_base = np.random.uniform(low=5, high=15., size=n_sim)
    prob_obj = multivariate_normal(mean_ws, std_ws)
    farm_power = np.zeros(n_sim,)
    prob = np.zeros(n_sim,)
    
    for j, (ws) in enumerate(zip(ws_base)):
            
        fi.reinitialize_flow_field(wind_speed=[ws])
        fi.calculate_wake()
        turb_powers = fi.get_turbine_power()
        farm_power[j] = np.sum(turb_powers)
        prob[j] = prob_obj.pdf([ws])

    prob_normalized = prob/np.sum(prob);

    # Mean
    power_mean = np.dot(farm_power,prob_normalized)

    # Std Deviation
    E_x_2 = np.dot(np.square(farm_power) , prob_normalized)
    power_std = np.sqrt(E_x_2 - power_mean**2)
                                                
    return power_std
        

def step(a,x,fi,sim_time,pwr_4s, wnd_4s,pwr_std_4s):
    
    '''
    This function contains the dyanmics of the wake
    
    a :: index of the action to be chosen
    x :: state (wind speed)
    fi :: floris interface object
    sim_time :: time of simulation
    Actions :: Action space
    
    '''
    # Given the index, select the action:
    if wnd_4s < 2 or pwr_4s <1e-4:
        power = 0;
    else:
        fi.reinitialize_flow_field(wind_speed=[wnd_4s],sim_time = sim_time)
        fi.calculate_wake(yaw_angles=a)
        pow_array = fi.get_turbine_power()
        power = np.sum(pow_array)
    
    # State 
    no_turb = 9; 
    Wnd = np.zeros(no_turb,)
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    x_next = Wnd;
    
    # Variance of the wind speed 
    #power_dev = pow_dev(fi,wnd_4s,1.2)
    power_dev = pwr_std_4s;
    
    # Rewards for the system
    rewards = 0;
    low_pow = 0.95*1e6*pwr_4s
    up_pow = 1.05*1e6*pwr_4s
    
    # If power generated is greater than the 
    if power > up_pow:
        pow_asc = np.sort(pow_array);
        pow_asc_idx = np.argsort(pow_asc)
        cum_pow = np.cumsum(pow_asc)
        idx = np.abs(pwr_4s - cum_pow).argmin()
        power = cum_pow[idx]
        rewards = 1;
        
    elif power > low_pow and power < up_pow:
        rewards = 1;

    
    done = bool(0)
    return x_next, rewards, done, fi, power



def step_exp(a,x,fi,sim_time,pwr_4s, wnd_4s,pwr_std_4s):
    
    '''
    This function contains the dyanmics of the wake
    
    a :: index of the action to be chosen
    x :: state (wind speed)
    fi :: floris interface object
    sim_time :: time of simulation
    Actions :: Action space
    
    '''
    # Given the index, select the action:
    if wnd_4s < 2 or pwr_4s <1e-4:
        power = 0;
    else:
        fi.reinitialize_flow_field(wind_speed=[wnd_4s],sim_time = sim_time)
        fi.calculate_wake(yaw_angles=a)
        pow_array = fi.get_turbine_power()
        power = np.sum(pow_array)
    
    # State 
    no_turb = 9; 
    Wnd = np.zeros(no_turb,)
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    x_next = Wnd;
    
    # Variance of the wind speed 
    #power_dev = pow_dev(fi,wnd_4s,1.2)
    power_dev = pwr_std_4s;
    
    # Rewards for the system
    rewards = 0;
    low_pow = 0.95*1e6*pwr_4s
    up_pow = 1.05*1e6*pwr_4s
    
    # If power generated is greater than the 
    if power > up_pow:
        pow_asc = np.sort(pow_array);
        pow_asc_idx = np.argsort(pow_asc)
        cum_pow = np.cumsum(pow_asc)
        idx = np.abs(pwr_4s - cum_pow).argmin()
        power = cum_pow[idx]
        rewards = np.exp(-np.abs(power - pwr_4s));
        
    elif power > low_pow and power < up_pow:
        rewards = 1;
        
    else:
        rewards = np.exp(-np.abs(power - pwr_4s));
    
    done = bool(0)
    return x_next, rewards, done, fi, power

