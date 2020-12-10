# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 01:20:52 2020

@author: sanja
"""


'''
Given:
    1) mean wind speed at 5min intervals
    2) std dev of wind speed 5 min intervals
    3) Cp vs Wind look-up table
    
To find:
    1) Expected aerodynamic power at 5 min intervals
    2) Variance of the
    
Assumptions:
    1) Standard Gaussian random variable (continuous)
    
Additional info:
    We compute the expected value using the following equation
    E[power] = int_-oo^oo  power(V) Gauss(mu, std) dV

'''


import csv
import pandas as pd
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from datetime import time
from scipy.stats import norm
from scipy.stats import norm,multivariate_normal
from scipy import interpolate


# Read the csv file and extract the columns pertaining to wind speed.
df = pd.read_csv("prob_forecast_201908.csv")
mean_ws = np.array(df.iloc[:,1] );
std_ws = np.array(df.iloc[:,2])
mean_wd = np.array(df.iloc[:,3] );
std_wd = np.array(df.iloc[:,4])


power = [0.414,0.426,0.426,0.427,0.427,0.438,0.439,0.427,0.377,0.297,0.238,0.193,0.159,0.133,0.112,0.095,0.081,0.07,
         0.061,0.054,0.047,0.042,0.037,0.033]
speed = [4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,
         26.0,27.0]
f = interpolate.interp1d(speed, power, fill_value='extrapolate')


# Parameters for power
rho = 1.225;
R = 77/2;
# turbine parameters
D = 93
rho = 1.225
Area = np.pi * (D/2)**2
cut_in = 3.0
cut_out = 25.0
rated_power = 2300000 # each machine can produce 2.3MW

def power(test_speed):
    Cp = f(test_speed)
    if test_speed < 4.0:
        out_power= 0.0
    elif test_speed> cut_out:
        out_power = 0.0
    else:
        out_power = 0.5 * rho * Area * Cp * test_speed**3
    # rated power
    if out_power > rated_power:
        out_power = rated_power

    
    return out_power
    
# Generate 1000 samples: uniformly distributed.
n_sim = 1000; wd_var_scale = 10;
ws_base = np.random.uniform(low=5, high=20., size=n_sim)
k = 0;


with open('power_data_prob.csv', 'w', newline='') as csvfile:
    fieldnames = ['HRS','MIN','WND_MEAN', 'WND_STD', 'PWR_MEAN','PWR_STD','CONF INT']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range (len(mean_ws)): 
        
        if std_ws[i] < 1e-2:
            std_ws[i] = std_ws[i-1]
            
        prob_obj = multivariate_normal(mean_ws[i], std_ws[i] )
        farm_power = np.zeros(n_sim,)
        prob = np.zeros(n_sim,)
        for j, ws in enumerate(ws_base):
            
               
               turb_powers = power(ws);
               farm_power[j] = 48*turb_powers
               prob[j] = prob_obj.pdf(ws)
                 
               
  
                
        prob_normalized = prob/np.sum(prob);

        # Mean
        # this is just mean_power
        power_mean = np.dot(farm_power,prob_normalized)
        

        # Std Deviation
        E_x_2 = np.dot(np.square(farm_power) , prob_normalized)
        power_std = np.sqrt(E_x_2 - power_mean**2)
        
        
        if mean_ws[i] < 4:
            power_mean = 0;
            power_std =0
        
        # confidence interval
        prob_exceed=0.85
        conf = norm.ppf(1-prob_exceed)*power_std+power_mean
        
        
        hr, mn = divmod(k,60);
        writer.writerow({'HRS':hr,'MIN':mn,'WND_MEAN': mean_ws[i], 'WND_STD': std_ws[i],\
                         'PWR_MEAN':power_mean*1e-6,\
                         'PWR_STD':power_std*1e-6,'CONF INT':conf})
        k = k+5;

