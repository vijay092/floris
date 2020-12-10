# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 03:07:00 2020

@author: sanja
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:07:45 2020

@author: svijaysh
"""


import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import os
import math
import torch
import gpytorch



df = pd.read_csv("power_data_prob.csv")
df1 = pd.read_csv("power_data_lill.csv")
start = 1; # start time
delta = 30# no. of minutes



mean_ws = np.array(df.iloc[start:start + delta,2] );
std_ws = np.array(df.iloc[start:start + delta,3])
mean_pwr = np.array(df.iloc[start:start + delta,4] );
std_pwr = np.array(df.iloc[start:start + delta,5] );
t = np.linspace(0,5*delta/60,delta)

start = 0; # start time

mean_pwr_prob = np.array(df1.iloc[start:start + delta,5] )*1e-6;
std_pwr_prob = np.array(df1.iloc[start:start + delta,6] )*1e-6;



# Plots
fig, axs = plt.subplots(3,figsize=(8,7))
axs[0].errorbar(t, mean_ws, std_ws, uplims=True, lolims=True,marker='s', mfc='red',
         mec='green', ms=6, mew=3)
axs[0].set(ylabel='Wind Speed (m/s)')
axs[0].set_ylim([5, 15])
plt.ylabel('Wind Speed (m/s)')
axs[1].errorbar(t, mean_pwr, std_pwr, uplims=True, lolims=True,marker='s', mfc='red',
         mec='green', ms=6, mew=3)
axs[1].set(ylabel='Power (MW)')
axs[1].set_ylim([0, 130])
axs[2].errorbar(t, mean_pwr_prob, std_pwr_prob, uplims=True, lolims=True,marker='s', mfc='red',
          mec='green', ms=6, mew=3)
axs[2].set(xlabel='t (in hours)', ylabel='Power with wake (MW)')
axs[2].set_ylim([-20, 130])
plt.show()




