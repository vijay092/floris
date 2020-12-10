# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:16:49 2020

@author: sanja
"""
import csv
import pandas as pd
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import time
from scipy.stats import norm,multivariate_normal
import floris.tools as wfct

# Read the csv file and extract the columns pertaining to wind speed.

df = pd.read_csv("real_time_reg");