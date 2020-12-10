import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def GetData():
    
    df = pd.read_csv("real_time_reg.csv")
    df_wind = pd.read_csv("nwtc_80m_5min_2019.csv")
    df_std = pd.read_csv("power_data_wake.csv")
    
    # Data from file
    start = 0
    end = 10000
    time_pow_4s = 4*np.array(df.iloc[start:end, 0])
    power_4s = np.array(df.iloc[start:end, 1])
    time_wnd_5min = 300*np.linspace(0,288,288)
    wind_5min = np.array(df_wind.iloc[16992:17280, 1])
    pwr_std_5min = np.array(df_wind.iloc[16992:17280, 1])
    
    
    # Interpolate to get wind at 4s intervals
    wind_5min = interpolate.interp1d(time_wnd_5min,wind_5min)
    wind_4s = wind_5min(time_pow_4s)
    
    # Interpolate to get wind at 4s intervals
    pwr_5min = interpolate.interp1d(time_wnd_5min,pwr_std_5min)
    power_std_4s = pwr_5min(time_pow_4s)    
    return time_pow_4s, wind_4s, power_4s
plt, axs = plt.subplots(2);
time_pow_4s, wind_4s, power_4s = GetData()
axs[0].plot(time_pow_4s,wind_4s)
axs[0].set(xlabel='time (s)',ylabel='speed (m/s)')
axs[1].plot(time_pow_4s,power_4s)
axs[1].set(xlabel='time (s)',ylabel='Power (MW)')