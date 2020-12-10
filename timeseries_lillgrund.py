import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def GetData():
    
    df = pd.read_csv("AGC_wind_plant_signal_2019_08_17.csv")
    df_wind = pd.read_csv("swift_201908_processed_mean.csv")

    
    # Data from file

    time_pow_4s = 4*np.linspace(0,len(df),len(df)+1)
    power_4s = np.array(df.iloc[:, 1])
    time_wnd_5min = 300*np.linspace(0,288,288+1)
    wind_5min = np.array(df_wind.iloc[4608:4897, 3])

    # Interpolate to get wind at 4s intervals
    wind_5min = interpolate.interp1d(time_wnd_5min,wind_5min)
    wind_4s = wind_5min(time_pow_4s)
    s = 0; e = 2000;
    return time_pow_4s[s:e], wind_4s[s:e], power_4s[s:e]



plt, axs = plt.subplots(2);
time_pow_4s, wind_4s, power_4s = GetData()
axs[0].plot(time_pow_4s,wind_4s)
axs[0].set(xlabel='time (s)',ylabel='speed (m/s)')
axs[1].plot(time_pow_4s,power_4s)
axs[1].set(xlabel='time (s)',ylabel='Power (MW)')
axs[0].grid()
axs[1].grid()










