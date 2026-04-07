# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:38:21 2025

@author: anton
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

#%%
# =============================================================================
# LOAD DATA
# =============================================================================
load_path='../../data/detected_events/'

list_events=glob.glob(load_path+'*/month*/*/event*')

print(len(list_events))

idx_plot=0

data_raw=np.load(list_events[idx_plot]+'/data_raw.npy')
data_red=np.load(list_events[idx_plot]+'/data_red.npy')
data_red_15s=np.load(list_events[idx_plot]+'/data_red_15s.npy')

times=np.load(list_events[idx_plot]+'/times.npy',allow_pickle=True)
times_15s=np.load(list_events[idx_plot]+'/times_15s.npy',allow_pickle=True)

# results=np.load(str(Path(list_events[idx_plot]).parent) +'/STA_LTA_results.npz', allow_pickle=True)
# data_STA_LTA=results['STA_LTA_vec_all']
# times_STA_LTA=results['STA_LTA_time']
# events_all_stations=results['events_all_stations']


save_file = str(Path(list_events[idx_plot]).parent) + '/STA_LTA_results.pkl'
with open(save_file, 'rb') as f:
    results = pickle.load(f)

data_STA_LTA = results["STA_LTA_vec_all"]
times_STA_LTA = results["STA_LTA_time"]
events_all_stations = results["events_all_stations"]

#%% Plot different data

fig, ax= plt.subplots(16,3)

for idx_sta in range(len(data_red)):
    ax[idx_sta][0].plot(times_STA_LTA,data_STA_LTA[idx_sta])
    ax[idx_sta][0].set_xlim(times[0],times[-1])
    
    ax[idx_sta][1].plot(times,data_red[idx_sta])
    
    ax[idx_sta][2].plot(times_15s,data_red_15s[idx_sta])


