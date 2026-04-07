# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 15:02:32 2025

@author: anton
"""
import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path
#%% LOAD DATA
load_path='../../data/raw_data'

list_events=glob.glob(load_path+'/*/month*/*')

list_events=[Path(f) for f in list_events]

list_events=[str(f) for f in list_events if any(f.iterdir())]


idx=0
folder=list_events[idx]
y = np.load(folder + '/data.npy')
time_arr = np.load(folder + '/time.npy')   # falls numpy datetimes; sonst konvertiere
while np.isnan(y).any() == True:
    print('nan detected,next...')
    idx += 1
    folder=list_events[idx]
    y = np.load(folder + '/data.npy')
    time_arr = np.load(folder + '/time.npy')   # falls numpy datetimes; sonst konvertiere
    


print("y.shape:", y.shape)
print("len(time):", len(time_arr))

# Quick stats
print("y: min,max,mean,nonzero fraction:",
      np.nanmin(y), np.nanmax(y), np.nanmean(y), np.count_nonzero(y) / y.size)

# Per-station stats (erste 5)
for i in range(min(5, y.shape[0])):
    tr = y[i]
    print(f"station {i}: min {np.nanmin(tr):.3e}, max {np.nanmax(tr):.3e}, nonzeros {np.count_nonzero(tr)}")

# If shapes mismatch, give direct hint:
if y.shape[1] != len(time_arr):
    print(">>> MISMATCH y samples vs time length. Will trim to min_len in safe mode.")
    

    
#%% check frequency content
fs = time_arr[1]-time_arr[0]
# Fourier-Transformation
fft_vals = np.fft.fft(y[0])
fft_freqs = np.fft.fftfreq(len(y[0]), 1/fs)

# Betrag nehmen (Amplituden)
fft_magnitude = np.abs(fft_vals)

# Nur positive Frequenzen behalten
mask = fft_freqs >= 0
fft_freqs = fft_freqs[mask]
fft_magnitude = fft_magnitude[mask]

# Plot
plt.plot(fft_freqs, fft_magnitude)
plt.xlabel("Frequenz [Hz]")
plt.ylabel("Amplitude")
plt.title("Frequenzspektrum")
plt.show()
    
#%% check how many events contain nans
list_events_data=glob.glob(load_path+'/*/month*/*/data.npy')

from concurrent.futures import ThreadPoolExecutor
import numpy as np

def check_nans(file):
    y = np.load(file, mmap_mode='r')
    return np.isnan(y).any()

with ThreadPoolExecutor() as executor:
    results = list(executor.map(check_nans, list_events_data))

number_nans = sum(results)
        

    
#%% PLOT DATA

fig,ax = plt.subplots(len(y),1,sharex=True)

for idx_sta in range(len(y)):
    ax[idx_sta].plot(time_arr,y[idx_sta])
    print(np.isnan(y[idx_sta]).any())