# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 18:37:23 2025

@author: anton
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram
from scipy.io.wavfile import read, write
import glob, pickle

#%% LOAD DATA
load_path_list=glob.glob('../../../DATA/MVO/ASCII/*')


event_names=['hybrid','longperiod','lr','mi','rockfall','tremor']
component_list=['Z','N','E']


print('Loading dictonary')
with open('../data/labeled_clustering/event_dictionary.pkl', 'rb') as f:
    events = pickle.load(f)
    
#%% READ DATA TO LIST AND PLOT EXAMPLES OF EVERY CATEGORY

data_list=list()
labels_list=list()
times_list=list()


for idx_event, event_name in enumerate(event_names):
    events_type_list=events[event_name]

for idx in range(len(events_type_list)):
    data_list.append(events_type_list[idx]['Z'])
    times_list.append(events_type_list[idx]['times'])
    labels_list.append(event_name)
    

        
#%% PLOT ONE EXAMPLE TO LISTEN TO
dt=events['hybrid'][0]['dt']    # [s]
fs_seismic=1/dt # [Hz]
ex_idx=3

seismic_data=data_list[ex_idx]

# === Eingaben ===
fs_audio, audio_data = read("seismic_transposed_440_Hz.wav")
audio_data = audio_data.astype(np.float32) / 32767.0  # normalisiert

# Wenn nötig: Zeitvektor
t_audio = np.arange(len(audio_data)) / fs_audio

# --- Spektrogramm vorbereiten ---
f, t_spec, Sxx = spectrogram(seismic_data, fs=fs_seismic, nperseg=256, noverlap=256)

# --- Seismisches Signal vorbereiten (optional rohdaten oder bandpass) ---
seismic_plot = seismic_data  # oder filtered_data
t_seismic = np.arange(len(seismic_plot)) / fs_seismic

# --- Figure & Subplots ---
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True)

# Obere Achse: Seismik
ax1.plot(t_seismic, seismic_plot, color='blue')
line1 = ax1.axvline(x=0, color='red', linewidth=2)
ax1.set_ylabel("Seismik")
ax1.set_title("Seismisches Signal")

# Untere Achse: Spektrogramm
im = ax2.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-20), shading='auto')
line2 = ax2.axvline(x=0, color='red', linewidth=2)
ax2.set_ylabel("Hz")
ax2.set_xlabel("Zeit [s]")
ax2.set_ylim(1, 10)
fig.colorbar(im, ax=ax2, label='Leistung [dB]')
ax2.set_title("Spektrogramm Audio")

# --- Animation Funktion ---
def animate(frame):
    t_pos = frame / fps  # Zeit in Sekunden
    line1.set_xdata([t_pos, t_pos])
    line2.set_xdata([t_pos, t_pos])
    return line1, line2

# --- Video Parameter ---
duration = t_audio[-1]             # Gesamtdauer Audio
fps = 30                           # Frames pro Sekunde
n_frames = int(duration * fps)

anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, blit=True)

# --- Video speichern ---
anim.save("seismic_video.mp4", fps=fps, dpi=150, codec='libx264')

plt.close(fig)
print("✅ Video gespeichert: seismic_video.mp4")