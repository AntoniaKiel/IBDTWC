# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 12:50:46 2025

@author: anton
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import pywt
import torch
import pandas as pd
#%% LOAD DATA
load_path_catalog='../data/'
# 1. --- Load Catalog ---
load_path = load_path_catalog
df = pd.read_csv(load_path + "catalog_data.csv")
beam_paths = df.loc[(df["beam"] == True) & (df["used"] == True), "ID"].unique().tolist()
print(f"Number of events: {len(beam_paths)}")
print(f'Path of medoid cluster 5:{beam_paths[1195]}')

data_med=np.load('../data/event015/data_red.npy')
BEAMS_med=np.load('../data/event015/BEAM_results.npy',allow_pickle=True)

# load data from clustering
# load_path =load_path_clustering
# labels = np.load(load_path + 'labels.npy')
# min_distances = np.load(load_path + 'min_distances.npy', allow_pickle=True)
# all_distances = np.load(load_path + 'all_distances.npy', allow_pickle=True)
# BEAMS = np.load(load_path + 'BEAMS.npy', allow_pickle=True)
# TIMES = np.load(load_path + 'TIMES.npy', allow_pickle=True)
# TIMES_UTC=np.load(load_path + 'TIMES_UTC.npy', allow_pickle=True)
# distance_threshold=np.load(load_path +'/distance_threshold.npy')

load_path='../data/2_clustering_results/good_results/2025_11_14/scRadi3_seed30/n_clusters7/'

files_list=glob.glob(load_path+'*')

BEAMS=np.load(load_path+'BEAMS.npy',allow_pickle=True)
TIMES=np.load(load_path+'TIMES.npy',allow_pickle=True)
labels=np.load(load_path+'labels.npy')
min_distances=np.load(load_path+'min_distances.npy')
all_distances=np.load(load_path+'all_distances.npy',allow_pickle=True)

idx_medoids=np.argwhere(min_distances==0)
print(idx_medoids)

beams_medoids=list()
times_medoids=list()

event_names_list=[f"cluster {l+1}"for l in np.unique(labels)[1:]]
event_names_list=["outlier"]+event_names_list
print(event_names_list)
event_names=np.unique(labels)


from clustering_labeled_func import plot_events_and_pie

# plot_events_and_pie(
#     TIMES,BEAMS,labels,
#     event_names,event_names_list)


percentage_list=[[1.2,1.2],[1.2,1.2],[1.2,1.2],[1.2,1.2],[1.2,1.2],[1.2,0.9],[1.0,1.0],[0.5,0.5],[1.1,1.1]]
# plot_events_and_pie(
#     TIMES,BEAMS,labels,
#     event_names,event_names_list,idx_medoids,
#     figsize=(14,7),
#     percentage_plot_list=percentage_list,
#     colormap="tab20b"
#     )

#%% PLOT MORE DETAILS
colormap='tab20b'
cmap = plt.get_cmap(colormap)
colors = cmap(np.linspace(0, 1, len(event_names)))
color_map = {event: colors[i] for i, event in enumerate(event_names)}
color_map[-1] = 'lightgray'  # Outlier immer grau


labels_for_medoids=np.array([labels[i][0] for i in idx_medoids])
labels_for_medoids = np.array([int(lab) for lab in labels_for_medoids])
idx_medoids = np.array(idx_medoids, dtype=int)

sort_idx = np.argsort(labels_for_medoids)
labels_for_medoids_sorted = labels_for_medoids[sort_idx]
idx_medoids_sorted = idx_medoids[sort_idx]


fig,ax=plt.subplots(len(idx_medoids),2,figsize=(8,12),sharex=True)

waveletname='cmor1-5'
freq_min=3
freq_max=8
sampling_rate=1/(TIMES[0][1]-TIMES[0][0])
freqs_desired = np.linspace(freq_min, freq_max, 50)
scales = pywt.central_frequency(waveletname) / (freqs_desired / sampling_rate)

def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

for idx in range(len(idx_medoids)):
    t=TIMES[idx_medoids_sorted[idx][0]]
    data=to_np(BEAMS[idx_medoids_sorted[idx][0]]).flatten()
    ax[idx][0].plot(t,data,color=color_map[idx])
    ax[idx][0].set_title(event_names_list[idx+1])
    
    # Calculate scales for CWT once
    scales_min = pywt.scale2frequency(waveletname, freq_max) * sampling_rate
    scales_max = pywt.scale2frequency(waveletname, freq_min) * sampling_rate
    scales = np.arange(scales_min, scales_max, 0.2)
    coeffs, freqs = pywt.cwt(data, scales, waveletname, sampling_period=1/sampling_rate)

    coeffs_log = np.log10(np.abs(coeffs)+1e-12)
    ax[idx][1].pcolormesh(t, freqs, coeffs_log, shading='gouraud', cmap='magma',
                       vmin=np.percentile(coeffs_log,3), vmax=np.percentile(coeffs_log,99))
    ax[idx][1].set_ylim(freq_min,freq_max)
    
    
plt.show()   
#%% DETAIL PLOTS FOR CLUSTER 5
print(BEAMS_med[0].keys())
plt.plot(np.mean(BEAMS_med[0]['shifted_traces'],axis=0))
plt.show()

import torch

slowness_space = BEAMS_med[0]['slowness_space']     # [45000, 2]
slowness_power = BEAMS_med[0]['slowness power']     # [45000]
slowness_best  = BEAMS_med[0]['slowness_best']       # [2]

# Index des Maximums (robust)
idx_best = torch.argmax(slowness_power)
slowness_best_val = slowness_space[idx_best]

sx = slowness_space[:,0]
sy = slowness_space[:,1]

azimuth = torch.atan2(sy, sx)      # Winkel [-pi, pi]
radius  = torch.sqrt(sx**2 + sy**2)

import numpy as np
import matplotlib.pyplot as plt

az = azimuth.numpy()
r  = radius.numpy()
p  = slowness_power.numpy()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='polar')

slowness_space = BEAMS_med[0]['slowness_space']
slowness_power = BEAMS_med[0]['slowness power']
slowness_best = BEAMS_med[0]['slowness_best']

def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

slowness_space_np = to_np(slowness_space)
slowness_power_np = to_np(slowness_power).flatten()
slowness_best_np = to_np(slowness_best)
from scipy.interpolate import griddata

# Polar coordinates from your sample
theta = np.arctan2(slowness_space_np[:,1], slowness_space_np[:,0])
r = np.linalg.norm(slowness_space_np, axis=1)
p = slowness_power_np

# Define regular grid
theta_grid = np.linspace(-np.pi, np.pi, 360)
r_grid = np.linspace(r.min(), r.max(), 200)
TH, RR = np.meshgrid(theta_grid, r_grid)

# Convert grid back to slowness components to interpolate
SX = RR * np.cos(TH)
SY = RR * np.sin(TH)

# Interpolate power onto the regular grid
P_grid = griddata(
    points=np.column_stack((slowness_space_np[:,0], slowness_space_np[:,1])),
    values=p,
    xi=np.column_stack((SX.ravel(), SY.ravel())),
    method='linear'
).reshape(RR.shape)

im = ax.pcolormesh(TH, RR, P_grid, shading='gouraud', cmap='magma')

az_best = float(azimuth[idx_best])
r_best  = float(radius[idx_best])

ax.scatter(az_best, r_best, color='red', s=50, marker='*', label='Maximum')
ax.legend()


plt.show()



#%%
idx=5

fig,ax=plt.subplots(2,1,sharex=True,figsize=(8,6))


t=TIMES[idx_medoids_sorted[idx][0]]
data=to_np(BEAMS[idx_medoids_sorted[idx][0]]).flatten()

# t=TIMES[886]
# data=to_np(BEAMS[886]).flatten()

ax[0].plot(t,data,color=color_map[idx])
ax[0].set_title(f"cluster {labels[idx_medoids_sorted[idx][0]]}")
#ax[0].set_title(f"cluster {labels[idx_medoids_sorted[idx][0]]}")

# Calculate scales for CWT once
scales_min = pywt.scale2frequency(waveletname, freq_max) * sampling_rate
scales_max = pywt.scale2frequency(waveletname, freq_min) * sampling_rate
scales = np.arange(scales_min, scales_max, 0.2)
coeffs, freqs = pywt.cwt(data, scales, waveletname, sampling_period=1/sampling_rate)

coeffs_log = np.log10(np.abs(coeffs)+1e-12)
ax[1].pcolormesh(t, freqs, coeffs_log, shading='gouraud', cmap='magma',
                   vmin=np.percentile(coeffs_log,3), vmax=np.percentile(coeffs_log,99))


ax[0].set_yticklabels([])

ax[1].set_xlabel('time [s]')
ax[1].set_ylabel('frequency [Hz]')

fig.tight_layout()


plt.show()
    

#%% PLOT MEDOIDS AND SEVERAL EXAMPLES OF EVERY CLUSTER
def select_examples_fraction(distances_idx, N_EXAMPLES, fraction_limit=1.0, include_medoid=False):
    """
    distances_idx: Liste von (index, distance) Paaren für den Cluster
    fraction_limit: float 0-1, Anteil der Samples von kleinster Distanz
    include_medoid: ob Medoid in Auswahl sein darf
    """
    if not include_medoid:
        # Medoid (distance=0) rausnehmen
        distances_idx = [pair for pair in distances_idx if pair[1] > 0]

    if len(distances_idx) == 0:
        return []

    # nach Distanz sortieren
    distances_sorted = sorted(distances_idx, key=lambda x: x[1])

    # nur ein bestimmter Anteil (z.B. 50%) der Liste verwenden
    n_keep = max(1, int(len(distances_sorted) * fraction_limit))
    distances_capped = distances_sorted[:n_keep]

    # gleichmäßig N_EXAMPLES auswählen
    n_samples = len(distances_capped)
    idx_pos = np.linspace(0, n_samples-1, min(N_EXAMPLES, n_samples)).astype(int)

    selected = [distances_capped[i][0] for i in idx_pos]
    return selected


N_EXAMPLES=3
np.random.seed(25)
FRACTION_LIMIT=1

colormap='copper'
cmap = plt.get_cmap(colormap)
colors = cmap(np.linspace(0, 1, len(event_names)))
color_map = {event: colors[i] for i, event in enumerate(event_names)}
color_map[-1] = 'lightgray'  # Outlier immer grau



# --- Cluster-Zuordnung ------------------------------------
FRACTION_LIMIT = 0.65   # z.B. 50% der maximalen Cluster-Distanz

examples_per_cluster = {}
medoid_for_cluster = {}

unique_clusters = np.unique(labels)

for cluster_id in unique_clusters:

    idx_list = np.where(labels == cluster_id)[0]

    # Paare (index, distance)
    dist_pairs = [(i, min_distances[i]) for i in idx_list]

    if cluster_id == -1:
        n_samples = len(dist_pairs)
        n_select = min(N_EXAMPLES + 1, n_samples)
        selected = np.random.choice([i for i, _ in dist_pairs], size=n_select, replace=False)
        examples_per_cluster[cluster_id] = list(selected)
    else:
        # Medoid
        medoid_idx = idx_medoids[np.argwhere(labels[idx_medoids]==cluster_id)[0][0]]
        medoid_for_cluster[cluster_id] = medoid_idx

        examples_per_cluster[cluster_id] = select_examples_fraction(
            dist_pairs,
            N_EXAMPLES,
            fraction_limit=FRACTION_LIMIT,
            include_medoid=False
        )


# --- Reihenfolge fürs Plotten: Outlier ganz oben -----------
ordered_clusters = [-1] + list(labels_for_medoids_sorted)
n_rows = len(ordered_clusters)


# PLOT ERSTELLEN 
fig, ax = plt.subplots(
    n_rows, N_EXAMPLES+1,
    figsize=(6.6, 9),
    sharex=True,
    gridspec_kw={'hspace': 0.2, 'wspace': 0}  # wspace leicht, hspace=0
)

for row, cluster_id in enumerate(ordered_clusters):

    # -----------------------------
    #      OUTLIER-ZEILE
    # -----------------------------
    if cluster_id == -1:
        example_indices = examples_per_cluster[-1]

        for j, ex_idx in enumerate(example_indices):
            t = TIMES[ex_idx]
            data = to_np(BEAMS[ex_idx]).flatten()
            ax[row][j].plot(t, data, color=color_map[-1],linewidth=0.5)

        # Outlier als y-Label
        ax[row][0].set_ylabel('outliers')

    # -----------------------------
    #      REGULÄRE CLUSTER
    # -----------------------------
    else:
        # Medoid
        medoid_idx = medoid_for_cluster[cluster_id]
        t_med = TIMES[medoid_idx][0]
        data_med = to_np(BEAMS[medoid_idx]).flatten()[0]

        ax[row][0].plot(t_med, data_med, color=color_map[cluster_id],
                        linewidth=0.5)
        ax[row][0].set_ylabel(f"{event_names_list[row]}")
        #ax[row][0].set_title("medoid", fontsize=9)

        # Beispiele
        example_indices = examples_per_cluster[cluster_id]
        for j, ex_idx in enumerate(example_indices):
            t_ex = TIMES[ex_idx]
            data_ex = to_np(BEAMS[ex_idx]).flatten()
            ax[row][j+1].plot(t_ex, data_ex, color=color_map[cluster_id],
                              linewidth=0.5, alpha=0.4)

# ------------------------------------------------------------
#     ACHSEN-NACHBEARBEITUNG (X-Label & Y-Ticks entfernen)
# ------------------------------------------------------------

# Y-Ticks *entfernen*, y-Labels *behalten*
for r in range(n_rows):
    for c in range(N_EXAMPLES+1):
        ax[r][c].set_yticks([])
        ax[r][c].tick_params(axis='x', direction='in')

# X-Label auf letzte Zeile
for c in range(N_EXAMPLES+1):
    ax[-1][c].set_xlabel("time [s]")

plt.tight_layout()
plt.savefig('../../papers/figures/icequakes.pdf', dpi=300)
plt.show()
    
    
    
#%%
cid = 0
cd = min_distances[labels == cid]
print("n samples:", cd.shape[0])
print("min (should be 0):", cd.min())
print("max:", cd.max())
print("5 smallest:", np.sort(cd)[:5])
print("5 largest:", np.sort(cd)[-5:])

    
    
    
