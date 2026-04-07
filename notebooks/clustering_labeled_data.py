# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 15:03:06 2025

@author: anton
"""
import numpy as np
import glob
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 10
})

import pickle
import os
import logging
from pathlib import Path
from scipy.signal import decimate
from clustering_labeled_func import plot_events_and_pie, read_eventtype, dtw_function

log_file = "clustering.log"
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
#%% CREATE OR LOAD EVENT CATALOG

load_path_list=glob.glob('../../../DATA/MVO/ASCII/*')


event_names=['hybrid','longperiod','lr','mi','rockfall','tremor']
component='Z'

dict_path=Path(__file__).parent.parent / "data" / "labeled_clustering" 
print(dict_path)

if not os.path.exists(dict_path /'event_dictionary.pkl'):
    if not os.path.exists('../data/labeled_clustering'):
        os.makedirs('../data/labeled_clustering')
        
        
    events={'hybrid':{},'longperiod':{},'lr':{},'mi':{},'rockfall':{}, 'tremor':{}}
    station_name='MBGA'
    
    for idx in range(len(event_names)):
        read_eventtype(events,idx,load_path_list,event_names,station_name)
        
        
    with open('../data/labeled_clustering/event_dictionary.pkl', 'wb+') as f:
        pickle.dump(events, f)        
        
else:
    print('Loading dictonary')
    with open(dict_path /'event_dictionary.pkl', 'rb') as f:
        events = pickle.load(f)
        
#%% READ DATA TO LIST AND PLOT EXAMPLES OF EVERY CATEGORY

data_list=list()
labels_list=list()
times_list=list()

event_names_label=['HYB','LPE','LPE+ROC','MISC','ROC','VTE']

n_examples=2
fig,ax=plt.subplots(n_examples,len(event_names),figsize=(10,7),sharex='col',dpi=200)
sampling_factor=1

for idx_event, event_name in enumerate(event_names):
    events_type_list=events[event_name]
    
    for idx in range(len(events_type_list)):
        data_ds = decimate(events_type_list[idx]['Z'], q=sampling_factor, ftype='iir', zero_phase=True)
        dt_new = events_type_list[idx]['dt'] * sampling_factor        
        data_list.append(data_ds)
        times_list.append(np.arange(len(data_ds)) * dt_new)
        labels_list.append(event_name)
        
        

        
        if idx < n_examples:
            ax[idx][idx_event].plot(events_type_list[idx]['times'],events_type_list[idx]['Z'])
            ax[idx][idx_event].set_title(event_name)
            ax[idx][idx_event].set_yticks([])

plt.show()



#%%
sampling_factor = 4
data_list=list()
labels_list=list()
times_list=list()
for idx_event, event_name in enumerate(event_names):
    events_type_list = events[event_name]
    
    for idx in range(len(events_type_list)):
        data_ds = decimate(events_type_list[idx]['Z'], q=sampling_factor, ftype='iir', zero_phase=True)
        dt_new = events_type_list[idx]['dt'] * sampling_factor        
        data_list.append(data_ds)
        times_list.append(np.arange(len(data_ds)) * dt_new)
        labels_list.append(event_name)
        
percentage_list=[[1.2,1.2],[1.2,1.2],[1.2,1.2],[1.2,1.2],[1.0,1.0],[1.2,1.2],[1.2,1.2]]   

plot_events_and_pie(
times_list,
data_list,
0.5,
18,
labels_list,
event_names,
event_names_label,
colormap="tab20b",
percentage_plot_list=percentage_list,
IDX_PLOT=[0,0,0,0,120,1],
dpi=300)
    
        
#%% EXCECUTE CLUSTERING
sakoe_chiba_radius_time=10
n_clusters=2
init_sample_size=350
n_min_cluster=3
n_max_cluster=10
subset=1000
#dt=events['hybrid'][0]['dt']
dt=dt_new
sampling_freq=1/dt_new
n_seed=42
sigma_factor=2

new_clustering=True

print(type(data_list),type(data_list[0]))

if new_clustering==True:
    dtw_function(data_list,times_list,labels_list, sakoe_chiba_radius_time, n_clusters, init_sample_size,
                     n_min_cluster, n_max_cluster, sampling_freq,
                     admissibility=sigma_factor, b_min=5,max_buffer_age=100,
                     save_data=False, reextracting_init_beams=True,
                     recalc_init_clustering=True, test_n_clusters=False, subset=subset,
                     n_seed=n_seed)
    
    save_path_main = f'../data/labeled_clustering/clustering_results/scRadi{sakoe_chiba_radius_time}_seed{n_seed}'
    save_path_cluster = f'{save_path_main}/n_clusters{n_clusters}'
    # --- Load Final Phase (already back in original order) ---
    labels_final = np.load(save_path_cluster + '/labels_final.npy')
    min_distances_final = np.load(save_path_cluster + '/min_distances_final.npy')
    all_distances_final = np.load(save_path_cluster + '/all_distances_final.npy', allow_pickle=True)
    BEAMS = np.load(save_path_cluster + '/BEAMS.npy', allow_pickle=True)
    TIMES = np.load(save_path_cluster + '/TIMES.npy', allow_pickle=True)
    TIMES_UTC = np.load(save_path_cluster + '/TIMES_UTC.npy', allow_pickle=True)
    shuffle_indices = np.load(save_path_cluster + '/shuffle_indices.npy')
    subset_idxs = np.load(save_path_cluster + '/subset_idxs.npy', allow_pickle=True)
    threshold_final = np.load(save_path_cluster + '/distance_threshold_final.npy')
    medoids = np.load(save_path_cluster + '/medoids.npy', allow_pickle=True)
    
    # --- Load Initial & Add Phase (original Reihenfolge) ---
    labels_init = np.load(save_path_cluster + '/labels_init.npy')
    min_distances_init = np.load(save_path_cluster + '/min_distances_init.npy')
    labels_add = np.load(save_path_cluster + '/labels_add.npy')
    min_distances_add = np.load(save_path_cluster + '/min_distances_add.npy')
    threshold_init = np.load(save_path_cluster + '/distance_threshold_init.npy')
    threshold_add = np.load(save_path_cluster + '/distance_threshold_add.npy')
    
    # --- Optional: quick checks ---
    print("Final clusters:", np.unique(labels_final))
    print("Initial clusters:", np.unique(labels_init))
    print("Added clusters:", np.unique(labels_add))
    print("Thresholds -> init:", threshold_init, "add:", threshold_add, "final:", threshold_final)
else:
    save_path_main = f'../data/labeled_clustering/clustering_results/scRadi{sakoe_chiba_radius_time}_seed{n_seed}'
    save_path_cluster = f'{save_path_main}/n_clusters{n_clusters}_good'
    # --- Load Final Phase (already back in original order) ---
    labels_final = np.load(save_path_cluster + '/labels_final.npy')
    min_distances_final = np.load(save_path_cluster + '/min_distances_final.npy')
    all_distances_final = np.load(save_path_cluster + '/all_distances_final.npy', allow_pickle=True)
    BEAMS = np.load(save_path_cluster + '/BEAMS.npy', allow_pickle=True)
    TIMES = np.load(save_path_cluster + '/TIMES.npy', allow_pickle=True)
    TIMES_UTC = np.load(save_path_cluster + '/TIMES_UTC.npy', allow_pickle=True)
    shuffle_indices = np.load(save_path_cluster + '/shuffle_indices.npy')
    threshold_final = np.load(save_path_cluster + '/distance_threshold_final.npy')
    medoids = np.load(save_path_cluster + '/medoids.npy', allow_pickle=True)
    subset_idxs = np.load(save_path_cluster + '/subset_idxs.npy', allow_pickle=True)
    
    # --- Load Initial & Add Phase (original Reihenfolge) ---
    labels_init = np.load(save_path_cluster + '/labels_init.npy')
    min_distances_init = np.load(save_path_cluster + '/min_distances_init.npy')
    labels_add = np.load(save_path_cluster + '/labels_add.npy')
    min_distances_add = np.load(save_path_cluster + '/min_distances_add.npy')
    threshold_init = np.load(save_path_cluster + '/distance_threshold_init.npy')
    threshold_add = np.load(save_path_cluster + '/distance_threshold_add.npy')
    
    # --- Optional: quick checks ---
    print("Final clusters:", np.unique(labels_final))
    print("Initial clusters:", np.unique(labels_init))
    print("Added clusters:", np.unique(labels_add))
    print("Thresholds -> init:", threshold_init, "add:", threshold_add, "final:", threshold_final)  
    


labels_final_list=np.unique(labels_final)
event_names_final=[f"cluster {i}" for i in labels_final_list]

idx_medoids=np.argwhere(min_distances_final==0)
percentage_list=[[1.2,1.2],[1.2,1.2],[1.2,1.2],[1.2,1.2],[1.0,1.0],[1.3,1.3],[0.6,0.6]]
plot_events_and_pie(
    times_list=TIMES,
    data_list=BEAMS,
    freq_min=1,
    freq_max=18,
    colormap="tab20b",
    labels_list=labels_final,
    event_names=labels_final_list,
    event_names_label=event_names_final,
    idx_medoids=idx_medoids,
    percentage_plot_list=percentage_list)
        
vals, counts = np.unique(min_distances_final, return_counts=True)


zero_mask = np.isclose(min_distances_final, 0.0, atol=1e-21)
zero_idx = np.where(zero_mask)[0]

#%% ACCURACY ANALYSIS
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------
# 1) True-Labels in Zahlen umwandeln
# ---------------------------------------------------------
y_true_str = [labels_list[idx] for idx in subset_idxs]
mapping = {name: i for i, name in enumerate(event_names)}
y_true = np.array([mapping[s] for s in y_true_str])

# nur die true Klassen: 0 .. len(event_names)-1
true_labels = np.unique(y_true)     

# predicted labels (Cluster IDs)
pred_labels, pred_counts = np.unique(labels_final, return_counts=True)
count_map = dict(zip(pred_labels, pred_counts))

# Confusion Matrix (normalize per prediction)
M_conf = confusion_matrix(
    y_true,
    labels_final,
    normalize='pred'
)
M_conf_total=confusion_matrix(
    y_true,
    labels_final
)

# ---------------------------------------------------------
# 2) Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.1, 3),dpi=300)

heat = ax.imshow(M_conf, cmap="Purples", origin="upper",vmin=0,vmax=1)
# cbar = plt.colorbar(heat, ax=ax, shrink=0.7)
# cbar.set_label("Normalized per predicted label [%]")
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="5%", pad=0.1)  # pad = Abstand zur Achse

cbar = fig.colorbar(heat, cax=cax, orientation="horizontal")
cbar.set_label("Normalised detection per predicted label [%]", labelpad=5)
cbar.ax.xaxis.set_label_position("top")       # Label nach oben
cbar.ax.xaxis.set_ticks_position("top")       # optional: Ticks oben anzeigen

ax.set_xlim(-0.5,4.5)
# ---------------------------------------------------------
# 3) Tick labels
# ---------------------------------------------------------
ax.set_xticks(np.arange(len(pred_labels)))
ax.set_yticks(np.arange(len(true_labels))+1)

cluster_names=[f'Outlier \n ({count_map[pred_labels[0]]})']+[f"Cluster {l} \n ({count_map[l]})" for l in pred_labels[1:]]

ax.set_xticklabels(cluster_names, rotation=90, ha='center',va='top')
ax.set_yticklabels([event_names_label[l] for l in true_labels])

ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")

#ax.set_title(f'sigma = {sigma_factor}')
# ---------------------------------------------------------
# 4) Text overlay
# ---------------------------------------------------------
for i in range(1,M_conf.shape[0]):
    for j in range(len(np.unique(labels_final))):
        val = M_conf_total[i, j]
        ax.text(
            j, i,
            f"{val}",
            ha="center",
            va="center",
            color="white" if abs(M_conf[i,j]) > 0.6 else "black",
            fontsize=7
            
        )
        
ax.set_ylim(0.5,6.5)
ax.invert_yaxis()

#plt.tight_layout()
#plt.savefig('../../papers/figures/conf_matrix.pdf',dpi=300,bbox_inches='tight')
plt.show()



from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(labels_true, labels_pred, exclude_classes=None):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # -----------------------------
    # 1) bestimmte Klassen ausschließen
    # -----------------------------
    if exclude_classes is not None:
        exclude_classes = set(exclude_classes)
        mask = ~np.isin(labels_true, list(exclude_classes))
        labels_true = labels_true[mask]
        labels_pred = labels_pred[mask]

    # -----------------------------
    # 2) Unique classes
    # -----------------------------
    true_classes  = np.unique(labels_true)
    pred_clusters = np.unique(labels_pred)

    # Confusion matrix: rows=true, columns=pred
    C = confusion_matrix(labels_true, labels_pred, labels=pred_clusters)

    # Hungarian: maximize matches → minimize negative
    cost = -C
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build mapping
    mapping = { pred_clusters[c]: true_classes[r] for r, c in zip(row_ind, col_ind) }

    # Apply mapping
    labels_pred_remapped = np.array([mapping.get(p, -1) for p in labels_pred])

    accuracy = np.mean(labels_pred_remapped == labels_true)

    return accuracy, mapping



mask = (labels_final != -1)

acc, mapping = clustering_accuracy(
    np.array(y_true)[mask], 
    np.array(labels_final)[mask],
    exclude_classes=[3]
)
print("Accuracy:", acc)
print("Mapping:", mapping)


acc_all, mapping_all = clustering_accuracy(y_true, labels_final)
print("Accuracy (with outliers):", acc_all)


#%% PLOT ONE LARGE OVERVIEW OF DATA FOR PAPER









