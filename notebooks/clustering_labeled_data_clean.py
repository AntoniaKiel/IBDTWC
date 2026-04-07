# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 11:18:06 2026

@author: anton
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import pandas as pd

poster_fig=False

if poster_fig==True:
    plt.rcParams.update({

        # ===== Schrift =====
        "font.family": "DejaVu Sans",
        "font.size": 16,

        "axes.titlesize": 18,
        "axes.labelsize": 17,

        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,

        # ===== Farben =====
        "text.color": "#1E3A5F",
        "axes.labelcolor": "#1E3A5F",
        "axes.titlecolor": "#1E3A5F",

        "xtick.color": "#3E5F8A",
        "ytick.color": "#3E5F8A",

        # ===== Achsenlinien =====
        "axes.edgecolor": "#6F8FB3",
        "axes.linewidth": 1.2,

        # ===== Grid =====
        "axes.grid": False,
        "grid.color": "#D3E2F3",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,

        # ===== Hintergrund =====
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # ===== Legende =====
        "legend.frameon": False,

        # ===== Linien =====
        "lines.linewidth": 2.5,

        # ===== Export =====
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })
else:
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.titlesize": 10
    })

cm_to_inch = 1 / 2.54

import pickle
import os
from scipy.signal import decimate

from clustering_labeled_func import plot_events_and_pie, read_eventtype, dtw_function, bandpass_filter

#%% CREATE OR LOAD EVENT CATALOG
load_path_list=glob.glob('../../../DATA/MVO/ASCII/*')


event_names=['hybrid','longperiod','lr','mi','rockfall','tremor']  # names of volcano-seismic events
event_names_label=['HYB','LPE','LPE+ROC','MISC','ROC','VTE'] # != to saved raw data but useful for labeling
component='Z'                       # choose component 
station_name='MBGA'                 # choose station

# if the catalog is not yet created, read data into correct format
# else simply read catalog (save computation time)
if not os.path.exists('../data/labeled_clustering/event_dictionary.pkl'):
    # prepare saving path
    if not os.path.exists('../data/labeled_clustering'):
        os.makedirs('../data/labeled_clustering')
        
    # prepare dictonary
    events={name: {} for name in event_names}
    
    
    for idx in range(len(event_names)):
        read_eventtype(events,idx,load_path_list,event_names,station_name)
        
        
    with open('../data/labeled_clustering/event_dictionary.pkl', 'wb+') as f:
        pickle.dump(events, f)        
        
else:
    print('Loading dictonary')
    with open('../data/labeled_clustering/event_dictionary.pkl', 'rb') as f:
        events = pickle.load(f)
        
 
# prepare lists which contain the bandpass-filtered data
data_list=list()
labels_list=list()
labels_idx_list=list()
times_list=list()


for idx_event, event_name in enumerate(event_names):
    events_type_list=events[event_name]
    
    for idx in range(len(events_type_list)):
        dt = events_type_list[idx]['dt']  
        data = bandpass_filter(events_type_list[idx]['Z'],dt,0.5,15)    # bp filter suitable freq-range
        data_list.append(data)
        times_list.append(np.arange(len(data)) * dt)
        labels_list.append(event_name)
        labels_idx_list.append(idx_event)
    
#%% CALCULATE OR READ CLUSTERING RESULTS

# set parameters for IB-DTWC
sakoe_chiba_radius_time=10  # SC-radius [s]
n_clusters=3    
init_sample_size=350
n_min_cluster=3
n_max_cluster=10
subset=None         # for entire data considered, useful for debugging
sampling_freq=1/dt
n_seed=42
sigma_factor=2      # strickness for clustering

# load results of SC-radius of 10 s -> addition of _10 in variable name
save_path_main = f'../data/labeled_clustering/clustering_results/scRadi{sakoe_chiba_radius_time}_seed{n_seed}'
save_path_cluster = f'{save_path_main}/n_clusters{n_clusters}_good'
# --- Load Final Phase ---
labels_final_10 = np.load(save_path_cluster + '/labels_final.npy')
min_distances_final_10 = np.load(save_path_cluster + '/min_distances_final.npy')
BEAMS_10 = np.load(save_path_cluster + '/BEAMS.npy', allow_pickle=True)
TIMES_10 = np.load(save_path_cluster + '/TIMES.npy', allow_pickle=True)
subset_idxs_10 = np.load(save_path_cluster + '/subset_idxs.npy', allow_pickle=True)
# --- Load initial phase ---
labels_init_10=np.load(save_path_cluster + '/labels_init.npy')
min_distances_init_10=np.load(save_path_cluster + '/min_distances_init.npy')
shuffle_idxs=np.load(save_path_cluster+'/shuffle_indices.npy')
# extract additional useful parameters from clustering for plotting
labels_final_list_10=np.unique(labels_final_10)   # contains clusters as int
event_names_final_10 = [
    f"cl. {i+1}" if i != -1 else "outlier"
    for i in labels_final_list_10
]
event_names_final_10[-1]='cl. 4*'

idx_medoids_10=np.argwhere(min_distances_final_10==0)   # idxs of medoids for every cluster

# set parameters for IB-DTWC
sakoe_chiba_radius_time=3  # SC-radius [s]
n_clusters=4
init_sample_size=350
n_min_cluster=3
n_max_cluster=10
subset=None         # for entire data considered, useful for debugging
sampling_freq=1/dt
n_seed=42
sigma_factor=2      # strickness for clustering

# load results of SC-radius of 5 s -> addition of _5 in variable name
save_path_main = f'../data/labeled_clustering/clustering_results/scRadi{sakoe_chiba_radius_time}_seed{n_seed}'
save_path_cluster = f'{save_path_main}/n_clusters{n_clusters}_good'
# --- Load Final Phase ---
labels_final_5 = np.load(save_path_cluster + '/labels_final.npy')
min_distances_final_5 = np.load(save_path_cluster + '/min_distances_final.npy')
BEAMS_5 = np.load(save_path_cluster + '/BEAMS.npy', allow_pickle=True)
TIMES_5 = np.load(save_path_cluster + '/TIMES.npy', allow_pickle=True)
subset_idxs_5 = np.load(save_path_cluster + '/subset_idxs.npy', allow_pickle=True)

labels_init_5=np.load(save_path_cluster + '/labels_init.npy')
min_distances_init_5=np.load(save_path_cluster + '/min_distances_init.npy')
# extract additional useful parameters from clustering for plotting
labels_final_list_5=np.unique(labels_final_5)   # contains clusters as int
event_names_final_5 = [
    f"cl. {i+1}" if i != -1 else "outlier"
    for i in labels_final_list_5
]
event_names_final_5[-1]='cl. 6*'
event_names_final_5[-2]='cl. 5*'
idx_medoids_5=np.argwhere(min_distances_final_5==0)   # idxs of medoids for every cluster
#%% PREPARE PLOTTING

# prepare colormap to display clusters
def segmented_colormap(names, cmap_name='magma', start=0.0, end=1.0):
    """
    names : list der Kategorien
    cmap_name : matplotlib-colormap
    start,end : Bereich (0..1) der Colormap
    """
    cmap = plt.get_cmap(cmap_name)
    positions = np.linspace(start, end, len(names))
    colors = cmap(positions)
    return {name: colors[i] for i, name in enumerate(names)}



color_map_true=segmented_colormap(event_names,  cmap_name='Blues', start=0.30, end=1)
color_map_pred_5=segmented_colormap(labels_final_list_5, cmap_name='Oranges', start=0.2, end=1)
color_map_pred_10=segmented_colormap(labels_final_list_10, cmap_name='Purples', start=0.5, end=1.00)


# sort events by medoids
            
event_medoid_map_5 = {}

for k in idx_medoids_5:
    k = int(k[0])  # flatten

    cluster = labels_final_5[k]   # ← richtiges Label verwenden!

    if cluster == -1:
        continue  # Outlier hat keinen Medoid

    event_medoid_map_5[int(cluster)] = k
    
event_medoid_map_10 = {}

for k in idx_medoids_10:
    k = int(k[0])

    cluster = labels_final_10[k]

    if cluster == -1:
        continue

    event_medoid_map_10[int(cluster)] = k
    
from matplotlib.patches import ConnectionPatch

def pie_with_leader_lines(ax, sizes, colors,line_vis,labels_pie_input, label_positions=None):
    print('new plot...')
    # default: automatically position labels in a circle
    if label_positions is None:
        # equidistant angles for labels
        angles = np.linspace(0, 2*np.pi, len(sizes), endpoint=False)
        label_positions = [(1.4*np.cos(a), 1.4*np.sin(a)) for a in angles]

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor='white')
    )
    
    total = sum(sizes)

    for i,(w,lab,color) in enumerate(zip(wedges,labels_pie,colors)):
        
        
        # Winkel des Mittelpunkts des Segments
        # make sure that no wedge starts at =0°
        ang = (w.theta2 + w.theta1) / 2
        ang = ang % 360
        if abs(w.theta2 - w.theta1) < 1e-3:
            ang += 0.01
        
        rad = np.deg2rad(ang)

        # Punkt am äußeren Rand des Segments
        x_r = np.cos(rad)
        y_r = np.sin(rad)

        # gewünschte Label-Position
        x_t, y_t = label_positions[i]

        text_color=w.get_facecolor()

        # Zeichne Text
        ax.text(
            x_t,
            y_t,
            f"{sizes[i]} \n {100*sizes[i]/total:.1f}%",
            ha='center',
            va='center',
            fontsize=9,
            color='black'
            # color=text_color,
            # bbox=dict(
            #     boxstyle="round,pad=0.3",  # runde Ecken + Innenabstand
            #     facecolor="white",          # Hintergrundfarbe
            #     edgecolor='white',       # Rahmenfarbe passend zum Text
            #     linewidth=0.8
            # )
        )

        # Zeichne Linie (Leader line)
        if line_vis[i]:

            # Berechne Leader-Line
            ang = (w.theta2 + w.theta1)/2
            x_r = np.cos(np.deg2rad(ang))
            y_r = np.sin(np.deg2rad(ang))
    
            # Label-Position
            if label_positions and i < len(label_positions):
                x_t, y_t = label_positions[i]
            else:
                x_t, y_t = 1.2*x_r, 1.2*y_r  # fallback
    
            # Debug
            print("used for ", i, " of ", w)
            print(x_r, y_r, x_t, y_t)

            con = ConnectionPatch(
                xyA=(x_r, y_r),     # Punkt am Segment
                xyB=(x_t, y_t),     # Punkt beim Label
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,
                arrowstyle="-",     # einfache Linie
                lw=0.8,
                color='black'
            )
            #con.set_zorder(10)
            ax.add_artist(con)

    ax.axis('equal')
#%% PLOT THE DATA
if poster_fig==True:
    fig = plt.figure(figsize=(8,8),dpi=300)
    gs=gridspec.GridSpec(7,2,height_ratios=[2,2,2,2,2,2,2],hspace=0)
else:
    fig = plt.figure(figsize=(18*cm_to_inch,20*cm_to_inch),dpi=300)

    gs=gridspec.GridSpec(9,3,height_ratios=[2,2,2,2,2,2,2,1,5],hspace=0)


# choose examples of every category of labeled data type to display event characteristics
IDX_PLOT=[0,0,0,0,120,1]



# settings for pie of true labels
label_positions_true = [
    (-0.5,0.3),
    (-0.7,-0.5),
    (-0.2,-0.8),
    (0.2,-1.1),
    (0.5,0),
    (0,1.3)]

line_on_true=[False,False,False,True,False,True]

# settings for pie of pred labels with smaller SCR
label_positions_pred_5 = [
    (-0.7,1),
    (-0.5,-0.1),
    (0.5,-0.5),
    (1.3,-0.3),
    (0.7,0.3),
    (0.7,1.1),
    (0,1.3)]

line_on_pred_5=[True,False,False,True,False,True,True]

# settings for pie of pred labels with bigger SCR
label_positions_pred_10 = [
    (-0.4,1.2),
    (-0.5,-0.1),
    (0.5,-0.4),
    (0.4,0.5),
    (0.4,1.2)]

line_on_pred_10=[True,False,False,False,True]

axes_medoids_true=[]
axes_medoids_pred=[]

#%%
# plot examples of labeled data
for i, event in enumerate(event_names):
    if i == 0:
        ax = fig.add_subplot(gs[i, 0])
    else:
        ax = fig.add_subplot(gs[i, 0], sharex=axes_medoids_true[0])

    if IDX_PLOT is not None:
        idxs=[idx for idx,label in enumerate(labels_list) if label==event]
        idx_to_plot=idxs[IDX_PLOT[i]]
    else:
        idxs=[idx for idx,label in enumerate(labels_list) if label==event]
        idx_to_plot=idxs[0]

    # ensure flat data
    t = np.ravel(times_list[idx_to_plot])
    sig = np.ravel(data_list[idx_to_plot])

    ax.plot(t, sig, color=color_map_true.get(event),linewidth=0.5)
    ax.set_yticks([])

    # display cluster / event type name
    ax.text(
        0.97, 0.95, event_names_label[i],
        transform=ax.transAxes, ha='right', va='top'
        # bbox=dict(
        #     boxstyle="round,pad=0.3",  # runde Ecken + Innenabstand
        #     facecolor="white",          # Hintergrundfarbe
        #     edgecolor='white',       # Rahmenfarbe passend zum Text
        #     linewidth=0
        # )
    )
    

    if i == len(event_names) - 1:
        ax.set_xlabel("time [s]")
    else:
        ax.tick_params(labelbottom=False)        
        
    if i == 3:        # this is the longest event. Fix therefore no determinition needed
        ax.set_xlim(t[0],t[-1])
        
    if poster_fig == True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        
    axes_medoids_true.append(ax)
    
    # while plotting medoids, count samples for pie chart
    counts = Counter(np.ravel(labels_list))
    labels_pie = list(counts.keys())
    sizes_pie = list(counts.values())
    pie_colors = [color_map_true.get(lab) for lab in labels_pie]

    

if poster_fig == False:
    axes_dist_true = fig.add_subplot(gs[8, 0])
    
    pie_with_leader_lines(axes_dist_true,sizes=sizes_pie,
                          colors=pie_colors,line_vis=line_on_true,labels_pie_input=labels_pie,
                          label_positions=label_positions_true)
    
            
    
    axes_dist_true.axis('equal')
    axes_dist_true.set_position([
        axes_dist_true.get_position().x0,
        axes_dist_true.get_position().y0 - 0.05,   # Pie etwas nach oben schieben
        axes_dist_true.get_position().width,
        axes_dist_true.get_position().height
    ])

#%%
# plot medoids of clustering results for SC=5
if poster_fig == False:
    for i, event in enumerate(labels_final_list_5):     
        if i == 0:
            ax = fig.add_subplot(gs[i, 1])
        else:
            ax = fig.add_subplot(gs[i, 1], sharex=axes_medoids_pred[0]) 
        
        if event == -1:
            idx_to_plot = np.argwhere(labels_final_5 == -1)[0][0]
        else:
            idx_to_plot = event_medoid_map_5[event]
    
        # Daten flach machen
        t = np.ravel(times_list[idx_to_plot])
        sig = np.ravel(data_list[idx_to_plot])
        
        ax.plot(t, sig, color=color_map_pred_5.get(event),linewidth=0.5)
        ax.set_yticks([])
        
        # Titel
        ax.text(
            0.97, 0.95, event_names_final_5[i],
            transform=ax.transAxes, ha='right', va='top'
        )
        
        
        if i == len(labels_final_list_5)-1 :
            ax.set_xlabel("time [s]")
        else:
            ax.tick_params(labelbottom=False)
            
        ax.set_xlim(0,120)
    
        axes_medoids_pred.append(ax)
    
    
    counts = Counter(np.ravel(labels_final_5))
    # sort clusters (Outlier last)
    labels_pie = sorted(counts.keys(), key=lambda x: (x != -1, x))
    sizes_pie = [counts[lab] for lab in labels_pie]
    pie_colors = [color_map_pred_5[lab] for lab in labels_pie]
    
    

    # add pie chart of cluster distributions
    axes_dist_pred = fig.add_subplot(gs[8,1])
    
    pie_with_leader_lines(axes_dist_pred,sizes=sizes_pie,colors=pie_colors,
                          line_vis=line_on_pred_5,labels_pie_input=labels_pie,
                          label_positions=label_positions_pred_5)
    
           
    axes_dist_pred.axis('equal')
    axes_dist_pred.set_position([
        axes_dist_pred.get_position().x0,
        axes_dist_pred.get_position().y0 - 0.05,   # Pie etwas nach oben schieben
        axes_dist_pred.get_position().width,
        axes_dist_pred.get_position().height
    ])    
 #%%       
# plot medoids of clustering results for SC=10

for i, event in enumerate(labels_final_list_10):    
    if poster_fig == False:
        if i == 0:
            ax = fig.add_subplot(gs[i, 2])
        else:
            ax = fig.add_subplot(gs[i, 2], sharex=axes_medoids_pred[0]) 
    else:
        if i == 0:
            ax = fig.add_subplot(gs[i, 1])
        else:
            ax = fig.add_subplot(gs[i, 1], sharex=axes_medoids_pred[0])        
    
    if event == -1:
        idx_to_plot = np.argwhere(labels_final_10 == -1)[0][0]
    else:
        idx_to_plot = event_medoid_map_10[event]

    # Daten flach machen
    t = np.ravel(times_list[idx_to_plot])
    sig = np.ravel(data_list[idx_to_plot])
    
    ax.plot(t, sig, color=color_map_pred_10.get(event),linewidth=0.5)
    ax.set_yticks([])
    
    # Titel
    ax.text(
        0.97, 0.95, event_names_final_10[i],
        transform=ax.transAxes, ha='right', va='top'
    )
    
    
    if i == len(labels_final_list_10)-1 :
        ax.set_xlabel("time [s]")
    else:
        ax.tick_params(labelbottom=False)
        
    ax.set_xlim(0,120)
    
    if poster_fig == True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes_medoids_pred.append(ax)
    
if poster_fig == False:
    # skip plotting pie chart for poster
    counts = Counter(np.ravel(labels_final_10))
    # sort clusters (Outlier last)
    labels_pie = sorted(counts.keys(), key=lambda x: (x != -1, x))
    sizes_pie = [counts[lab] for lab in labels_pie]
    pie_colors = [color_map_pred_10[lab] for lab in labels_pie]
        
        
    # add pie chart of cluster distributions
    axes_dist_pred = fig.add_subplot(gs[8,2])
    
    
    pie_with_leader_lines(axes_dist_pred,sizes=sizes_pie,colors=pie_colors,
                          line_vis=line_on_pred_10,labels_pie_input=labels_pie
                          ,label_positions=label_positions_pred_10)
    
    axes_dist_pred.axis('equal')
    axes_dist_pred.set_position([
        axes_dist_pred.get_position().x0,
        axes_dist_pred.get_position().y0 - 0.05,   # Pie etwas nach oben schieben
        axes_dist_pred.get_position().width,
        axes_dist_pred.get_position().height
    ])



if poster_fig ==True:
    plt.savefig('../../Presentations/DGG2026/figures/plot_labeled_pred_comp.png', transparent=True)
else:
    plt.savefig('../../papers/figures/plot_labeled_pred_comp.pdf',dpi=300,bbox_inches='tight')
plt.show()



#%% BOTH CONF MATR IN ONE FIGURE
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------
# 1) True-Labels in Zahlen umwandeln
# ---------------------------------------------------------
y_true_str = [labels_list[idx] for idx in subset_idxs_5]
mapping = {name: i for i, name in enumerate(event_names)}
y_true = np.array([mapping[s] for s in y_true_str])

# nur die true Klassen: 0 .. len(event_names)-1
true_labels = np.unique(y_true)     

# predicted labels (Cluster IDs)
pred_labels_5, pred_counts_5 = np.unique(labels_final_5, return_counts=True)
count_map_5 = dict(zip(pred_labels_5, pred_counts_5))

# Confusion Matrix (normalize per prediction)
M_conf_5 = confusion_matrix(
    y_true,
    labels_final_5,
    normalize='pred'
)
M_conf_total_5=confusion_matrix(
    y_true,
    labels_final_5
)
 

# predicted labels (Cluster IDs)
pred_labels_10, pred_counts_10 = np.unique(labels_final_10, return_counts=True)
count_map_10 = dict(zip(pred_labels_10, pred_counts_10))

# Confusion Matrix (normalize per prediction)
M_conf_10 = confusion_matrix(
    y_true,
    labels_final_10,
    normalize='pred'
)
M_conf_total_10=confusion_matrix(
    y_true,
    labels_final_10
)

# cut empty rows and columns
M_conf_5=M_conf_5[1:, :]
M_conf_total_5=M_conf_total_5[1:,:]
M_conf_10=M_conf_10[1:, :-2]
M_conf_total_10=M_conf_total_10[1:,:-2]

# ---------------------------------------------------------
# 2) Plot
# ---------------------------------------------------------



fig = plt.figure(figsize=(18*cm_to_inch, 10*cm_to_inch),dpi=300)
#import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 2, width_ratios=[M_conf_5.shape[1], M_conf_10.shape[1]],wspace=0.5)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# beide Matrizen bekommen dieselbe Zellengeometrie
extent1 = [0, M_conf_5.shape[1], 0, M_conf_5.shape[0]]
extent2 = [0, M_conf_10.shape[1], 0, M_conf_10.shape[0]]

#im1 = ax1.imshow(M1, cmap="Oranges", vmin=0, vmax=1, extent=extent1, origin="upper")
#im2 = ax2.imshow(M2, cmap="Oranges", vmin=0, vmax=1, extent=extent2, origin="upper")

# Zellen quadratisch
ax1.set_aspect("equal")
ax2.set_aspect("equal")

heat = ax1.imshow(M_conf_5, cmap="Oranges", origin="upper",vmin=0,vmax=1, extent=extent1)
# cbar = plt.colorbar(heat, ax=ax, shrink=0.7)
# cbar.set_label("Normalized per predicted label [%]")
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax1)
cax = divider.append_axes("top", size="5%", pad=0.1)  # pad = Abstand zur Achse

cbar = fig.colorbar(heat, cax=cax, orientation="horizontal")
cbar.set_label("Normalised detection \n per predicted label [%]", labelpad=5)
cbar.ax.xaxis.set_label_position("top")       # Label nach oben
cbar.ax.xaxis.set_ticks_position("top")       # optional: Ticks oben anzeigen

#ax1.set_xlim(-0.5,6.5)
# ---------------------------------------------------------
# 3) Tick labels
# ---------------------------------------------------------
ax1.set_xticks(np.arange(len(pred_labels_5))+0.5)
ax1.set_yticks(np.arange(len(true_labels))+0.5)

cluster_names=[f'Outlier \n ({count_map_10[pred_labels_5[0]]})']+[f"Cluster {l+1} \n ({count_map_5[l]})" for l in pred_labels_5[1:5]]+[f"Cluster {l+1}* \n ({count_map_5[l]})" for l in pred_labels_5[5:]]

ax1.set_xticklabels(cluster_names, rotation=90, ha='center',va='top')
ax1.set_yticklabels([event_names_label[l] for l in reversed(true_labels)])

ax1.set_xlabel("Predicted labels")
ax1.set_ylabel("True labels")
# ---------------------------------------------------------
# 4) Text overlay
# ---------------------------------------------------------
for i in range(M_conf_5.shape[0]):
    for j in range(len(np.unique(labels_final_5))):
        val = M_conf_total_5[i, j]
        ax1.text(
            j+0.5,
            M_conf_5.shape[0]-i-0.5,
            f"{val}",
            ha="center",
            va="center",
            color="white" if abs(M_conf_5[i,j]) > 0.6 else "black",
            fontsize=7
            
        )
        
#ax1.set_ylim(0.5,6.5)
#ax1.invert_yaxis()

heat = ax2.imshow(M_conf_10, cmap="Purples", origin="upper",vmin=0,vmax=1, extent=extent2)
# cbar = plt.colorbar(heat, ax=ax, shrink=0.7)
# cbar.set_label("Normalized per predicted label [%]")
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax2)
cax = divider.append_axes("top", size="5%", pad=0.1)  # pad = Abstand zur Achse

cbar = fig.colorbar(heat, cax=cax, orientation="horizontal")
cbar.set_label("Normalised detection \n per predicted label [%]", labelpad=5)
cbar.ax.xaxis.set_label_position("top")       # Label nach oben
cbar.ax.xaxis.set_ticks_position("top")       # optional: Ticks oben anzeigen

#ax2.set_xlim(-0.5,4.5)
# ---------------------------------------------------------
# 3) Tick labels
# ---------------------------------------------------------
ax2.set_xticks(np.arange(len(pred_labels_10))+0.5)
ax2.set_yticks(np.arange(len(true_labels))+0.5)

cluster_names=[f'Outlier \n ({count_map_10[pred_labels_10[0]]})']+[f"Cluster {l+1} \n ({count_map_10[l]})" for l in pred_labels_10[1:4]]+[f"Cluster {l+1}* \n ({count_map_10[l]})" for l in pred_labels_10[4:]]

ax2.set_xticklabels(cluster_names, rotation=90, ha='center',va='top')
ax2.set_yticklabels([event_names_label[l] for l in reversed(true_labels)])

ax2.set_xlabel("Predicted labels")
ax2.set_ylabel("True labels")


# ---------------------------------------------------------
# 4) Text overlay
# ---------------------------------------------------------
for i in range(M_conf_10.shape[0]):
    for j in range(len(np.unique(labels_final_10))):
        val = M_conf_total_10[i, j]
        ax2.text(
            j+0.5, 
            M_conf_10.shape[0]-i-0.5,
            f"{val}",
            ha="center",
            va="center",
            color="white" if abs(M_conf_10[i,j]) > 0.6 else "black",
            fontsize=7
            
        )
        
#ax2.set_ylim(0.5,6.5)
#ax2.invert_yaxis()
plt.savefig('../../papers/figures/M_conf_comparison.pdf',dpi=300,bbox_inches='tight')
plt.show()

#%% check medoids before and after incremental step

# determine medoids of inital clusters with SC=10
idx_medoids_init_10=[idx[0] for idx in np.argwhere(min_distances_init_10==0)]
idx_medoids_init_10_shuffled=[shuffle_idxs[idx] for idx in idx_medoids_init_10]


fig, ax = plt.subplots(len(idx_medoids_10),2,dpi=300,sharex=True)

# inital phase
for row, medoid in enumerate(sorted(idx_medoids_init_10, key=lambda x: labels_init_10[x])):
    samp = shuffle_idxs[medoid]  # richtige Shuffle-Verbindung
    ax[row][0].plot(times_list[samp], data_list[samp], linewidth=0.4)
    ax[row][0].set_title(labels_init_10[medoid])
    
# final phase
for row, medoid in enumerate(sorted(idx_medoids_10, key=lambda x: labels_final_10[x])):
    samp = medoid  # richtige Shuffle-Verbindung
    ax[row][1].plot(times_list[samp[0]],data_list[samp[0]],linewidth=0.4)
    ax[row][1].set_title(labels_final_10[samp[0]])

plt.show()

# determine medoids of inital clusters with SC=10
idx_medoids_init_5=[idx[0] for idx in np.argwhere(min_distances_init_5==0)]
idx_medoids_init_5_shuffled=[shuffle_idxs[idx] for idx in idx_medoids_init_5]

fig, ax = plt.subplots(len(idx_medoids_5),2,dpi=300,sharex=True)

# inital phase
for row, medoid in enumerate(sorted(idx_medoids_init_5, key=lambda x: labels_init_5[x])):
    samp = shuffle_idxs[medoid]  # richtige Shuffle-Verbindung
    ax[row][0].plot(times_list[samp], data_list[samp], linewidth=0.4)
    ax[row][0].set_title(labels_init_5[medoid])
    
# final phase
for row, medoid in enumerate(sorted(idx_medoids_5, key=lambda x: labels_final_5[x])):
    samp = medoid  # richtige Shuffle-Verbindung
    ax[row][1].plot(times_list[samp[0]],data_list[samp[0]],linewidth=0.4)
    ax[row][1].set_title(labels_final_5[samp[0]])

plt.show()

# plot all medoids with transparent background for schematic examples
#%% plot single traces with transparent backgorund


for row, medoid in enumerate(sorted(idx_medoids_5, key=lambda x: labels_final_5[x])):
    samp = medoid  # richtige Shuffle-Verbindung
    fig,ax=plt.subplots(1,1,figsize=(6,2))
    ax.plot(times_list[samp[0]],data_list[samp[0]],linewidth=1,color='#505050')
    #ax.set_title(labels_final_5[samp[0]])

    # Hintergrund der Achse transparent
    ax.set_facecolor("none")

    # Achsen unsichtbar machen
    ax.axis("off")  # entfernt Ticks, Rahmen, Labels komplett
    
    fig.patch.set_alpha(0)
    plt.savefig(f'../../Presentations/DGG2026/figures/trace_{row}.png',transparent=True,dpi=300)
    
    np.save(f'../../data/labeled_clustering/medoids/medoid_{row}.npy',data_list[samp[0]])
    np.save(f'../../data/labeled_clustering/medoids/time_{row}.npy',times_list[samp[0]])

#%% HIST of min distances
fig, ax = plt.subplots(1, 1,dpi=300,figsize=(5,4))

ax.hist(min_distances_final_5, bins=30,
        color='white', edgecolor='#2F2F2F',
        hatch='////', linewidth=0.7,alpha=0.8)

# Kennzahlen
mu = np.mean(min_distances_final_5)
sigma = np.std(min_distances_final_5)
threshold = mu + 2 * sigma

# Vertikale Linien mit deinen Farben
ax.axvline(mu, color='#2F5D8C', linestyle='--', linewidth=2)
ax.text(mu+0.01*mu, ax.get_ylim()[1] * 0.9, r'$\mu$', color='#2F5D8C',
        rotation=0, va='top', ha='left')

ax.axvline(threshold, color='#2F5D8C', linestyle='--', linewidth=2)
ax.text(threshold+0.01*threshold, ax.get_ylim()[1] * 0.9, r'$\tau = \mu + a * \sigma$', color='#2F5D8C',
        rotation=0, va='top', ha='left')

ax.set_xlim(0.003,0.013)
ax.set_xlabel('sample-medoid DTW distance')
ax.set_ylabel('density')

# ax.tick_params(axis='both', which='both',
#                bottom=False, top=False,
#                left=False, right=False,
#                labelbottom=False, labelleft=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(False)

plt.savefig(f'../../Presentations/DGG2026/figures/hist_min_dist.png',transparent=True)


#%% ONLY CONF MATRIX of second setting for poster


fig, ax2 = plt.subplots(1,1,figsize=(4.5, 6.5),dpi=300)

# beide Matrizen bekommen dieselbe Zellengeometrie
#extent1 = [0, M_conf_5.shape[1], 0, M_conf_5.shape[0]]
extent2 = [0, M_conf_10.shape[1], 0, M_conf_10.shape[0]]

#im1 = ax1.imshow(M1, cmap="Oranges", vmin=0, vmax=1, extent=extent1, origin="upper")
#im2 = ax2.imshow(M2, cmap="Oranges", vmin=0, vmax=1, extent=extent2, origin="upper")

# Zellen quadratisch
#ax1.set_aspect("equal")
ax2.set_aspect("equal")

        
#ax1.set_ylim(0.5,6.5)
#ax1.invert_yaxis()

heat = ax2.imshow(M_conf_10, cmap="Purples", origin="upper",vmin=0,vmax=1, extent=extent2)
# cbar = plt.colorbar(heat, ax=ax, shrink=0.7)
# cbar.set_label("Normalized per predicted label [%]")
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax2)
cax = divider.append_axes("top", size="5%", pad=0.1)  # pad = Abstand zur Achse

cbar = fig.colorbar(heat, cax=cax, orientation="horizontal")
cbar.set_label("Normalised detection \n per predicted label [%]", labelpad=5)
cbar.ax.xaxis.set_label_position("top")       # Label nach oben
cbar.ax.xaxis.set_ticks_position("top")       # optional: Ticks oben anzeigen

#ax2.set_xlim(-0.5,4.5)
# ---------------------------------------------------------
# 3) Tick labels
# ---------------------------------------------------------
ax2.set_xticks(np.arange(len(pred_labels_10))+0.5)
ax2.set_yticks(np.arange(len(true_labels))+0.5)

cluster_names=[f'Outlier \n ({count_map_10[pred_labels_10[0]]})']+[f"Cluster {l+1} \n ({count_map_10[l]})" for l in pred_labels_10[1:]]

ax2.set_xticklabels(cluster_names, rotation=90, ha='center',va='top')
ax2.set_yticklabels([event_names_label[l] for l in reversed(true_labels)])

ax2.set_xlabel("Predicted labels")
ax2.set_ylabel("True labels")

ax2.grid(False)


# ---------------------------------------------------------
# 4) Text overlay
# ---------------------------------------------------------
for i in range(M_conf_10.shape[0]):
    for j in range(len(np.unique(labels_final_10))):
        val = M_conf_total_10[i, j]
        ax2.text(
            j+0.5, 
            M_conf_10.shape[0]-i-0.5,
            f"{val}",
            ha="center",
            va="center",
            color="white" if abs(M_conf_10[i,j]) > 0.6 else "#1E3A5F",
            fontsize=20
            
        )
        
#ax2.set_ylim(0.5,6.5)
#ax2.invert_yaxis()
plt.savefig('../../Presentations/DGG2026/figures/M_conf.png',dpi=300,bbox_inches='tight',transparent=True)
plt.show()



        