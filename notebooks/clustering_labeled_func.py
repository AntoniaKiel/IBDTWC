# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 10:03:22 2025

@author: anton
"""

from scipy.signal import butter, filtfilt, decimate
import os
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.gridspec import GridSpec
import sys
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.clustering.dtw_utils_update import  IncrementalFDTW
import pickle
import pywt

def bandpass_filter(data, dt, fmin, fmax, order=4):
    fs = 1 / dt
    nyq = 0.5 * fs                   # Nyquist-Frequenz
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)  # vorwärts/rückwärts filtern → keine Phasenverschiebung
    
    # data : dein Signal
    # q    : Downsampling-Faktor (z.B. 2 → halb so viele Samples)

    return filtered

def read_eventtype(events, event_type_idx,load_path_list,event_names,station_name):
    data_dir = load_path_list[event_type_idx]
    ldir = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    str_event = event_names[event_type_idx]

    # Stelle sicher, dass wir eine Liste verwenden
    events[str_event] = []

    component = "Z"

    for dir_event in ldir:

        f = os.path.join(data_dir, dir_event, f'{station_name}_{component}.ASC')
        meta = open(f).readlines()[0:5]

        dt = float(meta[0][6:-1])
        data = bandpass_filter(np.loadtxt(f, skiprows=5), dt, 1, 30)

        # Skip invalid
        if np.all(data == 0) or np.any(np.isnan(data)):
            print(f"Skipping: {dir_event} invalid Z")
            continue

        # Neues Event sauber anhängen
        event = {
            'Z': data,
            'dt': dt,
            'n_samp': int(meta[1][7:-1]),
            'times': np.arange(int(meta[1][7:-1])) * dt,
            'event_index': len(events[str_event])   # neuer sauberer Index
        }

        events[str_event].append(event)
        print(f"{str_event}: event {len(events[str_event])} added!")

def plot_events_and_pie(
    times_list,
    data_list,
    freq_min,
    freq_max,
    labels_list,
    event_names,
    event_names_label=None,
    idx_medoids=None,
    percentage_plot_list=None,
    IDX_PLOT=None,
    colormap="tab20b",
    figsize=(9, 5),
    dpi=200):
    """
    Plot-Funktion:
    Links: pro Eventtyp ein Beispiel-Zeitverlauf (Medoid oder erstes Vorkommen)
    Rechts: Piechart über die Häufigkeiten der Event-Labels.
    """
    waveletname='cmor1-5'
    sampling_rate=1/(times_list[0][1]-times_list[0][0])
    freqs_desired = np.linspace(freq_min, freq_max, 50)
    scales = pywt.central_frequency(waveletname) / (freqs_desired / sampling_rate)
    
    # -------------------------
    # Outlier berücksichtigen (-1)
    # -------------------------
    labels_set = set(np.ravel(labels_list))
    event_names = list(event_names)  # sicherstellen, dass mutable

    if -1 in labels_set and (-1 not in event_names):
        event_names.append(-1)
        if event_names_label is not None:
            event_names_label = list(event_names_label) + ["Outlier"]
        else:
            event_names_label = [str(e) for e in event_names[:-1]] + ["Outlier"]

    if event_names_label is None:
        event_names_label = [str(e) for e in event_names]

    label_display_map = dict(zip(event_names, event_names_label))

    # -------------------------
    # Colormap vorbereiten
    # -------------------------
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(event_names)))
    color_map = {event: colors[i] for i, event in enumerate(event_names)}
    color_map[-1] = 'lightgray'  # Outlier immer grau

    # -------------------------
    # Medoids nach Event sortieren
    # -------------------------
    # Medoids nach Event sortieren, nur gültige Indizes nehmen
    event_medoid_map = {}
    if idx_medoids is not None:
        for k in idx_medoids:
            if k >= len(labels_list):
                continue  # Index außerhalb von labels_list ignorieren
            lab = int(np.ravel(labels_list[k])[0])
            if lab not in event_medoid_map:
                event_medoid_map[lab] = k

    # -------------------------
    # Figure + Grid
    # -------------------------
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(len(event_names), 3, width_ratios=[2,2,2], wspace=0.2)

    axes_events = []
    axes_cwt = []
    for i in range(len(event_names)):
        if i == 0:
            ax = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
        else:
            ax = fig.add_subplot(gs[i, 0], sharex=axes_events[0])
            ax2 = fig.add_subplot(gs[i, 1], sharex=axes_cwt[0])
        axes_events.append(ax)
        axes_cwt.append(ax2)

    # -------------------------
    # LINKER TEIL: Medoid oder erstes Event
    # -------------------------
    
    # 1) Map: label -> idx in idx_medoids
    if idx_medoids is not None:
        labels_for_medoids=np.array([labels_list[i][0] for i in idx_medoids])
        idx_medoids=np.array(idx_medoids)
        
        # 1) sortiere nach labels_for_medoids
        sort_idx = np.argsort(labels_for_medoids)  # Index-Array, das labels sortiert
        

        # 2) wende auf beide Arrays an
        labels_for_medoids_sorted = labels_for_medoids[sort_idx]
        idx_medoids = idx_medoids[sort_idx]
  
    
    
    
    for i, event in enumerate(event_names):
        ax = axes_events[i]
        
        if idx_medoids is not None and event != -1:
            #print(labels_for_medoids,'labels_for_medoids')
            idx_to_plot=idx_medoids[i-1][0]
            #print(idx_to_plot,'idx_to_plot',type(idx_to_plot))
        elif idx_medoids is not None:
            idx_to_plot=np.argwhere(labels_list==event)[0][0]
        elif IDX_PLOT is not None:
            idxs=[idx for idx,label in enumerate(labels_list) if label==event]
            idx_to_plot=idxs[IDX_PLOT[i]]
        else:
            idxs=[idx for idx,label in enumerate(labels_list) if label==event]
            idx_to_plot=idxs[0]
        print(idx_to_plot,'idx_to_plot for event',event)
        

    
        # Daten flach machen
        t = np.ravel(times_list[idx_to_plot])
        sig = np.ravel(data_list[idx_to_plot])
    
        ax.plot(t, sig, color=color_map.get(event, 'gray'))
        ax.set_yticks([])
    
        # Titel
        ax.text(
            0.5, 0.95, label_display_map[event],
            transform=ax.transAxes, ha='center', va='top'
        )
    
        if i == len(event_names) - 1:
            ax.set_xlabel("time [s]")
        elif i == 3:
            ax.set_xlim(t[0],t[-1])
        else:
            ax.tick_params(labelbottom=False)
            
        ax = axes_cwt[i]
        
        # Calculate scales for CWT once
        scales_min = pywt.scale2frequency(waveletname, freq_max) * sampling_rate
        scales_max = pywt.scale2frequency(waveletname, freq_min) * sampling_rate
        scales = np.arange(scales_min, scales_max, 0.2)
        coeffs, freqs = pywt.cwt(sig, scales, waveletname, sampling_period=1/sampling_rate)

        coeffs_log = np.log10(np.abs(coeffs)+1e-12)
        ax.pcolormesh(t, freqs, coeffs_log, shading='gouraud', cmap='magma',
                           vmin=np.percentile(coeffs_log,3), vmax=np.percentile(coeffs_log,99))
        ax.set_ylim(freq_min,freq_max)
        ax.set_xlabel('time [s]')
        if i ==3:
            ax.set_ylabel('frequency [Hz]')

    # -------------------------
    # RECHTER TEIL: Piechart
    # -------------------------
    ax_pie = fig.add_subplot(gs[:, 2])
    counts = Counter(np.ravel(labels_list))
    labels_pie = list(counts.keys())
    sizes_pie = list(counts.values())
    pie_colors = [color_map.get(lab, 'gray') for lab in labels_pie]

    wedges, _ = ax_pie.pie(
        sizes_pie,
        startangle=90,
        colors=pie_colors,
        wedgeprops={'edgecolor': 'white'}
    )

    # Labels + Prozentwerte
    total = sum(sizes_pie)
    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        if not percentage_plot_list:
            ax_pie.text(
                1.2 * x,
                1.2 * y,
                f"{label_display_map.get(labels_pie[i], labels_pie[i])}\n{sizes_pie[i]} ({100*sizes_pie[i]/total:.1f}%)",
                ha='center', va='center', fontsize=9
            )
        else:
            ax_pie.text(
                percentage_plot_list[i][0] * x,
                percentage_plot_list[i][1] * y,
                f"{label_display_map.get(labels_pie[i], labels_pie[i])}\n{sizes_pie[i]} ({100*sizes_pie[i]/total:.1f}%)",
                ha='center', va='center', fontsize=9
            )            

    ax_pie.axis('equal')
    ax_pie.set_title(f"Cluster/Event distribution ({len(labels_list)} samples)")

    plt.tight_layout()
    plt.show()
    
def plot_threshold_diagnostics_simple(model, min_distances, save_path_cluster):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import math

    clean = np.array([d for d in min_distances if d is not None and np.isfinite(d)])
    if clean.size == 0:
        print("[THRESH PLOT] No valid distances to plot.")
        return

    overall_mean = clean.mean()
    overall_std = clean.std(ddof=0)

    # compute PDF of normal distribution
    xs = np.linspace(clean.min(), clean.max(), 500)
    pdf_overall = (1.0 / (overall_std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((xs - overall_mean) / overall_std) ** 2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clean, bins=50, density=True, alpha=0.6, color='lightgray', edgecolor='black', label='init_min_distances')

    ax.plot(xs, pdf_overall, color='orange', linewidth=2, label=f'N(mean={overall_mean:.4f}, std={overall_std:.4f})')

    # vertical lines
    ax.axvline(overall_mean, color='blue', linestyle=':', linewidth=2, label='Mean')
    ax.axvline(model.threshold_init, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = mean + {getattr(model, "sigma_factor", 1.5)}·σ')

    ax.set_xlabel('DTW distance (normalized)')
    ax.set_ylabel('Density')
    ax.set_title('Threshold diagnostics (Normal PDF only)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    os.makedirs(save_path_cluster, exist_ok=True)
    savefile = os.path.join(save_path_cluster, 'threshold_diagnostics_simple.png')
    fig.savefig(savefile, dpi=200)
    plt.close(fig)
    print(f"[THRESH PLOT] Saved simplified threshold diagnostics to {savefile}")


def dtw_function(data_list,times_list,sakoe_chiba_radius_time, n_clusters, init_sample_size,sampling_freq,
                 admissibility=1.5, b_min=5,max_buffer_age=100,
                 save_data=True, recalc_init_clustering=False, subset=None, n_seed=42):
    """_summary_

    Parameters
    ----------
    data_list : list of np.arrays of shape (X,)
        list of all seismic traces. 
        Every seismic trace is a numpy-array of shape (X,) with X being the number of samples.
    times_list : list of np.arrays of shape (X,)
        list of all times related to seismic traces. The times is relative, starting with 0 [s].
        Every time window is a numpy-array of shape (X,) with X being the number of samples.
    sakoe_chiba_radius_time : int or float
        Sakoe-Chiba raidus for DTW-calculation [s]
    n_clusters : int
        number of clusteers to us for initial clustering
    init_sample_size : int
        number of samples to use for inital clustering step
    sampling_freq : int or float
        sampling frequency of the data
    admissibility : float, optional
        admissibbility of threshold for incremental step, by default 1.5
    b_min : int, optional
        minimum number of samples in buffer cluster neded for upgrade to core cluster, by default 5
    max_buffer_age : int, optional
        _description_, by default 100
    save_data : bool, optional
        _description_, by default True
    recalc_init_clustering : bool, optional
        _description_, by default False
    subset : _type_, optional
        _description_, by default None
    n_seed : int, optional
        _description_, by default 42
    """


    #% LOAD EVENT DATA
    # read file paths to seismic events with good quality signals
    BEAMS=data_list
    TIMES_UTC_start=times_list
    TIMES=times_list

        
    #%% PREPARE CLUSTERING STRUCTURE
    np.random.seed(n_seed)
    rng = np.random.default_rng(seed=n_seed)
    
    if subset:
        BEAMS_idx=rng.choice(len(BEAMS),size=subset,replace=False)
        BEAMS=[BEAMS[i] for i in BEAMS_idx]
        TIMES_UTC_start=[TIMES_UTC_start[i] for i in BEAMS_idx]
        TIMES=[TIMES[i] for i in BEAMS_idx]
        
    else:
        BEAMS_idx=np.arange(len(BEAMS))
    
    sakoe_chiba_radius_value=sakoe_chiba_radius_time * sampling_freq

    save_path_main = f'../data/labeled_clustering/clustering_results/scRadi{sakoe_chiba_radius_time}_seed{n_seed}'
    os.makedirs(save_path_main, exist_ok=True)
    save_path_cluster = f'{save_path_main}/n_clusters{n_clusters}'
    os.makedirs(save_path_cluster, exist_ok=True)

    #%%
    # =============================================================================
    # INITIAL CLUSTERING
    # =============================================================================

    print("Extracting data for inital clustering...")

    
    # --- Sample per day ---
    rng=np.random.default_rng(n_seed)
    init_indices = rng.choice(len(BEAMS),size=init_sample_size,replace=False)
    
    # --- Save cached initial clustering ---
    initial_beams = [BEAMS[i] for i in init_indices]
    initial_times=[TIMES[i] for i in init_indices]
    initial_times_UTC=[TIMES_UTC_start[i] for i in init_indices]
    
    print(len(initial_beams))


    # --- Remaining indices & beams ---
    remaining_indices = np.setdiff1d(np.arange(len(BEAMS)), init_indices)
    remaining_beams = [BEAMS[i] for i in remaining_indices]
    remaining_times = [TIMES[i] for i in remaining_indices]
    remaining_times_UTC = [TIMES_UTC_start[i] for i in remaining_indices]

    
    print("Fitting initial model...")
    
    # 1 Initialisiere ohne festen Threshold → datengetrieben!
    model = IncrementalFDTW( sakoe_chiba_radius=sakoe_chiba_radius_value,
                            sigma_factor=admissibility, seed=n_seed,
                            buffer_size=b_min,max_buffer_age=max_buffer_age)

    cache_file_model = os.path.join(save_path_main, "model_initial.pkl")
    
    if os.path.exists(cache_file_model) and (recalc_init_clustering==False):
        print("Loading cached initial model...")
        with open(cache_file_model, "rb") as f:
            model = pickle.load(f)
    else:
        # 2️Führe initiales Clustering durch
        labels_init, medoids_init, min_dists_init = model.fit_initial(initial_beams, n_clusters=n_clusters)
            
        print("init_min_dists len:", len(min_dists_init))
        print("n zeros in init_min_dists:", np.sum(np.isclose(min_dists_init, 0.0)))
        print("labels after initial", np.unique(labels_init))

        
        print("Saving initial model to cache...")
        
        with open(cache_file_model, "wb") as f:
            pickle.dump(model, f)


    
    plot_threshold_diagnostics_simple(model, min_dists_init, save_path_cluster)
    
    
    

    # =============================================================================
    # FIT INCREMENTAL FDTW
    # =============================================================================
    #min_distances=min_dists.copy()
    # Incrementally add remaining beams
    for counter, sample in enumerate(remaining_beams):
        model.add_sample(sample)

        print(f"{(counter / len(remaining_beams)) * 100:.2f}% added")
    


    print('labels before finalising:',np.unique(model.labels_add))
    print(model.threshold_add)

    # =============================================================================
    # FINAL CLUSTERING
    # =============================================================================
    
    # Finalize distances
    BEAMS_ordered = initial_beams + remaining_beams
    TIMES_ordered = initial_times + remaining_times
    TIMES_UTC_ordered= initial_times_UTC + remaining_times_UTC
    
    
    
    model.finalize_clustering()


    # =============================================================================
    # RESTORE ORIGINAL ORDER & SAVE
    # =============================================================================
    print(list(vars(model)))
    
    # --- Shuffle-Indices für Final Phase ---
    shuffle_indices = np.concatenate([init_indices, remaining_indices])
    inverse_indices = np.argsort(shuffle_indices)
    
    # --- Final Phase (zurücksortiert) ---
    labels_final_original_order = np.array(model.labels_final)[inverse_indices]
    min_distances_final_original_order = np.array(model.min_distances_final)[inverse_indices]
    all_distances_final_original_order = np.array(model.final_all_dists, dtype=object)[inverse_indices]
    
    BEAMS_original_order = [BEAMS_ordered[i] for i in inverse_indices]
    TIMES_original_order = [TIMES_ordered[i] for i in inverse_indices]
    TIMES_UTC_original_order = [TIMES_UTC_ordered[i] for i in inverse_indices]
    
    if save_data==True:
        # --- Save Final Phase ---
        np.save(save_path_cluster + '/labels_final.npy', labels_final_original_order)
        np.save(save_path_cluster + '/min_distances_final.npy', min_distances_final_original_order)
        np.save(save_path_cluster + '/all_distances_final.npy', all_distances_final_original_order)
        np.save(save_path_cluster + '/BEAMS.npy', np.array(BEAMS_original_order, dtype=object))
        np.save(save_path_cluster + '/TIMES.npy', np.array(TIMES_original_order, dtype=object))
        np.save(save_path_cluster + '/TIMES_UTC.npy', np.array(TIMES_UTC_original_order, dtype=object))
        np.save(save_path_cluster + '/shuffle_indices.npy', shuffle_indices)
        np.save(save_path_cluster + '/distance_threshold_final.npy', model.threshold_final)
        np.save(save_path_cluster + '/medoids.npy', np.array(model.medoids, dtype=object))
        np.save(save_path_cluster + '/subset_idxs.npy', BEAMS_idx)
        
        # --- Save Initial & Add Phase (original Reihenfolge) ---
        np.save(save_path_cluster + '/labels_init.npy', np.array(model.labels_init))
        np.save(save_path_cluster + '/min_distances_init.npy', np.array(model.min_distances_init))
        np.save(save_path_cluster + '/labels_add.npy', np.array(model.labels_add))
        np.save(save_path_cluster + '/min_distances_add.npy', np.array(model.min_distances_add))
        np.save(save_path_cluster + '/distance_threshold_init.npy', model.threshold_init)
        np.save(save_path_cluster + '/distance_threshold_add.npy', model.threshold_add)
        
        print("✅ Clustering results saved (Init, Add, Final).")
    

