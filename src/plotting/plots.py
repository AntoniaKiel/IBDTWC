import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
import random
import torch
import pywt

def plotting_func(load_path_catalog,
                  load_path_clustering,
                  save_plt_path,
                  sakoe_chiba_radius_time,
                  n_clusters,n_seed,
                  freq_min,freq_max,
                  sampling_rate,waveletname,n_examples_per_cluster,n_bins):


    #%% import clustering results of dtw of real data

    # 1. --- Load Catalog ---
    load_path = load_path_catalog
    df = pd.read_csv(load_path / "catalog_data.csv")
    beam_paths = df.loc[(df["beam"] == True) & (df["used"] == True), "ID"].unique().tolist()
    print(f"Number of events: {len(beam_paths)}")


    # load data from clustering
    load_path =load_path_clustering
    labels = np.load(load_path + 'labels.npy')
    min_distances = np.load(load_path + 'min_distances.npy', allow_pickle=True)
    all_distances = np.load(load_path + 'all_distances.npy', allow_pickle=True)
    BEAMS = np.load(load_path + 'BEAMS.npy', allow_pickle=True)
    TIMES = np.load(load_path + 'TIMES.npy', allow_pickle=True)
    TIMES_UTC=np.load(load_path + 'TIMES_UTC.npy', allow_pickle=True)
    distance_threshold=np.load(load_path +'/distance_threshold.npy')
    
    # create saving folder if it does not exist
    if not os.path.exists(save_plt_path):
        os.mkdir(save_plt_path)

    # save medoids separately
    IDX_medoids_all = []
    medoids_time = []
    medoids_beams = []

    for cluster_label in sorted(set(labels)):
        if cluster_label == -1:
            continue
        # find the index where this sample is the medoid
        cluster_indices = np.where(labels == cluster_label)[0]
        medoid_idx = cluster_indices[np.argmin(min_distances[cluster_indices])]
        IDX_medoids_all.append(medoid_idx)
        medoids_time.append(TIMES[medoid_idx])
        medoids_beams.append(BEAMS[medoid_idx])

    # mapping for plotting
    valid_clusters = [c for c in sorted(set(labels)) if c != -1]
    label_to_col = {label: i for i, label in enumerate(valid_clusters)}
    
    #%% PLOT MIN DISTANCES
    fig,ax=plt.subplots(1,1)
    ax.hist(min_distances,bins=100)
    ax.axvline(distance_threshold,color='red',label=f"threshold of {distance_threshold}")
    ax.legend()
    plt.savefig(save_plt_path+'/hist_min_distances.png')
    plt.show()

    #%% PLOT ONE EXAMPLE OF EVERY CLUSTER

    # --- Step 1: Count labels and define cluster info ---
    label_counts = Counter(labels)
    clusters = sorted(label_counts.keys())  # Sorted for consistency
    counts = [label_counts[label] for label in clusters]
    cluster_names = [f"Cluster {label}" for label in clusters]

    # --- Step 2: Generate consistent cluster colors ---
    n_clusters = len(clusters)
    if n_clusters <= 10:
        color_map = 'tab10'
    else:
        color_map = 'tab20'

    cmap = cm.get_cmap('tab20', n_clusters)
    cluster_colors = {}
    for label in clusters:
        if label == -1:
            cluster_colors[label] = 'black'
        else:
            if n_clusters <= 10:
                cluster_colors[label] = plt.cm.tab10(label % 10)
            else:
                cluster_colors[label] = plt.cm.tab20(label % 20)

    # --- Step 3: Pie Chart with consistent colors ---
    legend_labels = [f"{name}: {count} samples" for name, count in zip(cluster_names, counts)]
    pie_colors = [cluster_colors[label] for label in clusters]

    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        counts, labels=None, colors=pie_colors, autopct='%1.1f%%', startangle=90
    )
    plt.legend(wedges, legend_labels, title="Clusters", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Cluster Distribution (Pie Chart)")
    plt.axis('equal')  # Ensures circular shape
    plt.tight_layout()
    plt.savefig(save_plt_path+'/distribution_clusters.png')

    
    # --- Step 4: Plot signal examples with matching colors ---
    
    # Calculate scales for CWT once
    scales_min = pywt.scale2frequency(waveletname, freq_max) * sampling_rate
    scales_max = pywt.scale2frequency(waveletname, freq_min) * sampling_rate
    scales = np.arange(scales_min, scales_max, 0.2)
    
    # Map cluster label to example indices
    cluster_to_index = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_index[label].append(idx)
    
    # Alle Cluster inkl. Outlier
    clusters_all = sorted(np.unique(labels))  # z.B. [-1, 0, 1, 2, 3, 4, 5, 6]
    n_cols = len(clusters_all)
    
    n_rows_per_example = 4  # Trace, CWT, Beamforming, Distance
    n_hist_rows = 3          # Histogramme am Ende
    n_rows = n_examples_per_cluster * n_rows_per_example + n_hist_rows
    fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.5, wspace=0.3)

    
    for col, cluster_label in enumerate(clusters_all):
        color = cluster_colors.get(cluster_label, 'black')
        all_indices = cluster_to_index[cluster_label]
    
        # --- Beispiele auswählen ---
        if cluster_label != -1:  # Echte Cluster: Medoid + Zufallsbeispiele
            medoid_idx = IDX_medoids_all[label_to_col[cluster_label]]
            other_indices = [i for i in all_indices if i != medoid_idx]
            n_random = min(n_examples_per_cluster - 1, len(other_indices))
            random_examples = random.sample(other_indices, n_random)
            example_indices = [medoid_idx] + random_examples
        else:  # Outlier: nur zufällige Beispiele
            example_indices = random.sample(all_indices, min(n_examples_per_cluster, len(all_indices)))
    
        # --- Slowness-Angle für Histogramme sammeln ---
        slowness_angles = []
        slowness_mags = []
        durations = []
        for idx in all_indices:
            try:
                beam_path = beam_paths[idx]
                beam_file = os.path.join(beam_path, "BEAM_results.npy")
                beam_data = np.load(beam_file, allow_pickle=True)[0]
                slowness_best = beam_data['slowness_best']
                if isinstance(slowness_best, torch.Tensor):
                    slowness_best = slowness_best.detach().cpu().numpy()
                slowness_angles.append(np.arctan2(slowness_best[1], slowness_best[0]))
                slowness_mags.append(np.linalg.norm(slowness_best))
                durations.append(TIMES[idx].max() - TIMES[idx].min())
            except Exception as e:
                print(f"Cluster {cluster_label} histogram error idx {idx}: {e}")
    
        # --- Pro Beispiel: Trace, CWT, Beamforming, Distances ---
        for i, example_idx in enumerate(example_indices):
            try:
                row_base = i * n_rows_per_example
                beam_path = beam_paths[example_idx]
                beam_file = os.path.join(beam_path, "BEAM_results.npy")
                beam_data = np.load(beam_file, allow_pickle=True)[0]
                slowness_space = beam_data['slowness_space']
                slowness_power = beam_data['slowness power']
                slowness_best = beam_data['slowness_best']
    
                def to_np(x):
                    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
    
                signal = to_np(BEAMS[example_idx]).flatten()
                time = to_np(TIMES[example_idx]).flatten()
                slowness_space_np = to_np(slowness_space)
                slowness_power_np = to_np(slowness_power).flatten()
                slowness_best_np = to_np(slowness_best)
    
                # --- Trace ---
                ax_trace  = fig.add_subplot(gs[row_base, col])
                ax_trace.plot(time, signal, color=color)
                if i == 0:
                    ax_trace.set_title(f"Cluster {cluster_label} Medoid" if cluster_label!=-1 else "Outlier", fontsize=10)
                else:
                    ax_trace.set_title(f"Example {i}", fontsize=8)
                ax_trace.set_ylabel("Amplitude")
    
                # --- CWT ---
                ax_cwt    = fig.add_subplot(gs[row_base + 1, col])
                coeffs, freqs = pywt.cwt(signal, scales=scales, wavelet=waveletname, sampling_period=1/sampling_rate)
                coeffs_log = np.log10(np.abs(coeffs)+1e-12)
                ax_cwt.pcolormesh(time, freqs, coeffs_log, shading='gouraud', cmap='magma',
                                   vmin=np.percentile(coeffs_log,3), vmax=np.percentile(coeffs_log,99))
                ax_cwt.set_ylabel("Freq [Hz]")
                ax_cwt.set_xlabel("Time [s]")
                ax_cwt.set_ylim(freq_min, freq_max)
    
                # --- Beamforming ---
                ax_beam   = fig.add_subplot(gs[row_base + 2, col], projection='polar')
                theta = np.arctan2(slowness_space_np[:,1], slowness_space_np[:,0])
                r = np.linalg.norm(slowness_space_np, axis=1)
                ax_beam.scatter(theta, r, c=slowness_power_np, cmap='magma', s=5)
                ax_beam.plot(np.arctan2(slowness_best_np[1], slowness_best_np[0]), np.linalg.norm(slowness_best_np), 'ro')
                ax_beam.set_title("Beamforming", fontsize=8)
                ax_beam.set_yticklabels([])
                ax_beam.grid(False)
    
                # --- Distance zu allen Medoids (nur echte Cluster) ---
                ax_dist   = fig.add_subplot(gs[row_base + 3, col])
                
                valid_clusters_nonoutlier = [c for c in clusters_all if c >= 0]
                
                if example_idx < all_distances.shape[0]:
                    # Distanz von diesem Sample zu allen echten Medoids (Spaltenindex, nicht globaler Index!)
                    dists = [all_distances[example_idx, label_to_col[c]] for c in valid_clusters_nonoutlier]
                else:
                    dists = [np.nan]*len(valid_clusters_nonoutlier)
                
                ax_dist.bar(valid_clusters_nonoutlier, dists, color=[cluster_colors[c] for c in valid_clusters_nonoutlier])
                ax_dist.set_ylim(0, 0.025)
                ax_dist.set_ylabel("Dist to Medoids")
                ax_dist.set_xlabel("Cluster")
                ax_dist.set_title("Distances", fontsize=8)
                ax_dist.set_xticks(valid_clusters_nonoutlier)
                ax_dist.set_xticklabels([f"C{c}" for c in valid_clusters_nonoutlier], rotation=45)

    
            except Exception as e:
                print(f"Example plot error idx {example_idx}: {e}")
                for idx_off in range(4):
                    fig.add_subplot(gs[3*i + idx_off, col]).axis("off")
    
        # --- Histogramme für den Cluster / Outlier ---
        try:
            hist_start_row = n_examples_per_cluster * n_rows_per_example  # erste Reihe der Histogramme
            # Beamforming direction
            ax_hist_dir = fig.add_subplot(gs[hist_start_row, col], projection='polar')
            ax_hist_dir.hist(slowness_angles, bins=n_bins, color=color if cluster_label!=-1 else 'grey', alpha=0.8)
            ax_hist_dir.set_title("Dir Histogram", fontsize=8)
            ax_hist_dir.set_yticklabels([])
            ax_hist_dir.grid(True)
    
            # Slowness magnitude
            ax_rhist    = fig.add_subplot(gs[hist_start_row + 1, col])
            ax_rhist.hist(slowness_mags, bins=20, color=color, alpha=0.8)
            ax_rhist.set_title("|Slowness| [s/km]", fontsize=8)
            ax_rhist.set_xlabel("Magnitude")
            #ax_rhist.set_yticklabels([])
            ax_rhist.grid(True, alpha=0.3)
            
            # Duration
            ax_dur      = fig.add_subplot(gs[hist_start_row + 2, col])
            ax_dur.hist(durations, bins=10, color=color, alpha=0.8)
            ax_dur.set_title("Duration [s]", fontsize=8)
            ax_dur.set_xlabel("Seconds")
            #ax_dur.set_yticklabels([])
            ax_dur.set_xlim(10,30)
    
        except Exception as e:
            print(f"Cluster histogram error cluster {cluster_label}: {e}")
    
    plt.tight_layout()
    plt.savefig(save_plt_path+'/examples_every_cluster.png')
    



    #%% PLOT OF EXAMPLES AND HOW LIKELY THEY BELONG TO EACH CLUSTER
    clusters = sorted(cluster_to_index.keys())
    n_clusters = len(clusters)
    n_rows = n_examples_per_cluster
    fig, ax = plt.subplots(2 * n_rows + 2, n_clusters, figsize=(16, 10))

    for col, cluster_label in enumerate(clusters):
        color = cluster_colors[cluster_label]
        all_indices = cluster_to_index[cluster_label]

        if cluster_label >= 0:
            medoid_idx = IDX_medoids_all[label_to_col[cluster_label]]

            # plot medoid signal
            ax[0][col].plot(medoids_time[label_to_col[cluster_label]],
                            medoids_beams[label_to_col[cluster_label]], color=color)

            # bar of distances
            # Get the corresponding column indices for valid clusters
            valid_cols = [label_to_col[c] for c in valid_clusters if c in label_to_col]
            
            ax[1][col].bar(valid_clusters,
                           np.array(all_distances[medoid_idx])[valid_cols],
                           color=[cluster_colors[c] for c in valid_clusters])

            ax[1][col].set_ylim(0, 0.012)

            # examples
            example_indices = random.sample(all_indices, min(n_examples_per_cluster, len(all_indices)))
            for idx, ex_idx in enumerate(example_indices):
                signal = to_np(BEAMS[ex_idx]).flatten()
                time = to_np(TIMES[ex_idx]).flatten()
                ax[idx * 2 + 2][col].plot(time, signal, color=color)
                ax[idx * 2 + 2][col].set_yticks([])
                valid_cols = [label_to_col[c] for c in valid_clusters if c in label_to_col]

                ax[idx * 2 + 3][col].bar(
                    valid_clusters,
                    np.array(all_distances[ex_idx])[valid_cols],
                    color=[cluster_colors[c] for c in valid_clusters]
                )

                ax[idx * 2 + 3][col].set_ylim(0, 0.0025)

    # remove unused subplot
    ax[0][0].set_visible(False)
    ax[1][0].set_visible(False)
    ax[6][0].set_xlabel('time [s]')
    ax[6][0].set_ylabel('counts [-]')
    ax[7][0].set_xlabel('cluster')
    ax[7][0].set_ylabel('min dist')

    #%% PLOT TEMPORAL DISTRIBUTION OF CLUSTERS



    # 2. --- Plot temporal histogram of clustering ---
    df = pd.DataFrame({"time": TIMES_UTC, "label": labels})
    df["day"] = pd.to_datetime(df["time"]).dt.date

    # Count per day, per cluster
    counts = df.groupby(["label", "day"]).size().reset_index(name="count")
    clusters = sorted(set(labels))
    n_clusters = len(clusters)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 1 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]

    for ax, cluster in zip(axes, clusters):
        cluster_data = counts[counts["label"] == cluster]
        ax.bar(cluster_data["day"], cluster_data["count"], color=cluster_colors[cluster])
        ax.set_title(f"Cluster {cluster}")
        ax.set_ylabel("Detections")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Day")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.savefig(save_plt_path+'/temporal_distibution.png')
    print("Plotting funktioniert!")
    
