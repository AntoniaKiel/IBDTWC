
import pandas as pd
import os, pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

#from src.clustering.dtw_utils_update import evaluate_cluster_numbers, plot_elbow, IncrementalDTW

def plot_threshold_diagnostics(model, min_distances, save_path_cluster):
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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean, bins=50, density=True, alpha=0.6, color='lightgray', edgecolor='black', label='init_min_distances')

    ax.plot(xs, pdf_overall, color='orange', linewidth=2, label=f'N(mean={overall_mean:.4f}, std={overall_std:.4f})')

    # vertical lines
    ax.axvline(overall_mean, color='blue', linestyle=':', linewidth=2, label='Mean')
    ax.axvline(model.distance_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = mean + {getattr(model, "sigma_factor", 1.5)}·σ')

    ax.set_xlabel('DTW distance')
    ax.set_ylabel('Density')
    #ax.set_title('Threshold diagnostics (Normal PDF only)')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()

    os.makedirs(save_path_cluster, exist_ok=True)
    savefile = os.path.join(save_path_cluster, 'threshold_diagnostics_simple.png')
    fig.savefig(savefile, dpi=200)
    plt.close(fig)
    print(f"[THRESH PLOT] Saved simplified threshold diagnostics to {savefile}")


def dtw_function(load_path, sakoe_chiba_radius_time, n_clusters, init_sample_size,
                 n_min_cluster, n_max_cluster, sampling_freq,
                 sigma_factor=1.5, buffer_size=5,max_buffer_age=100,
                 save_data=True, reextracting_init_beams=True,
                 recalculating=False, test_n_clusters=True, subset=None,
                 n_seed=42):
    #%% LOAD EVENT DATA
    # read file paths to seismic events with good quality signals
    df=pd.read_csv(load_path / 'catalog_data.csv')
    beam_paths = df.loc[(df["beam"] == True) & (df["used"] == True), "ID"].unique().tolist()
    print(f"Number of events: {len(beam_paths)}")

    # subsample dataset if desired (using the first 'subset' ones)
    if subset:
        random_paths = beam_paths.copy()[:subset]
    else:
        random_paths = beam_paths.copy()

    # save beam, relative times and UTC times in list 
    #    list important due to difference in length of data
    BEAMS = []
    TIMES = []
    TIMES_UTC_start = []

    for idx, path in enumerate(random_paths):
        beam_data_path = os.path.join(path, "BEAM_results.npy")
        data = np.load(beam_data_path, allow_pickle=True)[0]
        
        # mean across shifted_traces (= beam)
        beam = np.mean(data["shifted_traces"], axis=0)  # shape: (N,)
    
        # --- Normalisierung (Z-Score) ---
        #    this is to help the clustering to be insensitive towards amplitudes
        beam_mean = np.mean(beam)
        beam_std = np.std(beam)

        beam = (beam - beam_mean) / beam_std
        
        # reshape for DTW (Form: (N, 1))
        BEAMS.append(beam.reshape(-1, 1))
        TIMES.append(data["times relative"])
        TIMES_UTC_start.append(data["times_date"][0])
    
        #print(f"Loaded beam {idx + 1}/{len(random_paths)} (normalized)")

    # ============== CLUSTERING ==============================
    #%% PREPARE CLUSTERING STRUCTURE
            
    np.random.seed(n_seed)
    rng = np.random.default_rng(seed=n_seed)
    
    sakoe_chiba_radius_value=sakoe_chiba_radius_time * sampling_freq # time -> samples

    save_path_main = f'./data/2_clustering_results/scRadi{sakoe_chiba_radius_time}_seed{n_seed}'
    os.makedirs(save_path_main, exist_ok=True)
    save_path_cluster = f'{save_path_main}/n_clusters{n_clusters}'
    os.makedirs(save_path_cluster, exist_ok=True)

    #%%
    # =============================================================================
    # CACHING INITIAL CLUSTERING
    # =============================================================================
    # this secion calculated the pairwise distance matrix which gets clustered. Based on this, the number of clusters is evaluated before other samples are added
    cache_file_indices = os.path.join(save_path_main, "init_indices.npy")
    cache_file_initial_beams = os.path.join(save_path_main, "initial_beams.npy")
    cache_file_initial_times = os.path.join(save_path_main, "initial_times.npy")
    cache_file_initial_times_UTC = os.path.join(save_path_main, "initial_times_UTC.npy")

    
    if os.path.exists(cache_file_indices) and os.path.exists(cache_file_initial_beams) and (reextracting_init_beams==False):
        # if for this setting inital clustering already exists, don not recalculate this (esp. since computationally expensive)
        print("Loading cached initial clustering data...")
        init_indices = np.load(cache_file_indices)
        initial_beams = list(np.load(cache_file_initial_beams, allow_pickle=True))
        initial_times = list(np.load(cache_file_initial_times, allow_pickle=True))
        initial_times_UTC = list(np.load(cache_file_initial_times_UTC, allow_pickle=True))
    else:
        print("Extracting beams for inital clustering...")
        # --- Group indices by day ---
        indices_by_day = defaultdict(list)
        for idx, t in enumerate(TIMES_UTC_start):
            indices_by_day[t.date()].append(idx)
        
        # --- Compute how many per day for initial clustering ---
        #   this is to ensure that I have an more enenly sampling and ideally clustering
        #   in the initial matrix which represent event types of different seasons
        days = list(indices_by_day.keys())
        samples_per_day = init_sample_size // len(days)
        
        init_indices = []
        
        # --- Sample per day ---
        #   here it 'neatly' samples from the existing days until len(init_indices) = days*samples_per_day
        for day, idxs in indices_by_day.items():
            n_to_sample = min(samples_per_day, len(idxs))
            init_indices.extend(rng.choice(idxs, size=n_to_sample, replace=False))
        
        # --- Fill up to total_init if we didn't reach it ---
        remaining_needed = init_sample_size - len(init_indices)
        if remaining_needed > 0:
            all_indices = set(range(len(TIMES_UTC_start)))
            already_selected = set(init_indices)
            remaining_candidates = list(all_indices - already_selected)
            extra_indices = rng.choice(remaining_candidates, size=remaining_needed, replace=False)
            init_indices.extend(extra_indices)
        
        init_indices = np.array(init_indices)
        
        # --- Save cached initial clustering ---
        #   
        np.save(cache_file_indices, init_indices)
        initial_beams = [BEAMS[i] for i in init_indices]
        initial_times=[TIMES[i] for i in init_indices]
        initial_times_UTC=[TIMES_UTC_start[i] for i in init_indices]
        np.save(cache_file_initial_beams, np.array(initial_beams, dtype=object))
        np.save(cache_file_initial_times, 
                np.array([t.detach().cpu().numpy() for t in initial_times], dtype=object))
        np.save(cache_file_initial_times_UTC, np.array(initial_times_UTC, dtype=object))

    #%%
    # =============================================================================
    # ELBOW METHOD ON INITIAL CLUSTERING
    # =============================================================================
    # if desired, the elbow method can be calculated on pairwise distance matrix 
    if test_n_clusters:
        elbow_results, cluster_range = evaluate_cluster_numbers(
            initial_beams,
            min_cluster=n_min_cluster,
            max_cluster=n_max_cluster,
            s_c_radius=sakoe_chiba_radius_value
        )
        plot_elbow(elbow_results, cluster_range)
        #for i, costs in enumerate(elbow_results):
        plt.plot(cluster_range, elbow_results, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Sum of DTW distances to medoids")
        plt.title("Elbow Method - Initial Clustering")
        plt.grid(True)
        plt.savefig(save_path_main + '/elbow_initial.png')
        plt.show()
        

    # load existing model if it was already calculated
    cache_file_model = os.path.join(save_path_main, "model_initial.pkl")

    if os.path.exists(cache_file_model) and (recalculating==False):
        print("Loading cached initial model...")
        with open(cache_file_model, "rb") as f:
            model = pickle.load(f)
    else:
        print("Fitting initial model...")
        
        # 1 Initialisiere clustering algorithm
        model = IncrementalDTW( sakoe_chiba_radius=sakoe_chiba_radius_value,
                                sigma_factor=sigma_factor, seed=n_seed,
                                buffer_size=buffer_size,max_buffer_age=max_buffer_age)

        
        # 2 excecute inital clustering
        labels, medoids, init_min_dists = model.fit_initial(initial_beams, n_clusters=n_clusters)
        
        # --- adaptive threshold calculation on distances of initial clustering ---
        # use only non-zero distances (these relate to medoids)
        init_min_dists_arr = np.array(init_min_dists, dtype=float).flatten()
        
        # crete mask: neither 0.0 nor NaN/inf
        mask = (~np.isclose(init_min_dists_arr, 0.0)) & np.isfinite(init_min_dists_arr)
        
        # new, cleaned distances without medoids
        init_min_dists_nzero = init_min_dists_arr[mask]

        
        new_threshold = model.update_distance_threshold(
            init_min_dists_nzero, sigma_factor=model.sigma_factor
        )

        plot_threshold_diagnostics(model, init_min_dists_nzero, save_path_cluster)

        model.distance_threshold = new_threshold
        print(f"new dynamic threshold: {new_threshold:.4f}")
        
            # --- SAVE THE MODEL ---
        print("Saving initial model to cache...")
        with open(cache_file_model, "wb") as f:
            pickle.dump(model, f)

    # FIT INCREMENTALLY ADDED SAMPLES
    # --- Remaining indices & beams ---
    remaining_indices = np.setdiff1d(np.arange(len(TIMES_UTC_start)), init_indices)
    remaining_beams = [BEAMS[i] for i in remaining_indices]
    remaining_times = [TIMES[i] for i in remaining_indices]
    remaining_times_UTC = [TIMES_UTC_start[i] for i in remaining_indices]
                     
    # Incrementally add remaining beams
    min_distances=list(init_min_dists)
    for counter, sample in enumerate(remaining_beams):
        label, min_dist = model.add_sample(sample)
        min_distances.append(min_dist)
        
        print(f"{(counter / len(remaining_beams)) * 100:.2f}% added")
        
    # =============================================================================
    # RESTORE ORIGINAL ORDER & SAVE
    # =============================================================================
    # Finalize distances
    BEAMS_ordered = initial_beams + remaining_beams
    TIMES_ordered = initial_times + remaining_times
    TIMES_UTC_ordered= initial_times_UTC + remaining_times_UTC
    
    # --- Recalculate threshold after all samples were added ---
    print("Recalculating distance threshold on all all min distances ...")
    
    # Clean min_distances
    final_min_dists_arr = np.array(min_distances, dtype=float).flatten()
    mask_final = (~np.isclose(final_min_dists_arr, 0.0)) & np.isfinite(final_min_dists_arr)
    final_min_dists_clean = final_min_dists_arr[mask_final]
    
    # Compute final threshold
    final_threshold = model.update_distance_threshold(
        final_min_dists_clean,
        sigma_factor=model.sigma_factor
    )
    
    model.distance_threshold = final_threshold
    
    print(f"Final dynamic distance threshold set to: {final_threshold:.6f}")
                     
    # re-calculate clustering distances using final threshold
    all_distances, min_distances, labels, cluster_labels = model.finalize_all_distances(BEAMS_ordered)
                     
    # Shuffle indices we used
    shuffle_indices = np.concatenate([init_indices, remaining_indices])
    inverse_indices = np.argsort(shuffle_indices)

    # Reorder everything back to match original random_paths order
    # this is important if we later want to reload some data from beam oaths together with the labels
    labels_original_order = np.array(labels)[inverse_indices]
    all_distances_original_order = np.array(all_distances, dtype=object)[inverse_indices]
    min_distances_original_order = np.array(min_distances)[inverse_indices]
    BEAMS_original_order = [BEAMS_ordered[i] for i in inverse_indices]
    TIMES_original_order = [TIMES_ordered[i] for i in inverse_indices]
    TIMES_UTC_original_order = [TIMES_UTC_ordered[i] for i in inverse_indices]
    

    if save_data:
        np.save(save_path_cluster + '/labels.npy', labels_original_order)
        np.save(save_path_cluster + '/min_distances.npy', min_distances_original_order)
        np.save(save_path_cluster + '/all_distances.npy', all_distances_original_order)
        np.save(save_path_cluster + '/BEAMS.npy', np.array(BEAMS_original_order, dtype=object))
        np.save(save_path_cluster + '/TIMES.npy', 
                np.array([t for t in TIMES_original_order], dtype=object))
        np.save(save_path_cluster + '/TIMES_UTC.npy', np.array(TIMES_UTC_original_order, dtype=object))
        np.save(save_path_cluster + '/shuffle_indices.npy', shuffle_indices)
        np.save(save_path_cluster + '/distance_threshold.npy', model.distance_threshold)

        print(" Clustering completed and saved.")






















