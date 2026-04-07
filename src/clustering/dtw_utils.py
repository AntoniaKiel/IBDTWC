# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 14:45:10 2025

@author: anton
"""
import numpy as np
import random, logging
from tslearn.metrics import cdist_dtw
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt




class IncrementalDTW:
    def __init__(self, seed=42,
                 buffer_size=5, buffer_threshold=0.15, max_buffer_age=20,
                  sakoe_chiba_radius=3,sigma_factor=2.0):

        self.medoids = []
        self.labels_ = []
        self.seed = seed
        self.unclustered_buffer = []
        self.buffer_labels = []
        self.buffer_size = buffer_size
        self.buffer_threshold = buffer_threshold
        self.max_buffer_age = max_buffer_age
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.samples = []  # <--- keep all samples here
        
        log_file="clustering.log"
        random.seed(seed)
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.sigma_factor = sigma_factor
    

    def _zscore_scale_series(self, series_list):
        """
        Apply Z-score scaling to each time series in a list.
        """
        scaled = []
        for s in series_list:
            arr = np.asarray(s, dtype=float).flatten()
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:  # vermeide Division durch Null
                scaled.append(arr - mean)
            else:
                scaled.append((arr - mean) / std)
            # Wichtig: wieder in die ursprüngliche Shape bringen (Nx1)
            scaled[-1] = scaled[-1].reshape(-1, 1)
        return scaled

    def compute_dtw(self, a, b):
        return cdist_dtw([a], [b], sakoe_chiba_radius=self.sakoe_chiba_radius)[0][0] / (len(a) + len(b))

    def _normalize_dtw_matrix(self, dtw_matrix, data):
        dtw_norm = dtw_matrix.copy()
        for i in range(len(dtw_matrix)):
            for j in range(len(dtw_matrix)):
                dtw_norm[i, j] = dtw_matrix[i, j] / (len(data[i]) + len(data[j]))
        return dtw_norm

    def fit_initial(self, initial_data, n_clusters=3):
        self.logger.info(f"[INIT] Initializing with {len(initial_data)} samples...")
        
        # Z-Score Scaling HIER
        scaled_data = self._zscore_scale_series(initial_data)
    
        self.samples = list(scaled_data)
    
        dtw_matrix = cdist_dtw(scaled_data, sakoe_chiba_radius=self.sakoe_chiba_radius)
        dtw_matrix = self._normalize_dtw_matrix(dtw_matrix, scaled_data)

        kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed",
                            init="k-medoids++", random_state=self.seed)
        kmedoids.fit(dtw_matrix)

        self.medoids = [initial_data[i] for i in kmedoids.medoid_indices_]
        self.labels_ = list(kmedoids.labels_)  # length = len(initial_data)
        
        # calculating minimal distances to medioids
        
        min_distances = np.zeros(len(initial_data))
        for idx, label in enumerate(self.labels_):
            medoid_idx = kmedoids.medoid_indices_[label]
            min_distances[idx] = dtw_matrix[idx, medoid_idx]
        
        self.min_distances_ = list(min_distances)  # <--- extra Output
        return self.labels_, self.medoids, self.min_distances_
    
    
    def update_distance_threshold(
        self,
        min_distances,
        sigma_factor
    ):
        """
        Simplified adaptive threshold update:
        - Uses only mean and std of all min_distances (no GMM).
        - Optionally uses percentile if provided.
        """
    
        clean = np.array([d for d in min_distances if d is not None and np.isfinite(d)])
        if clean.size == 0:
            raise ValueError("No valid min_distances provided.")
    
        overall_mean = clean.mean()
        overall_std = clean.std(ddof=0)
    
        # Default threshold: mean + sigma_factor * std
        threshold = overall_mean + sigma_factor * overall_std
    
    
        self.distance_threshold = float(threshold)
        self.threshold_mean = overall_mean
        self.threshold_std = overall_std
    
        print(f"[THRESH] mean={overall_mean:.6f}, std={overall_std:.6f}, threshold={threshold:.6f}")
        return threshold
    
        

    def add_sample(self, x):
        # record the sample so finalize() sees it
        self.samples.append(x)  # <--- append every new sample

        if not self.medoids:
            self.logger.info("[INIT] First medoid created.")
            self.medoids.append(x)
            self.labels_.append(0)
            return 0, 0.0

        distances = np.array([self.compute_dtw(x, medoid) for medoid in self.medoids])
        min_dist = float(np.min(distances))
        closest_cluster = int(np.argmin(distances))
        self.min_distances_.append(min_dist)

        if min_dist <= self.distance_threshold:
            self.logger.info(f"[ASSIGN] Sample assigned to existing cluster {closest_cluster} (DTW: {min_dist:.3f})")
            self.labels_.append(closest_cluster)
            return closest_cluster, min_dist

        # buffer path
        self.logger.info(f"[BUFFER] Sample added to buffer (DTW to closest: {min_dist:.3f})")
        self.unclustered_buffer.append(x)
        self.buffer_labels.append(len(self.labels_))  # index of this sample in labels_
        self.labels_.append(-1)

        if len(self.unclustered_buffer) >= self.buffer_size:
            scaled_buffer = self._zscore_scale_series(self.unclustered_buffer)
            dtw_matrix = cdist_dtw(scaled_buffer, sakoe_chiba_radius=self.sakoe_chiba_radius)
            dtw_matrix = self._normalize_dtw_matrix(dtw_matrix, scaled_buffer)
            avg_dtw = float(np.mean(dtw_matrix))

            if avg_dtw < self.distance_threshold:
            #if avg_dtw < self.buffer_threshold:
                kmedoids = KMedoids(n_clusters=1, metric="precomputed",
                                    init="k-medoids++", random_state=self.seed)
                kmedoids.fit(dtw_matrix)
                new_medoid = self.unclustered_buffer[kmedoids.medoid_indices_[0]]
                new_label = len(self.medoids)

                self.medoids.append(new_medoid)
                for idx in self.buffer_labels:
                    self.labels_[idx] = new_label

                self.logger.info(f"[NEW CLUSTER] Formed cluster {new_label} from buffer "
                                 f"(size: {len(self.buffer_labels)}, avg DTW: {avg_dtw:.3f})")
                self.unclustered_buffer.clear()
                self.buffer_labels.clear()
                return new_label, float("inf")

        if len(self.unclustered_buffer) > self.max_buffer_age:
            self.logger.info(f"[FLUSH] Buffer flushed. {len(self.unclustered_buffer)} samples marked as outliers.")
            self.unclustered_buffer.clear()
            self.buffer_labels.clear()
            return -1, min_dist

        return -1, min_dist

    
    
    def finalize_all_distances(self, beams=None):
        if beams is None:
            beams = self.samples
    
        n_samples = len(beams)
        n_medoids = len(self.medoids)
    
        # --- Compute distance matrix ---
        all_distances = np.empty((n_samples, n_medoids), dtype=float)
        for i, x in enumerate(beams):
            for j, m in enumerate(self.medoids):
                all_distances[i, j] = self.compute_dtw(x, m)
    
        final_min_distances = np.min(all_distances, axis=1)
        closest_medoid_idx = np.argmin(all_distances, axis=1)
    
        # Use medoid indices as labels: 0..n_medoids-1
        new_labels = closest_medoid_idx.copy()
    
        # Mark outliers
        new_labels[final_min_distances > self.distance_threshold] = -1
    
        # Count cluster sizes
        unique_labels, counts = np.unique(new_labels[new_labels != -1], return_counts=True)
    
        # Identify clusters too small to keep
        small_clusters = set(unique_labels[counts < self.buffer_size])
    
        # Mark them as outliers
        for sc in small_clusters:
            new_labels[new_labels == sc] = -1
    
        # --- IMPORTANT: Remove unused medoids ---
        used_clusters = sorted(np.unique(new_labels[new_labels >= 0]))
        new_medoids = [self.medoids[idx] for idx in used_clusters]
    
        # Remap labels to compact 0..K-1
        remap = {old: new for new, old in enumerate(used_clusters)}
        new_labels = np.array([remap[l] if l in remap else -1 for l in new_labels])
    
        # Update model state
        self.medoids = new_medoids
        self.labels_ = new_labels.tolist()
    
        return all_distances, final_min_distances, new_labels, used_clusters



    


def evaluate_cluster_numbers(BEAMS, min_cluster=3, max_cluster=11, s_c_radius=10):
    cluster_range = range(min_cluster, max_cluster)



    print("Computing DTW matrix for elbow method...")
    dtw_matrix = cdist_dtw(BEAMS,sakoe_chiba_radius=s_c_radius)

    # Elbow Method (KMedoids)
    costs = []
    for k in cluster_range:
        kmedoids = KMedoids(n_clusters=k, metric="precomputed", init="k-medoids++")
        kmedoids.fit(dtw_matrix)
        cost = np.sum(np.min(dtw_matrix[:, kmedoids.medoid_indices_], axis=1))
        costs.append(cost)
    
    return costs,  list(cluster_range)

# Optionally visualize elbow results
def plot_elbow(costs, cluster_range):
    #for i, costs in enumerate(elbow_results):
    plt.plot(cluster_range, costs, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Sum of DTW distances to medoids")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()








