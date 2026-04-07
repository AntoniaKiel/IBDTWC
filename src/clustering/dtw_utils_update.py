import numpy as np
import random
import logging
from sklearn_extra.cluster import KMedoids
from tslearn.metrics import cdist_dtw
import matplotlib.pyplot as plt

class IncrementalFDTW:
    '''
    Incrementally buffered Dynamic Time Warping Clustering (IB-DTWC) algorithm.


    This class manages all three steps of the clustering:

    1) Initial Clustering on Pairwise Distance Matrix of Subset
    2) Incrementally addition of new samlpes including a buffer
        1. Check 
        This can result in both outlieres as well as new clusters 

    '''
    def __init__(self, seed=42, buffer_size=5, max_buffer_age=20,
                 sakoe_chiba_radius=50, sigma_factor=2.0,cluster_range=range(3,7)):
        """Initialise clustering algorithm

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. 
            Ensures same samples are used for initial subset and inital clustering of
            pairwise distance matrix, by default 42
        buffer_size : int, optional
            number of samples, which have to exist in a buffer cluster to be upgraded
            to a new, true cluster, by default 5
        max_buffer_age : int, optional
            after this number of added samples the buffer is checked for samples, which are
            still without any other buffer cluster samples. If they fulfill this condition, they
            are permanently classified as outlier. This save RAM because less buffer clusters are 
            stored, by default 20
        sakoe_chiba_radius : int, optional
            Number of samples of signal to check for shift in DTW. This reduced computiational cost due to decrease of
            calculations for every DTW combination of samples, by default 50
        sigma_factor : float, optional
            Parameters defining the DTW distance threshold for samples to still relate to a medoid.
            This admissibility factor is used in mu=mean(min_distances)+sigma*std(min_distances), making
            the threshold data driven, by default 2.0
        cluster_range: list of integers
            Cluster numbers to check for inertia in inital clustering of pairwise distance matrix
        """

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.buffer_size = buffer_size
        self.max_buffer_age = max_buffer_age
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.sigma_factor = sigma_factor
        self.cluster_range= cluster_range

        ### PREPARATION OF LISTS IN IB-DTWC
        # --- Medoids & Samples ---
        self.medoids = []               # List of normalised medoids
        self.samples = []               # List of normalised samples

        # --- Labels / min distances ---
        self.labels_ = []
        self.min_distances_ = []    # DTW distance to closest medoid

        # --- Separate storage after initial clustering,
        #  incremental addition and fianlised clustering ---
        self.labels_init = []
        self.labels_add = []
        self.labels_final = []

        self.min_distances_init = []
        self.min_distances_add = []
        self.min_distances_final = []

        self.threshold_init = None
        self.threshold_add = None
        self.threshold_final = None

        # --- store inertia information ---
        self.inertia = []

        # --- store buffer groups ---
        self.buffer_groups = []

        # start logging
        self.logger = logging.getLogger(__name__)

    # ---------------- Z-Score Scaling ----------------
    def _zscore_scale_series(self, series_list):
        """Thus function calculates the z-score scaling on input seismic traces.
        This is important to avoid bias towards amplitudes in DTW calculation while
        keepting the ratio between peaks.

        Parameters
        ----------
        series_list : list
            list of seismic traces (list important to work for different window length)

        Returns
        -------
        list
            list of input seismic traces, each normalised using z-score
        """

        scaled = []
        for s in series_list:
            arr = np.asarray(s, dtype=float).flatten()
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0: # avoid dividing by 0 if std = 0
                scaled.append(arr - mean)
            else:
                scaled.append((arr - mean) / std)
            scaled[-1] = scaled[-1].reshape(-1, 1)
        return scaled

    # ---------------- DTW Distance ----------------
    def compute_dtw(self, a, b):
        """This function calculated the normalised dynamic time warping distance of two signals

        Parameters
        ----------
        a : _type_
            Time series of one signal
        b : _type_
            Time signal of the compared signal

        Returns
        -------
        float
            Normalised DTW distance between the two signals
        """
        d = cdist_dtw([a], [b], sakoe_chiba_radius=self.sakoe_chiba_radius)[0][0]
        return float(d) / (len(np.asarray(a).flatten()) + len(np.asarray(b).flatten()))

    # ---------------- Initial Clustering ----------------
    def fit_initial(self, initial_data:list, n_clusters:int,cluster_range,compute_inertia=True):
        """This function clusters the pairwise DTW distance matrix of the inital subset.
        It includes the inertia calculation for several cluster numbers if compute_inertia = True.
        The results are saved in the object so no return needed.

        Parameters
        ----------
        initial_data : list
            Subset of data to use for initial clustering. This is the direct input data, scaling is applied
            within this function
        n_clusters : int
            set number of clusters to use to kmedoids clustering of pairwise DTW distance matrix
        cluster_range : _type_
            range of int values to check for clustering performance using inertia as quality criteria
        compute_inertia : bool, optional
            should the inertia be calculated to given cluster_range? can be skipped if this was already
            investigated and n_clusters decided. False saves computing time, by default True

        """
        
        self.logger.info("Starting intial clustering step of IB-DTWC")
        # z-score normalisation of input data
        scaled_data = self._zscore_scale_series(initial_data)
        self.samples = list(scaled_data)

        # pairwise DTW distance matrix calculation
        dtw_matrix = np.zeros((len(scaled_data), len(scaled_data)))
        for i in range(len(scaled_data)):
            for j in range(len(scaled_data)):
                dtw_matrix[i, j] = self.compute_dtw(scaled_data[i], scaled_data[j])

        self.logger.debug("dtw matrix calculated!")

        # KMedoids for different number of clusters
        if compute_inertia == True: # skip calculation if number of clusters defined to save computating time
            for n_cl in cluster_range:
                kmedoids = KMedoids(
                    n_clusters=n_cl,
                    metric="precomputed",
                    init="k-medoids++",
                    random_state=self.seed
                )
                kmedoids.fit(dtw_matrix)
                self.inertia.append(kmedoids.inertia_)

                self.logger.debug(f"inertia calculated for {n_cl} of {cluster_range[-1]}")
        
        # calculate for set number of clusters
        kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed",
                    init="k-medoids++", random_state=self.seed)
        kmedoids.fit(dtw_matrix)
        self.logger.debug('k-medoids on pairwise DTW distance done')

        # determine medoids of intial clusters
        self.medoids = [scaled_data[i] for i in kmedoids.medoid_indices_]

        # labels + min-Distances of initial clustering
        self.labels_ = list(kmedoids.labels_)
        min_distances = np.zeros(len(initial_data))
        for idx, label in enumerate(self.labels_):
            medoid_idx = kmedoids.medoid_indices_[label]
            min_distances[idx] = dtw_matrix[idx, medoid_idx]

        # store intial labels and min_distances seperatly
        self.labels_init = list(self.labels_)
        self.min_distances_init = list(min_distances)

        # calculate distance threshold automatically based on min_distances
        self.threshold_init = self.update_distance_threshold_auto(min_distances)

    # ---------------- Adaptive Threshold ----------------
    def update_distance_threshold_auto(self, min_distances:list):
        """this function calculated the threshold based on the sigma factor and
        min_distances of the input data.

        Parameters
        ----------
        min_distances : list
            this list contains the DTW distances to the closest medoid of the given samples
            The list is used as input instead of e.g. self.min_distances to ensure the correct set of
            min distances is used depending on the application.
        Returns
        -------
        float
            DTW distance value to use as threshold based on the threshold= mean(min_distances) + sigma * std(min_distances)

        Raises
        ------
        ValueError
            if the given min distances provieded do not contain valid values, return ValueError
        """
        clean = np.array([d for d in min_distances if d is not None and np.isfinite(d)]) # check that the min_distances list has only finite values and is not None
        if clean.size == 0:
            raise ValueError("No valid min_distances provided.")
        
        mean = clean.mean()
        std = clean.std(ddof=0)
        threshold = mean + self.sigma_factor * std
            
        return threshold

    # ---------------- Add Sample ----------------
    def add_sample(self, x):
        """ This function contains the logic of incrementally adding samples.
        The incremental step only adds new samples and defined new medoids if buffer cluster is upgrades to true cluster.
        Assigning the labels to previous buffer cluster samples is done in the finialising step.

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # z-score scaling of input seismic event
        x_scaled = self._zscore_scale_series([x])[0]
        self.samples.append(x_scaled)

        # calculate distances to existing medoids
        distances = np.array([self.compute_dtw(x_scaled, m) for m in self.medoids])
        min_dist = float(np.min(distances))
        closest_cluster = int(np.argmin(distances))

        # Adaptive Threshold aus Initial
        threshold_add = self.update_distance_threshold_auto(self.min_distances_init)
        self.threshold_add = threshold_add

        # --- add sample to existing cluster if cloestes medoid closer than threshold ---
        if closest_cluster is not None and min_dist <= threshold_add:
            self.labels_.append(closest_cluster) # add sample label to entirety of labels
            self.labels_add.append(closest_cluster) # add sample label to labels related purely to the incremental step
            self.min_distances_.append(min_dist)  # add shortest DTW distance to all min_distances
            self.min_distances_add.append(min_dist) # add shortest DTW distance to min distances related purely to the incremental 
            
            # Prüfen, ob Buffer-Gruppen wachsen
            # self._check_buffer_groups_against_sample(x_scaled)
            return closest_cluster, min_dist

        # --- Sample passt zu keinem Cluster → Buffer ---
        assigned = False # save if the sample gets added to a different buffer cluster

        for group in self.buffer_groups: 
            dist_to_group = self.compute_dtw(x_scaled, group[0]) # compute dtw distance to medoids of buffer group
            if dist_to_group <= self.threshold_init:
                group.append(x_scaled)  # assign to buffer cluster if sample fits
                assigned = True
                if len(group) >= self.buffer_size: # check if buffer cluster large engough
                    new_medoid = group[0]  
                    self.medoids.append(new_medoid)
                    # remove buffer cluster from buffer (samples besides medoid stay unassigned until finalising step but are removed from buffer)
                    self.buffer_groups.remove(group)
                break

        # no buffer samples fit -> sample forms new buffer cluster
        if not assigned:
            self.buffer_groups.append([x_scaled])


        # mark sample as outlier (as soon as it was once related to the buffer. This includes new medoids and related samples. They are assigned to their labels in the finialising step!)
        self.labels_.append(-1)
        self.labels_add.append(-1)

        # add lowest distance to closets medoid 
        self.min_distances_.append(min_dist)
        self.min_distances_add.append(min_dist)
        return -1, min_dist

    # def _check_buffer_groups_against_sample(self, x_scaled):
    #     # Prüft, ob ein neues Sample einen Buffer anschließt
    #     for group in self.buffer_groups:
    #         dist_to_group = self.compute_dtw(x_scaled, group[0])
    #         if dist_to_group <= self.threshold_init:
    #             group.append(x_scaled)

    # ---------------- Final Clustering ----------------
    def finalize_clustering(self):
        """This function finalised the cluster assignment.
        For every sample the shortest DTW distance to a medoid is calculated.
            if < distance_threshold: assigned to cluster
            the cluster numbers are in the same order as the inital clusters with new cluster label numbers added at the end.

            if > distance_threshold: classified as oulier (-1 as label)
        """

        n_samples = len(self.samples)
        n_medoids = len(self.medoids)
        all_distances = np.zeros((n_samples, n_medoids)) # pre-allocate all_distances (for every sample the distance to all medoids is calculated)
        final_labels = np.zeros(n_samples, dtype=int)
        final_min_distances = np.zeros(n_samples)

        # set final threshold based on given min distances
        threshold_used = self.update_distance_threshold_auto(self.min_distances_)
        self.threshold_final = threshold_used
        
        # (re-)assign every sample to cloestest medoid
        for i, x in enumerate(self.samples):
            dists = np.array([self.compute_dtw(x, m) for m in self.medoids]) # calculate distances to all medoids
            all_distances[i, :] = dists
            min_idx = np.argmin(dists)  # check index of shortest distance 
            final_min_distances[i] = dists[min_idx]
            final_labels[i] = min_idx if dists[min_idx] <= threshold_used else -1 # assign sample to closets medoid or outlier (-1)

        # ensure no cluster is smaller than minimal buffer size.
        ## This is unlikely to happen but could be if too many samples from inital cluster are assigned to other new clusters.
        unique_labels, counts = np.unique(final_labels[final_labels != -1], return_counts=True)
        for lbl, cnt in zip(unique_labels, counts):
            if cnt < self.buffer_size:
                final_labels[final_labels == lbl] = -1

        self.labels_final = list(final_labels)
        self.min_distances_final = list(final_min_distances)
        self.final_all_dists = all_distances
