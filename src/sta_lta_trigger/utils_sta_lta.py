# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 12:43:39 2025

@author: anton
"""
import numpy as np
import logging

def cosine_similarity_with_nan(data):
    """
    Compute cosine similarity matrix, ignoring NaN values.
    """
    # Mask NaN values
    mask = ~np.isnan(data)
    data_nan_to_zero = np.where(mask, data, 0)  # Replace NaNs with 0 for dot product

    # Compute pairwise dot products
    dot_products = np.dot(data_nan_to_zero, data_nan_to_zero.T)

    # Compute norms (adjusted for NaNs)
    norms = np.sqrt(np.dot(mask, mask.T))  # Number of valid elements in pairwise calculations
    squared_sums = np.sqrt(np.sum(data_nan_to_zero**2, axis=1))
    normalization = np.outer(squared_sums, squared_sums)

    # Compute cosine similarity (adjust for valid elements)
    similarities = np.divide(
        dot_products, normalization,
        out=np.zeros_like(dot_products), where=normalization != 0
    )

    # Set invalid similarities (if norms are 0 due to all NaNs)
    similarities[norms == 0] = np.nan

    return similarities

def normalize_minus_std_to_std(vector):
    """
    Normalize a 1D numpy array such that the range corresponds to [-1, 1], 
    where the original standard deviation maps to -1 and +1.

    Parameters:
    - vector: np.ndarray
        Input 1D array.

    Returns:
    - np.ndarray
        Normalized array in the range [-1, 1], where the original std maps to -1 and +1.
    """
    mean_val = np.mean(vector)
    std_val = np.std(vector)
    
    # Avoid division by zero if standard deviation is zero
    if std_val == 0:
        return np.zeros_like(vector)  # All elements are zero in this case
    
    normalized_vector = (vector - mean_val) / std_val  # Normalize to mean 0, std 1
    return normalized_vector

def find_closest_datetime_index(date_list, target_date):
    # Use the `min` function with a key argument to find the index of the closest datetime
    closest_index = min(range(len(date_list)), key=lambda i: abs(date_list[i] - target_date))
    return closest_index

def merge_sublists(lists):
    merged = True  # To track if we made any merge in the loop
    iteration = 0  # To track the number of iterations

    while merged:
        merged = False  # Reset for this pass through the lists
        new_lists = []
        skip = set()  # Indices of lists to skip as they are already merged
        iteration += 1  # Increment the iteration counter

        for i, lst1 in enumerate(lists):
            if i in skip:  # Skip if already merged
                continue
            for j, lst2 in enumerate(lists):
                if i >= j or j in skip:  # Avoid redundant checks and skip merged lists
                    continue
                if set(lst1) & set(lst2):  # Check for common elements
                    logging.debug(f"Merging sublist {lst1} with {lst2} (common elements: {set(lst1) & set(lst2)})")
                    lst1 = list(set(lst1) | set(lst2))  # Merge the two lists
                    skip.add(j)  # Mark the second list as merged
                    merged = True  # Mark that a merge happened
            new_lists.append(np.unique(lst1))  # Append the merged or unchanged list
        
        lists = new_lists  # Update the lists with merged lists

    return lists

def windowed_rms(data, window_length):
    """
    Calculate windowed RMS for each station in a (N, T) seismic dataset.
    
    Parameters:
    - data: ndarray of shape (N, T) where N is the number of stations, T is the number of time samples
    - window_length: int, the length of the sliding window in samples

    Returns:
    - rms_values: ndarray of shape (N, T) containing the windowed RMS values
    """
    N, T = data.shape
    rms_values = np.empty((N, T))
    
    for station in range(N):
        signal = data[station]
        n_windows = T - window_length + 1
        rms_station = np.empty(n_windows)
        
        # Calculate RMS for each window for this station
        for i in range(n_windows):
            window = signal[i:i + window_length]
            rms_station[i] = np.sqrt(np.mean(window**2))
        
        # Pad to match the original length
        rms_station_padded = np.pad(rms_station, (window_length // 2, window_length - 1 - window_length // 2), mode='edge')
        rms_values[station] = rms_station_padded
    
    return rms_values




    
