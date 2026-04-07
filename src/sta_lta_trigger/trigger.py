import numpy as np  
import pandas as pd 
import logging
from typing import Union
from scipy.ndimage import median_filter
from src.sta_lta_trigger.utils_sta_lta import normalize_minus_std_to_std, merge_sublists, windowed_rms

def sta_compute_ratio( 
    y:np.ndarray, 
    win_STA:Union[int, float], 
    win_LTA:Union[int, float], 
    time_vector:list, 
    trigger_threshold_on:Union[int, float],
    trigger_threshold_off=None,         
    dead_time=None,               
    freeze_lta=False              
):
    """This function computes the standard STA/LTA for a given input array 'a' of one .

    Parameters
    ----------
    a : np.array or list
        seismic trace of one station
    win_STA : float / int
        length of STA window [s]
    win_LTA : float / int
        length of LTA window [s]
    time_vector : list of datetime-objects
        list of datetime objects corresponding to input data
    threshold : float
        threshold for triggering STA/LTA ratio (only start if off_threshold is given)
    off_threshold : float or None
        threshold for turgning trigger off, by default None
    dead_time : float / int, optional
        minimum time after event ends before retrigger [s], by default None
    freeze_lta : bool, optional
        if True, stop updating LTA during triggers, by default False

    Returns
    -------
    Tuple
        (sta_lta ratio, related time vector, trigger mask, time offset to raw data due to STA/LTA windows)
    """   
    logging.debug(f"Starting STA/LTA calculation on given input data!")
    logging.debug(f"parameter a of type {type(y)} and shape {y.shape}")
    logging.debug(f"parameter win_STA {win_STA} of type {type(win_STA)}")
    logging.debug(f"parameter win_LTA {win_LTA} of type {type(win_LTA)}")
    logging.debug(f"parameter time_vector of type {type(time_vector)} of length {len(time_vector)}")
    logging.debug(f"parameter threshold_on {trigger_threshold_on} of type {type(trigger_threshold_on)}")
    logging.debug(f"parameter threshold_off {trigger_threshold_off} of type {type(trigger_threshold_off)}")
    logging.debug(f"parameter dead_time {dead_time} of type {type(dead_time)}")

    # --- Setup ---
    # --- Setup ---
    dt = (time_vector[1] - time_vector[0]).total_seconds()
    N = len(y)
    
    nsta = max(1, int(win_STA / dt))
    nlta = max(1, int(win_LTA / dt))
    
    y2 = np.nan_to_num(y)**2

    # --- Kausales STA ---
    sta = np.zeros(N)
    csum = 0.0
    for i in range(N):
        csum += y2[i]
        if i >= nsta:
            csum -= y2[i - nsta]
            sta[i] = csum / nsta
        else:
            sta[i] = csum / (i + 1)

    # --- Kausales LTA (mit optionalem Freeze) ---
    lta = np.zeros(N)
    csum = 0.0
    in_trig = False
    for i in range(N):
        if freeze_lta and in_trig:
            # LTA nicht updaten
            lta[i] = lta[i-1] if i > 0 else y2[0]
        else:
            csum += y2[i]
            if i >= nlta:
                csum -= y2[i - nlta]
                lta[i] = csum / nlta
            else:
                lta[i] = csum / (i + 1)
        
        # Trigger-Status nur für Freeze-LTA notwendig
        if freeze_lta and trigger_threshold_off is not None:
            r = sta[i] / max(lta[i], np.finfo(float).tiny)
            if in_trig and r < trigger_threshold_off:
                in_trig = False
            elif not in_trig and r > trigger_threshold_on:
                in_trig = True

    lta_safe = np.maximum(lta, np.finfo(float).tiny)
    
    sta_lta_raw = sta / lta_safe

    # --- Rolling MAD Normalisierung via Pandas ---
    # roll_N = int(300 / dt)  # ca. 5 Minuten
    # if roll_N < nlta * 2:
    #     roll_N = nlta * 4
    
    # series = pd.Series(sta_lta_raw)
    # median_roll = series.rolling(window=roll_N, min_periods=1).median()
    # mad_roll = series.rolling(window=roll_N, min_periods=1)\
    #                  .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    
    # sigma = 1.4826 * np.maximum(mad_roll.to_numpy(), 1e-12)
    # sta_lta = (sta_lta_raw - median_roll.to_numpy()) / sigma

    # scipy version since this is fasteer

    roll_N = max(nlta*4, int(300/dt))
    median_roll = median_filter(sta_lta_raw, size=roll_N, mode='reflect')
    mad_roll = median_filter(np.abs(sta_lta_raw - median_roll), size=roll_N, mode='reflect')
    eps=1e-6
    sigma = 1.4826 * np.maximum(mad_roll, eps)
    sta_lta = (sta_lta_raw - median_roll) / sigma

    # --- Trigger-Logik ---
    trigger = np.zeros(N, dtype=bool)
    in_trig = False
    ndead = int(dead_time / dt) if dead_time else 0
    dead_counter = 0

    for i in range(N):
        r = sta_lta[i]
        if in_trig:
            if trigger_threshold_off is not None and r < trigger_threshold_off:
                in_trig = False
                dead_counter = ndead
            trigger[i] = True
        else:
            if dead_counter > 0:
                dead_counter -= 1
                continue
            if r > trigger_threshold_on:
                in_trig = True
                trigger[i] = True

    return sta_lta, time_vector, trigger, 0

#%% LOOK AT ALL EVENTS THAT GOT TRIGGERED
def find_consecutive_trues(output_vector,STA_LTA_sampling,t_add_start,t_add_end,add_timewindow=True,merge=True):
    
    add_start=int(t_add_start/STA_LTA_sampling)
    add_end=int(t_add_end/STA_LTA_sampling)
    
    consecutive_windows = []
    current_window = []
    
    

    for i, value in enumerate(output_vector):
        if value:  # If the value is True
            current_window.append(i)  # Add index to current window
        else:
            if current_window:  # If the current window is not empty
                consecutive_windows.append(current_window)  # Save the current window
                current_window = []  # Reset the current window

    # If the vector ends with a True sequence, add the last window
    if current_window:
        consecutive_windows.append(current_window)
    
    if add_timewindow:  # add time before & after trigger
        consecutive_windows_new=list()
        for window in consecutive_windows:
            if len(window) > 1:
                add_list_start=np.arange(int(window[0]-add_start),window[0]).tolist()
                add_list_end=np.arange(window[-1],window[-1]+add_end).tolist()
                new_time_window=add_list_start+window+add_list_end
                
                if all (0 < value < len(output_vector) for value in new_time_window):
                    consecutive_windows_new.append(new_time_window)
                
        consecutive_windows=consecutive_windows_new
    
    # check if there are overlapping windows and if so merge them
    if merge:
        consecutive_windows=merge_sublists(consecutive_windows)

    return consecutive_windows

def find_common_time_windows(detection_lists, 
                             min_station_count=10, 
                             overlap_threshold=0.6,
                             inner_station_indices={0,1,2,3},
                             min_inner_required=3):
    from collections import defaultdict
    
    # (1) Index-to-event map
    index_to_events = defaultdict(list)
    for station_idx, station_events in enumerate(detection_lists):
        for event_idx, event_indices in enumerate(station_events):
            for index in event_indices:
                index_to_events[index].append((station_idx, event_idx))
    
    processed_events = set()
    expanded_windows = []

    for station_idx, station_events in enumerate(detection_lists):
        for event_idx, event_indices in enumerate(station_events):
            if (station_idx, event_idx) in processed_events:
                continue
            
            # (2) Overlaps sammeln
            overlapping_events = set()
            for index in event_indices:
                overlapping_events.update(index_to_events[index])
            
            # (3) Threshold prüfen
            valid_overlapping_events = set()
            for overlap_station, overlap_event in overlapping_events:
                overlap_indices = set(event_indices).intersection(
                    detection_lists[overlap_station][overlap_event]
                )
                smaller_event_size = min(
                    len(event_indices), 
                    len(detection_lists[overlap_station][overlap_event])
                )
                if len(overlap_indices) / smaller_event_size >= overlap_threshold:
                    valid_overlapping_events.add((overlap_station, overlap_event))
            
            # (4) beteiligte Stationen
            stations_involved = {ev[0] for ev in valid_overlapping_events}
            
            # **** NEU: Mindestanzahl innerer Stationen prüfen ****
            inner_hits = len(stations_involved.intersection(inner_station_indices))
            
            if (len(stations_involved) >= min_station_count 
                and inner_hits >= min_inner_required):
                
                # expand
                all_indices = set()
                for ov_st, ov_ev in valid_overlapping_events:
                    all_indices.update(detection_lists[ov_st][ov_ev])
                    processed_events.add((ov_st, ov_ev))
                
                expanded_windows.append(sorted(all_indices))
            
            else:
                processed_events.add((station_idx, event_idx))
    
    return expanded_windows

#%% WRITE A FUNCTION WHICH CAN ZERO OUTLIER TRACES FROM SOME STATIONS
def identify_and_zero_outliers(data, threshold=0.8,taper_frac=None,clipping=None,clipping_factor=4):
    from scipy.signal import correlate
    from scipy.signal.windows import tukey
    
    if taper_frac:
        # Taper data
        # Generate a Tukey window for tapering (adjust alpha for taper percentage)
        alpha = taper_frac  # Fraction of the window to taper (0.1 means 10% tapering on both ends)
        taper = tukey(len(data[0]), alpha)  
    
        # Apply taper to each signal
        data = data.copy() * taper  # Element-wise multiplication
        
    """
    Identify and set to zero traces with cross-correlation below a threshold.

    Parameters:
    - data: np.ndarray
        Array of shape (N, T), where N is the number of stations and T is the length of each trace.
    - threshold: float, optional
        The cross-correlation threshold. Default is 0.8.

    Returns:
    - np.ndarray
        Modified array with outlier traces set to zero.
    """


    if clipping == True:  
        # from scipy.signal import hilbert
        # # Compute the Hilbert transform along the time axis
        # hilbert_transformed = np.abs(hilbert(data, axis=1))
        hilbert_transformed=np.abs(windowed_rms(data,100))
        
        # Calculate the standard deviation of the Hilbert transform (row-wise)
        std_values = np.std(np.abs(hilbert_transformed), axis=1, keepdims=True)  # Shape: (N, 1)
        mean_values = np.mean(np.abs(hilbert_transformed), axis=1, keepdims=True)  # Shape: (N, 1)
               
        
        
        # Identify outliers (where abs(Hilbert transform) > 4 * std_values)
        outliers_mask = np.abs(data) > (clipping_factor * std_values + mean_values) # Shape: (N, T), boolean mask
        
        # Create a copy of data to modify only where outliers are found
        clipped_data = data.copy()
        
        # Replace values in `data` where Hilbert transform exceeds the threshold
        for station_idx in range(data.shape[0]):  # Loop through each station
            threshold_clip = clipping_factor * std_values[station_idx, 0] + mean_values[station_idx] # Scalar threshold for this station
            station_mask = outliers_mask[station_idx]   # Boolean mask for this station
            clipped_data[station_idx, station_mask] = (
                np.sign(data[station_idx, station_mask]) * threshold_clip
            )
            
        
        # Update the original data with the clipped data
        data = clipped_data.copy()
        

            
        
    elif clipping:
        from scipy.signal import hilbert
        
        # Define the clipping percentile
        percentile = clipping  # e.g., 99
        
        # Compute the Hilbert transform along the time axis
        hilbert_transformed = hilbert(np.abs(data), axis=1)
        
        # Calculate the standard deviation of the Hilbert transform (row-wise)
        std_values = np.std(np.abs(hilbert_transformed), axis=1, keepdims=True)  # Shape: (N, 1)
        
        # Calculate the percentile threshold of the Hilbert transform magnitude for each station
        thresholds = np.percentile(np.abs(hilbert_transformed), percentile, axis=1, keepdims=True)  # Shape: (N, 1)
        
        # Identify outliers (Hilbert transform magnitude > std or > percentile threshold)
        outliers_mask = (np.abs(hilbert_transformed) > std_values) & (np.abs(hilbert_transformed) > thresholds)  # Shape: (N, T)
        
        # Create a copy of data to modify only where outliers are found
        clipped_data = data.copy()
        
        # Replace values in `data` where Hilbert transform exceeds both thresholds
        for station_idx in range(data.shape[0]):  # Loop through each station
            hilbert_threshold = thresholds[station_idx, 0]  # Percentile threshold for this station
            station_mask = outliers_mask[station_idx]       # Boolean mask for this station
            
            # Clip data using the percentile threshold
            clipped_data[station_idx, station_mask] = (
                np.sign(data[station_idx, station_mask]) * hilbert_threshold
            )
        
        # Update the original data with the clipped data
        data = clipped_data.copy()

    
    # Number of stations
    num_stations = data.shape[0]
    # Array to store whether a trace is an outlier
    is_outlier = np.zeros(num_stations, dtype=bool)
    

        
    # Loop through each station
    for i in range(num_stations):
        # Cross-correlations of the current trace with all others
        correlations = []
        for j in range(num_stations):
            if i != j:  # Skip self-comparison
                # Compute the normalized cross-correlation
                corr = correlate(normalize_minus_std_to_std(data[i]), normalize_minus_std_to_std(data[j]), mode='full', method='auto')
                # Normalize correlation to [-1, 1]
                corr /= (np.linalg.norm(normalize_minus_std_to_std(data[i])) * np.linalg.norm(normalize_minus_std_to_std(data[j])))
                #corr=stats.pearsonr(data[i],data[j])
                
                # Find the maximum correlation (allowing for time shifts)
                max_corr = np.max(corr)
                correlations.append(max_corr)
            else:
                # Compute the normalized cross-correlation
                corr = correlate(normalize_minus_std_to_std(data[i]), normalize_minus_std_to_std(data[j]), mode='full', method='auto')
                # Normalize correlation to [-1, 1]
                corr /= (np.linalg.norm(normalize_minus_std_to_std(data[i])) * np.linalg.norm(normalize_minus_std_to_std(data[j])))
                #corr=stats.pearsonr(data[i],data[j])
                
                # Find the maximum correlation (allowing for time shifts)
                max_corr = np.max(corr)
                correlations.append(max_corr)        
        # Count how many correlations are above the threshold
        strong_correlations = np.sum(np.array(correlations) >= threshold)
        # Determine if the trace is an outlier
        if strong_correlations < (num_stations - 1) // 2:  # More than half must correlate well
            is_outlier[i] = True

    # Create a copy of the data and set outlier traces to zero
    data_new = data.copy()
    data_new[is_outlier] = 0

    return data_new



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
