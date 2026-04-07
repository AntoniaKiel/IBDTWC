# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 13:50:38 2025

@author: anton
"""
import numpy as np

def is_mostly_zero(trace, threshold=0.3, atol=1e-3):
    return (np.sum(np.isclose(trace, 0, atol=atol)) / trace.size) > threshold

def is_overlapping(new_times, processed_times, threshold_seconds=1):
    """Check if any event time in new_times overlaps with previously processed times."""
    for new_time in new_times:
        for processed_time in processed_times:
            if abs((new_time - processed_time).total_seconds()) < threshold_seconds:
                return True
    return False
def check_similarity(arr):
    """
    Check how similar the values in a NumPy array are by calculating the absolute differences
    between consecutive elements.
    
    Parameters:
        arr (numpy array): The input array.
    
    Returns:
        dict: Containing the mean absolute difference and maximum absolute difference.
    """
    # Compute the absolute differences between consecutive values
    abs_diff = np.abs(np.diff(arr))
    
    # Summary statistics
    mean_diff = np.mean(abs_diff)  # Mean of the absolute differences
    max_diff = np.max(abs_diff)    # Maximum absolute difference
    
    return {
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "differences": abs_diff  # All absolute differences for inspection
    }

def check_for_noisy_event(trace, n_sections=15):
    """
    Check for noisy events by calculating RMS for different sections and SNR (Signal-to-Noise Ratio).
    
    Parameters:
        trace (numpy array): The input seismic trace.
        n_sections (int): Number of sections to divide the trace into for RMS computation.
        
    Returns:
        dict: Containing RMS values, SNR, and similarity statistics.
    """
    

    
    # Step 1: Divide the trace into n_sections and compute RMS for each section
    section_length = len(trace) // n_sections  # Length of each section
    rms_values = []

    for i in range(n_sections):
        start_idx = i * section_length
        end_idx = (i + 1) * section_length if i < n_sections - 1 else len(trace)  # Ensure last section captures all data
        section = trace[start_idx:end_idx]
        rms = np.sqrt(np.mean(section**2))  # Compute RMS for this section
        rms_values.append(rms)

    # Step 2: Calculate similarity of the RMS values
    similarity = check_similarity(np.array(rms_values))
    
    # Step 3: Find the window with the largest amplitude (signal window)
    peak_amplitude = np.max(np.abs(trace))  # Peak amplitude in the entire trace
    signal_window = np.argmax(np.abs(trace))  # Index of the peak signal (for simplicity, just using this as an example)
    
    # To define a "signal window" more rigorously, you could select a surrounding window
    window_size = len(trace) // 10  # Example: 10% of the trace length
    start_signal = max(0, signal_window - window_size // 2)
    end_signal = min(len(trace), signal_window + window_size // 2)
    
    
    # Step 4: Compute the noise RMS - assuming sections that are not the signal are noise
    noise_sections = [trace[:start_signal], trace[end_signal:]]  # Sections before and after the signal window
    noise_trace = np.concatenate(noise_sections)
    noise_rms = np.sqrt(np.mean(noise_trace**2))  # RMS for the noise section

    # Step 5: Calculate the SNR (Peak amplitude of signal / RMS of noise)
    snr = peak_amplitude / noise_rms if noise_rms > 0 else np.inf  # Avoid division by zero
    
    return {
        "rms_values": rms_values,
        "similarity": similarity,
        "snr": snr
    }