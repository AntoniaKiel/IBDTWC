import numpy as np
from typing import Union
from scipy.signal import hilbert


def windowed_rms(data:np.ndarray, 
                 window_length:Union[int,float], 
                 sampling_rate:Union[int,float]):
    """ Windowed RMS using the envelope (Hilbert transform) for more stable thresholding.

    Parameters
    ----------
    data : np.ndarray of shape (T,1)
        data array where T is the number of time samples
    window_length : Union[int,float]
        time of windowed rms slider [s]
    sampling_rate : Union[int,float]
        sampling rate of data

    Returns
    -------
    rms_value: np.ndarray of shape (T,1)
        contains the windowed RMS values
    """
    T = len(data)

    # envelope via Hilbert
    envelope = np.abs(hilbert(data))

    rms_window_samples = int(window_length * sampling_rate) # turn time[s] into samples
    n_windows = T - rms_window_samples + 1
    rms_station = np.empty(n_windows)
        
    # Calculate RMS for each window for this station
    for i in range(n_windows):
        window = envelope[i:i + rms_window_samples]
        rms_station[i] = np.sqrt(np.mean(window**2))
    
    # Pad to match the original length
    rms_station_padded = np.pad(rms_station, (rms_window_samples // 2, rms_window_samples - 1 - rms_window_samples // 2), mode='edge')
    
    return rms_station_padded