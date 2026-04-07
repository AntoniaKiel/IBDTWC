# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:20:57 2025

@author: anton
"""
import torch
import numpy as np

def normalize_minus_one_one(arr, device=None):
    """
    Normalizes array/tensor to range [-1, 1].
    Works for both NumPy arrays and Torch tensors.
    """

    # --- Torch Tensor (CPU or GPU) ---
    if isinstance(arr, torch.Tensor):
        min_val = arr.min()
        max_val = arr.max()
        if (max_val - min_val) == 0:
            return torch.zeros_like(arr, device=arr.device)
        return 2 * (arr - min_val) / (max_val - min_val) - 1

    # --- NumPy array ---
    else:
        arr = np.asarray(arr)
        min_val = arr.min()
        max_val = arr.max()
        if (max_val - min_val) == 0:
            return np.zeros_like(arr)
        return 2 * (arr - min_val) / (max_val - min_val) - 1


def shift_signal(signal, t_signal, sampling_rate, t_shift, device='cpu'):
    """
    Shift a 1D signal in time using Fourier-domain multiplication.

    Parameters:
    - signal: np.ndarray or torch.Tensor
    Input signal.
    - t_signal: array-like
    Time vector corresponding to the signal.
    - sampling_rate: float
    Sampling rate of the signal in Hz.
    - t_shift: float
    Time shift in seconds (positive shifts the signal to later times).
    - device: str or torch.device
    Device where computation should occur.

    Returns:
    - torch.Tensor
    Time-shifted signal on the specified device.
    """
    signal = torch.as_tensor(signal, device=device, dtype=torch.float32)
    N = len(t_signal)
    dt = 1.0 / sampling_rate
    omega_signal = 2 * torch.pi * torch.fft.fftfreq(N, d=dt).to(device)
    signal_fft = torch.fft.fft(signal)
    shifted_spectra_signal = signal_fft * torch.exp(-1j * omega_signal * t_shift)
    signal_shifted = torch.fft.ifft(shifted_spectra_signal).real  # Keep real part
    return signal_shifted

