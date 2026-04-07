# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 15:43:50 2025

@author: anton
"""
import pickle
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import resample, butter, filtfilt, hilbert, stft, istft
from scipy.fft import rfft, rfftfreq
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram
from scipy.io.wavfile import read, write
import pywt
load_path='../data/2_clustering_results/good_results/2025_11_14/scRadi3_seed30/n_clusters7/'




print('Loading data')
data_list=np.load(load_path+'BEAMS.npy',allow_pickle=True)
labels_list=np.load(load_path+'labels.npy',allow_pickle=True)
times_list=np.load(load_path+'TIMES.npy',allow_pickle=True)   
min_distances=np.load(load_path+'min_distances.npy',allow_pickle=True)
#%% READ DATA TO LIST AND PLOT EXAMPLES OF EVERY CATEGORY

# data_list=list()
# labels_list=list()
# times_list=list()


# for idx_event, event_name in enumerate(event_names):
#     events_type_list=events[event_name]

# for idx in range(len(events_type_list)):
#     data_list.append(events_type_list[idx]['Z'])
#     times_list.append(events_type_list[idx]['times'])
#     labels_list.append(event_name)
    

        
#%% PLOT ONE EXAMPLE TO LISTEN TO

# --- Parameter ---
lowcut = 1.0
highcut = 10.0           # Seismisches Band
filter_order = 4
gain = 200.0             # Lautstärkeverstärkung
orig_band = (3, 8)      # Original Frequenzbereich
target_band = (80,800) # gewünschter hörbarer Frequenzbereich

# --- Funktionen ---
def bandpass(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def fade_in_out(x, fs, fade_s=0.01):
    n = len(x)
    fade_n = int(min(n//2, max(1, int(fade_s * fs))))
    win = np.ones(n)
    fade = np.linspace(0.0, 1.0, fade_n)
    win[:fade_n] = fade
    win[-fade_n:] = fade[::-1]
    return x * win

def plot_fft(x, fs, title, xlim=8000):
    N = 16384
    Y = np.abs(rfft(x, n=N))
    f = rfftfreq(N, 1.0/fs)
    plt.figure(figsize=(9,3))
    plt.semilogy(f, Y + 1e-20)
    plt.xlim(0, min(xlim, fs/2))
    plt.xlabel("Hz"); plt.title(title); plt.grid(True)
    plt.show()
    
def bandwise_normalize_stft(x, fs, bands,
                            target_db=0.0, max_gain_db=12.0,
                            nperseg=256, noverlap=128, smooth_freq_bins=3,
                            set_gain=None):
    """
    Bandweise Normalisierung + optionale manuelle Gain-Faktoren pro Band.
    set_gain: list of floats, z.B. [1.0, 0.5, 2.0]
              oder None → keine manuelle Verstärkung
    """
    # STFT
    f, t, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   boundary=None, padded=False)
    P = np.abs(Z)**2   # Power

    # Output initialisieren
    Z_adj = Z.copy()

    # Falls set_gain gegeben ist → prüfen
    if set_gain is not None:
        if len(set_gain) != len(bands):
            raise ValueError("set_gain length must match number of bands!")

    # Schleife über alle Bänder
    for i, (f_lo, f_hi) in enumerate(bands):
        idx = np.where((f >= f_lo) & (f <= f_hi))[0]
        if len(idx) == 0:
            continue

        # --- Normalisierung ---
        band_power = P[idx].mean()
        if band_power > 0:
            band_db = 10*np.log10(band_power)
            diff = target_db - band_db
            diff = np.clip(diff, -max_gain_db, max_gain_db)
            scale = 10**(diff/20)
        else:
            scale = 1.0

        # --- ggf. manuelles Gain anwenden ---
        if set_gain is not None:
            scale *= set_gain[i]

        # --- auf das Band anwenden ---
        Z_adj[idx, :] *= scale

    # Rücktransformation
    _, x_out = istft(Z_adj, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     boundary=None)

    return x_out

# --- Beispiel: Event auswählen ---
idx_medoids=np.argwhere(min_distances==0) # save medoids separately
print(idx_medoids)
IDX_medoids_all = []
medoids_time = []
medoids_beams = []

medoids_beams=list()
medoids_time=list()

fig,ax=plt.subplots(len(sorted(set(labels_list)))-1,1,figsize=(8,10))
for idx_plot,cluster_label in enumerate(sorted(set(labels_list))):
    if cluster_label == -1:
        continue
    # find the index where this sample is the medoid
    cluster_indices = np.where(labels_list == cluster_label)[0]
    medoid_idx = cluster_indices[np.argmin(min_distances[cluster_indices])]
    IDX_medoids_all.append(medoid_idx)
    medoids_time.append(times_list[medoid_idx])
    medoids_beams.append(data_list[medoid_idx])
    ax[idx_plot-1].plot(times_list[medoid_idx],data_list[medoid_idx])
    
plt.show()

print(IDX_medoids_all)

def psycho_boost(x, fs, boost_low=200, boost_high=1000, factor=1.4):
    """
    Verstärkt psychoakustisch 'schwache' hohe Frequenzen.
    boost_low:  ab welcher Frequenz (Hz) Boost beginnt
    boost_high: maximale Ziel-Frequenz (Hz)
    factor:     maximale Verstärkung
    """
    N = len(x)
    freqs = np.fft.rfftfreq(N, 1/fs)
    X = np.fft.rfft(x)

    # Smooth gain curve
    gain = np.ones_like(freqs)
    mask = freqs > boost_low
    gain[mask] = 1 + (factor - 1) * ((freqs[mask] - boost_low) / (boost_high - boost_low))
    gain[freqs > boost_high] = factor

    X_boosted = X * gain
    return np.fft.irfft(X_boosted, N)

for cluster_idx in range(len(IDX_medoids_all)):
    seismic_data = np.array(medoids_beams[cluster_idx], dtype=np.float64)
    seismic_data /= np.max(np.abs(seismic_data)).flatten()
    plt.plot(seismic_data
             )
    plt.title(labels_list[IDX_medoids_all[cluster_idx]])
    dt = medoids_time[cluster_idx][1]-medoids_time[cluster_idx][0]
    fs_seismic = 1/dt
    
    
    def seismic_to_audio(seismic_data, fs_seismic, orig_band=(3,8), target_band=(80,800),
                         gain=200.0, fade_s=0.02, compressor_low=0.3, compressor_high=0.5,
                         compressor_gain=1.0, log_k=5.0, filename="seismic_output.wav"):
        """
        Konvertiert ein bereits gefiltertes Seismogramm in hörbares Audio
        unter Beibehaltung der Peaks (peak-preserving Compressor).
        """
        # --- Verstärkung + Spike-Schutz ---
        filtered = seismic_data.copy() * gain
        p99 = np.percentile(np.abs(filtered), 99.9)
        filtered = np.clip(filtered, -p99, p99)
        filtered /= np.max(np.abs(filtered))
        filtered=filtered.flatten()
        
        # --- Automatische Ziel-Samplingrate ---
        f_min, f_max = orig_band
        F_min, F_max = target_band
        fs_audio = int(fs_seismic * (F_max / f_max))
        
        # define bands in seismic domain (Hz)
        bands = [ (3.0, 5.0), (5.0, 10.0)]
        nperseg = int(2 * fs_seismic  ) 
        noverlap = 25
        print(nperseg,noverlap,len(seismic_data),filtered.shape)
        # set desired target_db = mean power baseline. You can set target_db relative, here 0 => no absolute change.
        filtered_boosted = bandwise_normalize_stft(filtered, fs_seismic, bands,
                                                   target_db=0.0, max_gain_db=12.0,
                                                   nperseg=nperseg, noverlap=noverlap, smooth_freq_bins=5,
                                                   set_gain=[0.7,2])
        
        n_samples = int(len(filtered) * (fs_audio / fs_seismic))
        resampled = resample(filtered_boosted, n_samples).astype(np.float32)
        resampled /= np.max(np.abs(resampled))  # Stabilisierung nach Resample
        t = np.arange(len(resampled)) / fs_audio
        resampled = resampled.ravel()
        resampled = psycho_boost(resampled, fs_audio,
                         boost_low=200,
                         boost_high=600,
                         factor=1.5)
        # --- Frequenzverschiebung ---
        analytic = hilbert(resampled).flatten()
        current_center = (f_min + f_max)/2.0
        desired_center = (F_min + F_max)/2.0
        freq_shift = desired_center - current_center
        

        analytic_shifted = analytic * np.exp(2j * np.pi * freq_shift * t)
        shifted = np.real(analytic_shifted)
        
        # --- Fade in/out ---
        n = len(shifted)
        fade_n = int(min(n//2, max(1, int(fade_s * fs_audio))))
        win = np.ones(n)
        fade = np.linspace(0.0, 1.0, fade_n)
        win[:fade_n] = fade
        win[-fade_n:] = fade[::-1]
        shifted *= win
        shifted -= np.mean(shifted)
        
        # --- Peak-preserving Compressor ---
        def peak_preserving_compressor(x, low_thresh=0.05, high_thresh=0.6, gain=2.0):
            x_out = np.copy(x)
            abs_x = np.abs(x_out)
            mask = abs_x < high_thresh
            x_out[mask] = np.sign(x_out[mask]) * (
                abs_x[mask] * (1 + (gain-1)*(high_thresh-abs_x[mask])/(high_thresh-low_thresh))
            )
            return np.clip(x_out, -1.0, 1.0)
        
        shifted = peak_preserving_compressor(shifted,
                                             low_thresh=compressor_low,
                                             high_thresh=compressor_high,
                                             gain=compressor_gain)
        
        # --- Logarithmische Verstärkung (optional) ---
        if log_k > 0:
            shifted = np.sign(shifted) * np.log1p(log_k * np.abs(shifted)) / np.log1p(log_k)
        

        def soft_knee_compressor(x, threshold=0.6, ratio=4.0, knee=0.2, attack=0.005, release=0.050, fs=44100):
            """
            RMS-basierter Soft-Knee-Kompressor für Audio.
            threshold : Pegel (0–1), ab dem komprimiert wird
            ratio     : Kompressionsverhältnis
            knee      : Weiche Übergangsbreite
            attack    : Attack-Zeit (Sekunden)
            release   : Release-Zeit (Sekunden)
            fs        : Samplingrate
            """
        
            # RMS Detector
            out = np.zeros_like(x)
            gain = 1.0
            rms = 0.0
        
            alpha_a = np.exp(-1.0 / (fs * attack))
            alpha_r = np.exp(-1.0 / (fs * release))
        
            for i, sample in enumerate(x):
                # RMS tracking
                rms = alpha_a * rms + (1 - alpha_a) * sample**2
        
                level = np.sqrt(rms)
        
                # Soft knee region
                if level < threshold - knee/2:
                    desired_gain = 1.0
                elif level > threshold + knee/2:
                    desired_gain = (level/threshold)**(1/ratio - 1)
                else:
                    # Soft knee transition
                    t = (level - (threshold - knee/2)) / knee
                    comp_gain = (level/threshold)**(1/ratio - 1)
                    desired_gain = (1 - t) * 1.0 + t * comp_gain
        
                # Smooth attack & release
                if desired_gain < gain:
                    gain = alpha_a * gain + (1 - alpha_a) * desired_gain
                else:
                    gain = alpha_r * gain + (1 - alpha_r) * desired_gain
        
                out[i] = sample * gain
        
            return out
        
        # Nutzung:
        shifted = soft_knee_compressor(
            shifted,
            threshold=0.5,   # ab hier wird komprimiert
            ratio=4.0,       # mittlere Kompression
            knee=0.3,        # weicher Übergang
            attack=0.002,    # schnell → gut für Transienten
            release=0.050,   # langsam → natürlicher Klang
            fs=fs_audio
        )
        # --- Endnormierung ---
        #shifted /= np.max(np.abs(shifted))
        
        # --- WAV schreiben ---
        write(filename, fs_audio, np.int16(shifted * 32767))
        print(f"✅ WAV gespeichert: {filename}")
        
        # --- Plot ---
        plt.figure(figsize=(12,3))
        plt.plot(shifted)
        plt.title("Shifted & Peak-preserved Audio Signal")
        plt.show()
        
        plt.plot(shifted)
    
        # --- Diagnose-Plots ---
        plot_fft(resampled, fs_audio, "FFT vor Shift (resampled)")
        plot_fft(shifted, fs_audio, "FFT nach Shift (transposed)")
        
        return shifted, fs_audio
    
    audio_path=f"seismic_transposed_{int((target_band[0]+target_band[1])/2)}_Hz_ice_{cluster_idx}.wav"
    shifted_audio, fs_audio = seismic_to_audio(
        seismic_data,
        fs_seismic,
        orig_band=(3,8),
        target_band=(80,800),
        gain=200.0,
        fade_s=0.02,
        compressor_low=0.35,
        compressor_high=0.5,
        compressor_gain=1.0,
        log_k=5.0,
        filename=audio_path
    )
    
    
    
    
    #%% Animate video
    
    # === Eingaben ===
    fs_audio, audio_data = read(audio_path)
    audio_data = audio_data.astype(np.float32) / 32767.0  # normalisiert
    
    # Wenn nötig: Zeitvektor
    t_audio = np.arange(len(audio_data)) / fs_audio
    
    
    # --- Seismisches Signal vorbereiten (optional rohdaten oder bandpass) ---
    seismic_plot = seismic_data  # oder filtered_data
    t_seismic = np.arange(len(seismic_plot)) / fs_seismic
    
    # CWT berechnen
    # --- Wavelet-Parameter ---
    wavelet_name = 'cmor1-5'   # komplexer Morlet (B-C), hier B=1.5, C=1.0
    wavelet = wavelet_name
    
    # Frequenzbereich, den du darstellen willst (Hz)
    f_min = 1   # z.B. 0.5 Hz
    f_max = 10.0  # z.B. 20 Hz
    n_f = 500     # Anzahl Frequenzpunkte (feinere Auflösung = größerer Rechenaufwand)
    
    freqs = np.linspace(f_min, f_max, n_f)
    central_freq = pywt.central_frequency(wavelet)  # zentralfrequenz des Wavelet
    scales = central_freq / (freqs * dt)
    # CWT
    coeffs, freqs_out = pywt.cwt(seismic_data.squeeze(), scales, wavelet, sampling_period=dt)
    
    # Leistung & dB
    power = np.abs(coeffs)**2
    power = np.squeeze(power)  # entfernt evtl. unnötige Dimensionen
    power_db = 10 * np.log10(power + 1e-16)
    t_grid, f_grid = np.meshgrid(t_seismic, freqs)
    
    
    # --- Figure & Subplots ---
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6))
    
    # Obere Achse: Seismik
    ax1.plot(t_seismic, seismic_plot, color='blue')
    line1 = ax1.axvline(x=0, color='red', linewidth=2)
    ax1.set_ylabel("amplitude [counts]")
    #ax1.set_title("Seismisches Signal")
    ax1.set_xlim(t_seismic[0],t_seismic[-1])
    
    # Untere Achse: Spektrogramm
    im = ax2.pcolormesh(t_grid, f_grid, power_db, cmap='magma', shading='auto')
    line2 = ax2.axvline(x=0, color='red', linewidth=2)
    ax2.set_ylabel("frequency [Hz]")
    ax2.set_xlabel("time [s]")
    ax2.set_ylim(3, 8)
    ax2.set_xlim(t_seismic[0],t_seismic[-1])
    #ax2.set_title("Spektrogramm Audio")
    
    cbar = fig.colorbar(
        im,
        ax=[ax1, ax2],
        orientation="horizontal",
        pad=0.15,     # Abstand zu den Achsen
        fraction=0.05 # Höhe der Colorbar (kleiner = dünner)
    )
    cbar.set_label('Leistung [dB]')
    
    # --- Animation Funktion ---
    def animate(frame):
        t_pos = frame / fps  # Zeit in Sekunden
        line1.set_xdata([t_pos, t_pos])
        line2.set_xdata([t_pos, t_pos])
        return line1, line2
    
    # --- Video Parameter ---
    duration = t_audio[-1]             # Gesamtdauer Audio
    fps = 30                           # Frames pro Sekunde
    n_frames = int(duration * fps)
    
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, blit=True)
    
    # --- Video speichern ---
    anim.save(f"seismic_video_ice_{cluster_idx}.mp4", fps=fps, dpi=150, codec='libx264')
    
    plt.close(fig)
    print("✅ Video gespeichert: seismic_video.mp4")
