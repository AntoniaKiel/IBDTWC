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
from scipy.signal import resample, butter, filtfilt, hilbert
from scipy.fft import rfft, rfftfreq
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram
from scipy.io.wavfile import read, write
import pywt
load_path_list=glob.glob('../../../DATA/MVO/ASCII/*')


event_names=['hybrid','longperiod','lr','mi','rockfall','tremor']
component_list=['Z','N','E']


print('Loading dictonary')
with open('../data/labeled_clustering/event_dictionary.pkl', 'rb') as f:
    events = pickle.load(f)
    
#%% READ DATA TO LIST AND PLOT EXAMPLES OF EVERY CATEGORY

data_list=list()
labels_list=list()
times_list=list()


for idx_event, event_name in enumerate(event_names):
    events_type_list=events[event_name]

for idx in range(len(events_type_list)):
    data_list.append(events_type_list[idx]['Z'])
    times_list.append(events_type_list[idx]['times'])
    labels_list.append(event_name)
    

        
#%% PLOT ONE EXAMPLE TO LISTEN TO

# --- Parameter ---
lowcut = 1.0
highcut = 15.0           # Seismisches Band
filter_order = 4
gain = 200.0             # Lautstärkeverstärkung
orig_band = (1, 10)      # Original Frequenzbereich
target_band = (80,1000) # gewünschter hörbarer Frequenzbereich

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

# --- Beispiel: Event auswählen ---
ex_idx = 3
seismic_data = np.array(data_list[ex_idx], dtype=np.float64)
dt = events['hybrid'][0]['dt']
fs_seismic = 1/dt

print(seismic_data.shape)


# --- Bandpass & Gain ---
filtered = bandpass(seismic_data, fs_seismic, lowcut, highcut, order=filter_order)
filtered *= gain

# --- Normalisierung vor Resample ---
if np.max(np.abs(filtered)) > 0:
    filtered = filtered / np.max(np.abs(filtered))

# --- Berechne automatische Ziel-Samplingrate ---
f_min, f_max = orig_band
F_min, F_max = target_band
fs_audio = int(fs_seismic * (F_max / f_max))  # so wird das Originalband auf Target-Band skaliert
print(f"Resampling von {fs_seismic:.2f} Hz auf {fs_audio} Hz")

# --- Resample ---
n_samples = int(len(filtered) * (fs_audio / fs_seismic))
resampled = resample(filtered, n_samples).astype(np.float32)
t = np.arange(len(resampled)) / fs_audio

# --- Frequenzverschiebung (Heterodyning) ---
analytic = hilbert(resampled)
current_center = (lowcut + highcut) / 2.0
desired_center_hz = (F_min + F_max)/2.0   # automatisch Zielmittelpunkt
freq_shift = desired_center_hz - current_center
print(f"Shift: {freq_shift:.1f} Hz -> target center ~ {desired_center_hz:.1f} Hz")
analytic_shifted = analytic * np.exp(2j * np.pi * freq_shift * t)
shifted = np.real(analytic_shifted)

# --- Funktion: Abschnittsweise RMS-Normalisierung ---
def peak_preserving_compressor(x, low_thresh=0.05, high_thresh=0.6, gain=2.0):
    """
    Sanfte Dynamik: leise Abschnitte werden angehoben, starke Peaks bleiben dominant
    low_thresh  : untere Grenze für Verstärkung (z.B. Hintergrundrauschen)
    high_thresh : obere Grenze, ab der Peaks unberührt bleiben
    gain        : Verstärkungsfaktor für leise Signale
    """
    x_out = np.copy(x)
    abs_x = np.abs(x_out)

    # Leise Abschnitte < high_thresh anheben, linear skaliert
    mask = abs_x < high_thresh
    x_out[mask] = np.sign(x_out[mask]) * (
        abs_x[mask] * (1 + (gain-1)*(high_thresh-abs_x[mask])/(high_thresh-low_thresh))
    )

    # Peaks über high_thresh bleiben unberührt (nur Clip zum Schutz)
    x_out = np.clip(x_out, -1.0, 1.0)
    return x_out

# --- Normalisierung & Fade nach Shift ---
shifted -= np.mean(shifted)
shifted = fade_in_out(shifted, fs_audio, fade_s=0.02)

plt.plot(shifted)
plt.show()
# --- Abschnittsweise RMS-Normalisierung ---
shifted = peak_preserving_compressor(shifted, low_thresh=0.35,high_thresh=0.5,gain=1.0)

# --- Logarithmische Verstärkung (optional) ---
k = 5.0  # Stärke der Kontrastverstärkung, kann getestet werden
shifted = np.sign(shifted) * np.log1p(k * np.abs(shifted)) / np.log1p(k)

# --- abschließende globale Normierung auf [-1,1] ---
if np.max(np.abs(shifted)) > 0:
    shifted = shifted / np.max(np.abs(shifted))

# --- WAV schreiben ---
write(f"seismic_transposed_{int(desired_center_hz)}_Hz.wav", fs_audio, np.int16(shifted * 32767))
print(f"✅ WAV geschrieben: seismic_transposed_{int(desired_center_hz)}_Hz.wav")

plt.plot(shifted)

# --- Diagnose-Plots ---
plot_fft(resampled, fs_audio, "FFT vor Shift (resampled)")
plot_fft(shifted, fs_audio, "FFT nach Shift (transposed)")


#%% Animate video

# === Eingaben ===
fs_audio, audio_data = read("seismic_transposed_440_Hz.wav")
audio_data = audio_data.astype(np.float32) / 32767.0  # normalisiert

# Wenn nötig: Zeitvektor
t_audio = np.arange(len(audio_data)) / fs_audio


# --- Seismisches Signal vorbereiten (optional rohdaten oder bandpass) ---
seismic_plot = seismic_data  # oder filtered_data
t_seismic = np.arange(len(seismic_plot)) / fs_seismic

# CWT berechnen
# --- Wavelet-Parameter ---
wavelet_name = 'cmor1.5-1.0'   # komplexer Morlet (B-C), hier B=1.5, C=1.0
wavelet = wavelet_name

# Frequenzbereich, den du darstellen willst (Hz)
f_min = 1   # z.B. 0.5 Hz
f_max = 10.0  # z.B. 20 Hz
n_f = 500     # Anzahl Frequenzpunkte (feinere Auflösung = größerer Rechenaufwand)

freqs = np.linspace(f_min, f_max, n_f)
central_freq = pywt.central_frequency(wavelet)  # zentralfrequenz des Wavelet
scales = central_freq / (freqs * dt)
coeffs, freqs_out = pywt.cwt(seismic_data, scales, wavelet, sampling_period=dt)
t_grid, f_grid = np.meshgrid(t_seismic, freqs)

power = np.abs(coeffs) ** 2   # Skalarmatrix (Leistung)
power_db = 10 * np.log10(power + 1e-16)  # in dB, kleines offset um log(0) zu vermeiden

# --- Figure & Subplots ---
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6))

# Obere Achse: Seismik
ax1.plot(t_seismic, seismic_plot, color='blue')
line1 = ax1.axvline(x=0, color='red', linewidth=2)
ax1.set_ylabel("amplitude [counts]")
#ax1.set_title("Seismisches Signal")
ax1.set_xlim(t_seismic[0],t_seismic[-1])

# Untere Achse: Spektrogramm
im = ax2.pcolormesh(t_grid, f_grid, power_db, shading='auto')
line2 = ax2.axvline(x=0, color='red', linewidth=2)
ax2.set_ylabel("frequency [Hz]")
ax2.set_xlabel("time [s]")
ax2.set_ylim(1, 10)
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
anim.save("seismic_video.mp4", fps=fps, dpi=150, codec='libx264')

plt.close(fig)
print("✅ Video gespeichert: seismic_video.mp4")
