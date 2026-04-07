


import os
#os.environ["OPENBLAS_NUM_THREADS"] = "8"
#os.environ["MKL_NUM_THREADS"] = "8"
#os.environ["NUMEXPR_NUM_THREADS"] = "8"
#os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np
import torch, gc, glob, os
from obspy.signal.util import util_geo_km
from itertools import product
from src.beamforming.beamforming_utils import normalize_minus_one_one, shift_signal
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
import psutil
import logging

import warnings

# Nur UserWarnings von torch.tensor unterdrücken
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="To copy construct from a tensor, it is recommended.*"
)

def beamforming_events(
    event_path,
    station_path,
    fmin,
    fmax,
    device,
    backazimuth_spacing=2,
    slowness_min=0,
    slowness_max=0.5,
    slowness_spacing=1e-2
):
    """
    Clean beamforming function: returns a flat dictionary with ready-to-plot results.
    """

    # --- LOAD STATIONS ---
    sta_coord = np.load(str(station_path) + '/station_coord.npy')
    coordinates = torch.tensor(sta_coord.T, device=device)

    # Convert to cartesian (km)
    coordinates_cartesian = torch.tensor([
        util_geo_km(coordinates[0,0].cpu(), coordinates[0,1].cpu(), sta[0], sta[1])
        for sta in coordinates.cpu()
    ], device=device)

    # --- CREATE SLOWNESS SPACE ---
    azs = torch.arange(0, 2*np.pi, backazimuth_spacing * 2 * np.pi / 360, device=device)
    slows = torch.arange(slowness_min, slowness_max, slowness_spacing, device=device)
    slowness_space = torch.tensor(
        [(torch.cos(az) * s, torch.sin(az) * s) for az, s in product(azs, slows)],
        device=device
    )

    # --- LOAD WAVEFORMS ---
    data_file = 'data_red.npy'
    time_file = 'times.npy'
    data = np.load(f"{event_path}/{data_file}")
    times_date = np.load(f"{event_path}/{time_file}", allow_pickle=True)

    waveforms = torch.tensor(np.array([normalize_minus_one_one(y, device) for y in data]), device=device)
    sampling_rate = 1 / (times_date[1] - times_date[0]).total_seconds()
    times_rel = torch.tensor([(dt - times_date[0]).total_seconds() for dt in times_date], device=device)

    # --- BEAMFORMING SETUP ---
    window_length = times_rel[-1]
    reference_point = coordinates_cartesian[0]

    # --- PREPARE WAVEFORMS ---
    waveforms_prep = torch.stack([
        normalize_minus_one_one(y, device=device)
        for y in waveforms
    ]).to(torch.float32)

    # FFT
    data_spectra_all = torch.fft.fft(waveforms_prep, dim=-1).to(torch.complex64)

    # FFT LENGTH
    nfft = data_spectra_all.shape[-1]   # THIS is 1344

    # CORRECT FREQUENCY AXIS
    f = torch.fft.fftfreq(nfft, d=1/sampling_rate).to(device)

    # FREQUENCY MASK
    freq_mask = (f > fmin) & (f < fmax)

    # APPLY MASK
    f_limited = f[freq_mask]
    omega_limited = 2 * np.pi * f_limited
    omega_limited = omega_limited.to(torch.float32)
    data_spectra_all_limited = data_spectra_all[:, freq_mask]

    # --- BEAMFORMING CORE ---
    # --- BEAMFORMING CORE (memory-safe) ---
    num_s = slowness_space.shape[0]
    batch = 256   # oder größer, wenn GPU mehr kann
    beampower_list = []
    
    # Precompute data cross-spectrum once
    K = torch.einsum("iw, jw -> ijw", data_spectra_all_limited, torch.conj(data_spectra_all_limited))
    diag = torch.arange(K.shape[0], device=device)
    K[diag, diag] = 0
    K = K.to(torch.complex64)
    
    for start in range(0, num_s, batch):
    
        end = min(start + batch, num_s)
        s_batch = slowness_space[start:end]
    
        trav = torch.einsum("nx, sx -> sn", reference_point - coordinates_cartesian, s_batch)
    
        synth = torch.exp(-1j * torch.einsum("sn, w -> snw", trav, omega_limited)).to(torch.complex64)
    
        # compute beampower WITHOUT building S
        # S_ijw = synth_iw * conj(synth_jw)
        power = torch.einsum("siw, sjw, ijw -> s", synth, torch.conj(synth), K).real
    
        beampower_list.append(power)
    
    beampower = torch.cat(beampower_list, dim=0)

    # --- FIND MAXIMUM ---
    idx_best = torch.argmax(beampower)
    slowness_best = slowness_space[idx_best]
    s_best = 1 / torch.linalg.norm(slowness_best)  # convert to apparent velocity (m/s)
    sx_best, sy_best = slowness_best[0].item(), slowness_best[1].item()
    backazimuth =  ((np.arctan2(sx_best, sy_best) - np.pi) % (2*np.pi) * 180) /np.pi  # backazimuth in deg


    # --- SHIFT AND SUM TRACES ---
    delays = torch.matmul(coordinates_cartesian, slowness_best)
    shifted_traces = np.zeros_like(waveforms.cpu().numpy())
    for i in range(len(waveforms)):
        shifted_traces[i] = normalize_minus_one_one(
            shift_signal(
                waveforms[i].cpu(), times_rel.cpu(), sampling_rate, delays[i].item(), device=device
            ).numpy(), device=device
        )
    beam = shifted_traces.mean(axis=0)

    # --- PREPARE GRID FOR POLAR PLOTS ---
    sx_grid = slowness_space[:,0].cpu().numpy()
    sy_grid = slowness_space[:,1].cpu().numpy()
    beam_power_grid = beampower.cpu().numpy()

    # Polarkoordinaten für den Plot
    r_grid = np.sqrt(sx_grid**2 + sy_grid**2)
    theta_grid = (np.arctan2(sx_grid, sy_grid) - np.pi) % (2*np.pi)  # Backazimuth 0-2π
    theta_deg = theta_grid * 180/np.pi


    # --- BUILD RESULT DICT ---
    result = {
        "backazimuth": backazimuth,
        "velocity": s_best,
        "sx": sx_best,
        "sy": sy_best,

        "beam": beam,
        "shifted_traces": shifted_traces,
        "times_date": times_date,

        "r_grid": r_grid,
        "theta_grid": theta_grid,
        "theta_deg": theta_deg,
        "beam_power": beam_power_grid
    }

    # --- SAVE ---
    np.save(f"{event_path}/BEAM_results_clean.npy", result)

    return result




def print_memory(prefix=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024**2)  # RSS in MB
    print(f"[{now}] {prefix}Memory usage: {mem_mb:.1f} MB", flush=True)

def process_single_event(event_path, metadata_path, device,
                         fmin, fmax, slowness_min, slowness_max,
                         slowness_spacing, backazimuth_spacing, plot_beamforming=False):
    # Daten laden
    data_red = np.load(os.path.join(event_path, 'data_red.npy'), allow_pickle=True)
    #data_red_15s = np.load(os.path.join(event_path, 'data_red_15s.npy'), allow_pickle=True)
    times = np.load(os.path.join(event_path, 'times.npy'), allow_pickle=True)
    #times_15s = np.load(os.path.join(event_path, 'times_15s.npy'), allow_pickle=True)
    
    
    # Dauerfilter
    if (times[-1] - times[0]).total_seconds() > 80:
        return event_path, "skipped"  # überspringen

    # Beamforming
    RESULTS_beam = beamforming_events(event_path, metadata_path, fmin, fmax,device=device,
                                     backazimuth_spacing=backazimuth_spacing,
                                     slowness_min=slowness_min, slowness_max=slowness_max,
                                     slowness_spacing=slowness_spacing)

    # RESULTS_beam_15s = beamforming_events(event_path, metadata_path, fmin, fmax, device=device,
    #                                      backazimuth_spacing=backazimuth_spacing,
    #                                      slowness_min=slowness_min, slowness_max=slowness_max,
    #                                      slowness_spacing=slowness_spacing, cut15s=True)
    
    if plot_beamforming:
        fig, ax = plt.subplots(len(data_red)+1, 2, figsize=(12,6), sharex='row', dpi=200)
        for idx_sta in range(len(data_red)):
            ax[idx_sta][0].plot(times, data_red[idx_sta], c='k')
            #ax[idx_sta][1].plot(times_15s, data_red_15s[idx_sta], c='k')
        ax[-1][0].plot(RESULTS_beam[0]['times_date'], np.mean(RESULTS_beam[0]['shifted_traces'], axis=0), color='blue')
        #ax[-1][1].plot(RESULTS_beam_15s[0]['times_date'], np.mean(RESULTS_beam_15s[0]['shifted_traces'], axis=0), color='blue')
        plt.savefig(os.path.join(event_path, 'seismogram.png'))
        plt.close(fig)
    


def process_single_event_wrapper(args):
    #print_memory()  # print current memory usage
    return process_single_event(*args)

def beamforming_pipeline(
        t_start, t_end,
        fmin, fmax, device,
        slowness_min, slowness_max, slowness_spacing, backazimuth_spacing, n_processes,
        path_events, metadata_path, re_process=False, plot_beamforming=False,
        ):

    t_start_dt = datetime.fromisoformat(t_start)
    t_end_dt   = datetime.fromisoformat(t_end)
    
    events_all = glob.glob(str(path_events) + '/*/month*/*/event*')
    
    if not re_process:
        events_list = [
            event for event in events_all
            if not any(glob.glob(os.path.join(event, 'BEAM_*.npy')))
        ]
    else:
        events_list = events_all.copy()
    
    def extract_event_date(event_path):
        basename = os.path.basename(os.path.dirname(event_path))
        return datetime.fromisoformat(basename)
    
    # Filter nach Zeitraum
    events_list = [
        e for e in events_list
        if t_start_dt <= extract_event_date(e) <= t_end_dt
        and os.path.isfile(os.path.join(e, 'data_red.npy'))
    ]
    logging.info(f"Starting calculation for {t_start} until {t_end}, {len(events_list)} unprocessed events")

    # Prüfen, ob GPU verfügbar
    device_type = "cuda" if device.type.startswith("cuda") else "cpu"
    logging.info(f"Running on device: {device}")

    # Argumente vorbereiten
    args = [
        (event, metadata_path,device, fmin, fmax,  slowness_min, slowness_max,
         slowness_spacing, backazimuth_spacing, plot_beamforming)
        for event in events_list
    ]

    if device_type == "cuda":
        # GPU: normale for-loop
        for args_single in tqdm(args, desc="Beamforming on GPU"):
            process_single_event_wrapper(args_single)
    else:
        # CPU: zwingend CPU als device in args
        cpu_args = [
            (event, metadata_path,torch.device("cpu"), fmin, fmax,  slowness_min, slowness_max,
             slowness_spacing,backazimuth_spacing,plot_beamforming)
            for event in events_list
        ]
        with Pool(processes=n_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(process_single_event_wrapper, cpu_args),
                total=len(cpu_args),
                desc="Beamforming on CPU (multiprocessing)"
            ):
                pass




























