# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 08:27:00 2025

@author: anton
"""
import numpy as np   
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle
from functools import partial
import logging
from typing import Union
from concurrent.futures import ThreadPoolExecutor

from src.sta_lta_trigger.utils_event_processing import find_folders_to_process, find_events_to_process,  find_raw_folder_for_event
from src.sta_lta_trigger.utils_sta_lta import find_closest_datetime_index
from src.sta_lta_trigger.trigger import sta_compute_ratio, find_consecutive_trues, find_common_time_windows, identify_and_zero_outliers

def sta_lta_pipeline(
    raw_data_path: Path,
    event_base_path: Path,
    start_date: str,
    end_date: str,
    t_add_start: float,
    t_add_end: float,
    fmin: float,
    fmax: float,
    win_STA: float,
    win_LTA: float,
    dead_time:float,
    trigger_threshold: float,
    threshold_off:float,
    min_stations: int,
    n_kernels=1
    ):
    """
    STA/LTA pipeline: iterates over raw data folders,
    computes STA/LTA triggers, and stores detected events.
    
    Args:
        raw_data_path: Basisordner der Rohdaten
        event_base_path: Basisordner für detected_events
        t_add_start, t_add_end: Zeitfenster vor/nach Trigger
        fmin, fmax: Frequenzfilter
        win_STA, win_LTA: Fensterlängen für STA/LTA
        trigger_threshold: Threshold für Trigger
        min_stations: Min. Stationen, auf denen Trigger auftreten muss
        window_duration: Dauer des geschnittenen Events
    """

    raw_data_path = Path(raw_data_path)
    event_base_path = Path(event_base_path)

    # Ordner filtern, die noch keine Events haben
    folders_to_process = find_folders_to_process(raw_data_path, event_base_path,start_date,end_date)
    logging.info(f"{len(folders_to_process)} folders need processing.")

    n_processes = min(n_kernels, cpu_count())
    # STA / LTA trigger on continuous and raw data    
    # for folder in folders_to_process: 
    #    process_folder_raw(Path(folder))
    if len(folders_to_process) > 0:     # only start multiprocessing if anything is to be done
        # using multiprocessing
        logging.info(f"Multiprocessing of STA/LTA detection on {n_processes} kernels started!")
        
        # Prepare argument tuples
        args = [
            (
                folder,
                t_add_start,
                t_add_end,
                fmin,
                fmax,
                win_STA,
                win_LTA,
                dead_time,
                trigger_threshold,
                threshold_off,
                min_stations,
                event_base_path
            )
            for folder in folders_to_process
        ]
        
        with Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(mp_unwrap_proc_folder, args),
                total=len(args)
            ))
    logging.info('-------- STA/LTA detection done --------------')

    # --- 2 Event-Processing ---
    events_to_process = find_events_to_process(event_base_path,start_date,end_date)
    if len(events_to_process) > 0:
        logging.info('-----------Starting single event processing----------')
        process_func_events = partial(
            process_single_event,
            raw_data_path=raw_data_path,
            trigger_threshold=trigger_threshold,
            win_STA=win_STA,
            win_LTA=win_LTA
        )
        
        with Pool(processes=n_processes) as pool:
            list(tqdm(pool.imap_unordered(process_func_events, events_to_process), total=len(events_to_process)))
    
    
    logging.info("All STA/LTA and event processing finished!")

def mp_unwrap_proc_folder(args):

    """This function is created to enable the use of pool.map(), which takes only one parameter.
    Since the called process_folder_raw function has multiple parameters, the function has to be wraped.

    Parameters
    ----------
    args : tuple
        folder : Path
            path to the raw data containing the daily measurements of data and time as .npy file.
            data is in the shape of (n_stations, len(data)) and time (len(data))
            time is given in unix
        t_add_start : float
            time to add before the STA/LTA is triggered (to catch weaker first arrivals) [s]
        t_add_end : float
            time to add after the STA/LTA was triggered (to catch weaker coda) [s]
        fmin : float
            lower frequency band end [Hz]
        fmax : float
            upper frequency band end [Hz]
        win_STA : float
            length of short-time average window [s]
        win_LTA : float
            length of long-time average window [s]
        dead_time : float
            time the trigger has to be below threshold to get triggered again
        trigger_threshold : float
            threshold the STA/LTA ratio has to exceed for the trigger to be met
        threshold_off: float
            threshold of STA/LTA ratio to fall below to stop event detection
        min_stations : int
            number of stations the STA/LTA has to be triggered on at least to be included in beamforming catalog
        window_duration : float
            duration of the extracted event window
        event_base_path : pathlib.Path
            root directory for storing event output

    Returns
    -------
    Any
         The output of `process_folder_raw()` for the given argument set.
    """
    return proc_day_raw(*args)

def proc_day_raw(folder: Path,
                       t_add_start: float,
                       t_add_end: float,
                       fmin: float,
                       fmax: float,
                       win_STA: float,
                       win_LTA: float,
                       dead_time: float,
                       trigger_threshold: float,
                       threshold_off: float,
                       min_stations: int,
                       event_base_path: Path
                       ):
    """This function processes the raw input files of one folder (day) and applies the STA/LTA trigger to the data.
    It returns an error message in case of processing problems.

    Parameters
    ----------
    folder : Path
        path to the raw data containing the daily measurements of data and time as .npy file.
        data is in the shape of (n_stations, len(data)) and time (len(data))
        time is given in unix
    t_add_start : float
        time to add before the STA/LTA is triggered (to catch weaker first arrivals) [s]
    t_add_end : float
        time to add after the STA/LTA was triggered (to catch weaker coda) [s]
    fmin : float
        lower frequency band end [Hz]
    fmax : float
        upper frequency band end [Hz]
    win_STA : float
        length of short-time average window [s]
    win_LTA : float
        length of long-time average window [s]
    dead_time : float
        time the trigger has to be below threshold to get triggered again
    trigger_threshold : float
        threshold the STA/LTA ratio has to exceed for the trigger to be met
    threshold_off: float
        threshold of STA/LTA ratio to fall below to stop the event
    min_stations : int
        number of stations the STA/LTA has to be triggered on at least to be included in beamforming catalog
    window_duration : float
        duration of the extracted event window
    event_base_path : pathlib.Path
        root directory for storing event output
    """
    logging.debug(f"Starting STA/LTA of {folder}")
    try:
        # load data and time 
        data = np.load(folder / 'data.npy')
        t_unix = np.load(folder / 'time.npy')
        t = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in t_unix]

        # get t_start and t_end automatically from folder name
        # format: .../'YYYY-MM-DDTHH:MM:SS_YYYY-MM-DDTHH:MM:SS'
        folder_name = folder.name  
        t_start, _ = folder_name.split('_')

        # Events erstellen
        create_events(
            time=t,
            y=data,
            t_start=t_start,
            t_add_start=t_add_start,
            t_add_end=t_add_end,
            win_STA=win_STA,
            win_LTA=win_LTA,
            dead_time=dead_time,
            trigger_threshold_on=trigger_threshold,
            trigger_threshold_off=threshold_off,
            min_stations=min_stations,
            save_path_base=event_base_path
        )

    except Exception as e:
        logging.info(f"Error processing folder {folder}: {e}")

def create_events(
    time:list, y:np.ndarray, t_start:str,
    t_add_start: Union[int, float], t_add_end: Union[int, float],
    win_STA:Union[int, float], win_LTA:Union[int, float],dead_time:Union[int, float],
    trigger_threshold_on:Union[int, float], 
    trigger_threshold_off:Union[int, float],
    min_stations:int,
    save_path_base: Path
    ):
    """_summary_

    Parameters
    ----------
    time : list
        time vector related to seismic data
    y : np.array
        seismic data of all n stations of shape (n,len(time))
    t_start : str
        start time of day to process (YYYY-MM-DDTHH-MM-SS)
    t_add_start : int / float
        time to add before trigger is met [s]
    t_add_end : int / float
        time to add at the end of event afteer trigger is stopped [s]
    win_STA : int / float
        time of short time average window [s]
    win_LTA : _type_
        time of long time average window [s]
    dead_time : int / float
        time after one trigger during which a new event cannot be triggered
    trigger_threshold_on : int / float
        STA/LTA ratio has to exceed this value to start event detection
    trigger_threshold_off: int / float
        STA/LTA ratio has to fall below this value to stop event detection
    min_stations : int
        number of stations the event has to be detected on
    save_path_base : Path
        path to folder to save the data in

    saves the following parameters:

    triggers_all : list of np.ndarray(bool)
        One boolean trigger mask per station.
        Each entry is an array aligned with the STA/LTA output of that station,
        where True indicates that the STA/LTA ratio exceeded the trigger threshold.
    
    """
    logging.debug("=== create_events() started ===")

    # log types and shapes
    logging.debug(f"type(time):{type(time)}, len={len(time)}, values in {type(time[0])}")
    logging.debug(f"type(y):{type(y)}, shape={getattr(y, 'shape', None)}")
    logging.debug(f"t_start: {t_start} ({type(t_start)})")

    logging.debug(f"t_add_start: {t_add_start}, t_add_end: {t_add_end}")
    logging.debug(f"win_STA={win_STA}, win_LTA={win_LTA}, dead_time={dead_time}")
    logging.debug(f"trigger_threshold={trigger_threshold_on}, threshold_off={trigger_threshold_off}")
    logging.debug(f"min_stations={min_stations}")
    logging.debug(f"save_path_base: {save_path_base}")    

    # calculate STA/LTA for every station with threading to decrease computaion time
    def worker(idx):
        # Berechne STA/LTA für Station idx
        return sta_compute_ratio(
            y=y[idx],
            win_STA=win_STA,
            win_LTA=win_LTA,
            time_vector=time,
            trigger_threshold_on=trigger_threshold_on,
            trigger_threshold_off=trigger_threshold_off,
            dead_time=dead_time,
            freeze_lta=False
        )

    # ThreadPool für 16 Stationen (oder so viele, wie du hast)
    with ThreadPoolExecutor(max_workers=len(y)) as executor:
        results = list(executor.map(worker, range(len(y))))

    # Ergebnisse zusammensetzen
    sta_lta_all_stations = np.vstack([r[0] for r in results])
    trigger_mask_all_stations = np.vstack([r[2] for r in results])
    sta_lta_time = results[0][1]
    offset_to_raw=results[0][3]


    # for idx in range(len(y)):
    #     sta_lta, sta_lta_time, trigger_mask, offset_to_raw = sta_compute_ratio(
    #         y=y[idx], 
    #         win_STA=win_STA,
    #         win_LTA=win_LTA, 
    #         time_vector=time, 
    #         trigger_threshold_on=trigger_threshold_on,
    #         trigger_threshold_off=trigger_threshold_off, 
    #         dead_time=dead_time, 
    #         freeze_lta=False
    #     )

    #     logging.debug(f"sta_lta (station {idx}): shape={sta_lta.shape}, dtype={sta_lta.dtype}")
    #     logging.debug(f"sta_lta time (station {idx}): len={len(sta_lta_time)}, type={type(sta_lta_time)}")
    #     logging.debug(f"trigger_mask (station {idx}): len={len(trigger_mask)}, true_count={np.sum(trigger_mask)}")

    #     # create sta_lta_all_stations & trigger_mask_all_stations to save the daily STA/LTA data for every station seperatly
    #     if idx == 0:
    #         sta_lta_all_stations = sta_lta
    #         trigger_mask_all_stations = trigger_mask
    #     else:
    #         sta_lta_all_stations = np.vstack((sta_lta_all_stations, sta_lta))
    #         trigger_mask_all_stations = np.vstack((trigger_mask_all_stations, trigger_mask))
            
            
    # extract year, month and day from t_start path
    year = t_start[:4]
    month = t_start[5:7]
    day = t_start[:10]  # YYYY-MM-DD        


    # find triggers relating to the same event
    consecutive_detections_all_stations = []
    for idx in range(len(y)):
        consecutive_detections = find_consecutive_trues(
            trigger_mask_all_stations[idx], (sta_lta_time[1]-sta_lta_time[0]).total_seconds(),
            t_add_start, t_add_end, add_timewindow=True, merge=False
        )
        consecutive_detections_all_stations.append(consecutive_detections)

    # filter: keeping only the event which show up on at least min_stations 
    result = find_common_time_windows(consecutive_detections_all_stations, min_station_count=min_stations, overlap_threshold=0.6)
    events_by_station = result    

    logging.debug(f"found {len(events_by_station)} combined events.")
    
    # save STA/LTA values for the day to avoid re-calculation
    # save consecutive time windows too
    path_sta_lta_output = save_path_base / year / f"month{month}" / day 
    path_sta_lta_output.mkdir(parents=True, exist_ok=True)


    logging.debug(f"triggers all before saving STA/LTA is: {type(trigger_mask_all_stations)}")

    # save STA_LTA_time and events_by_station seperatly because their shape can differ 
    # with events_all_stations being a list of lists of different length (??????)
    logging.debug(f"sta_lta_all_stations is {len(sta_lta_all_stations)} of type {type(sta_lta_all_stations)} and shape {sta_lta_all_stations.shape}")
    logging.debug(f"sta_lta_time is {len(sta_lta_time)} of type {type(sta_lta_time)}")
    logging.debug(f"events_by_station is {len(events_by_station)} of type {type(events_by_station)}; first example: {events_by_station[0]}")
    logging.debug(f"trigger_mask_all_stations is {len(trigger_mask_all_stations)} of type {type(trigger_mask_all_stations)} and shape {trigger_mask_all_stations.shape}")
    logging.debug(f"offset_to_raw is  of type {type(offset_to_raw)}")

    np.savez_compressed(
        path_sta_lta_output / "sta_lta_data.npz",
        sta_lta_all_stations=sta_lta_all_stations,
        sta_lta_time=np.array(sta_lta_time),
        trigger_mask_all_stations=trigger_mask_all_stations,
        offset_to_raw=offset_to_raw
    )
    # save events_by_stations seperatly because list contains sublists of different lengths
    with open(path_sta_lta_output / "events_by_station.pkl", "wb") as f:
        pickle.dump(events_by_station, f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(path_sta_lta_output / 'STA_LTA_results.pkl', 'wb') as f:
    #     pickle.dump(
    #     {
    #         "sta_lta":sta_lta_all_stations,
    #         "time": sta_lta_time,
    #         "event_windows": events_by_station,
    #         "triggers_all":trigger_mask_all_stations,
    #         "offset_to_raw": offset_to_raw
    #     },
    #     f,
    #     protocol=pickle.HIGHEST_PROTOCOL
    # )
        
    
    logging.debug(f'STA/LTA calculated for every station for {day}') 
    # make directories for every event detection 
    # making sure as soon as data is processed once, all event folders are created 
    # base future calculations on empty folders in case processing was interrupted
    # (helps to later keep processing events which were only detected, not processed yet)
    existing_events = sorted([p for p in (save_path_base / year / f"month{month}" / day).glob("event*")])
    next_idx = len(existing_events)
    
    for idx, trigger_window in enumerate(events_by_station):
        event_idx = next_idx + idx  # continue counting from existing folder
        save_event_path = save_path_base / year / f"month{month}" / day / f"event{str(event_idx).zfill(3)}"
        if not save_event_path.exists():
            save_event_path.mkdir(parents=True)
        # save trigger of respective event
        np.save(save_event_path / "trigger_window.npy", trigger_window)
        start_time = sta_lta_time[trigger_window[0]]
        end_time = sta_lta_time[trigger_window[-1]]
        np.save(save_event_path / "event_times.npy", np.array([start_time, end_time], dtype=object))

    logging.debug(f'STA/LTA saved')
   
def process_single_event(event_folder: Path, raw_data_path: Path, trigger_threshold, win_STA, win_LTA):
    try:
        # Prüfe, ob Event schon verarbeitet wurde
        if (event_folder / "data_red.npy").exists():
            logging.info(f"{event_folder} already processed, skipping.")
            return

        # --- Zugehörigen Raw-Ordner finden ---
        raw_path = find_raw_folder_for_event(event_folder, raw_data_path)
        y = np.load(raw_path / 'data.npy')
        t_unix = np.load(raw_path / 'time.npy')
        t = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in t_unix]

        # Trigger laden
        trigger_window = np.load(event_folder / "trigger_window.npy", allow_pickle=True)

        # Event-Processing
        events_processing(
            #STA_LTA_vec_all=None,
            STA_LTA_time=t,
            events_by_station=[trigger_window],
            event_folder=event_folder,
            time=t,
            y=y,
            CC_threshold_zeroing=0.4,
            #win_STA=win_STA,
            #win_LTA=win_LTA
        )

    except Exception as e:
        logging.info(f"Error processing {event_folder}: {e}")

def events_processing(
    #STA_LTA_vec_all,
    STA_LTA_time,
    events_by_station,
    event_folder: Path,  # to one single event
    time,
    y,
    CC_threshold_zeroing=0.4,
    plotting=False
):
    """
    Process a single event.
    """

    # Wir nehmen nur den Event-Index
    idx = 0  # da der Ordner nur ein Event enthält
    trigger_window = events_by_station[idx]

    save_event_path = event_folder
    save_event_path.mkdir(parents=True, exist_ok=True)

    # Prüfe, ob der Ordner schon Daten enthält
    data_files_needed = ["data_red.npy"]
    if all((save_event_path / f).exists() for f in data_files_needed):
        logging.info(f"Event {save_event_path.name} already processed for {str(time[0])}... skipping.")
        return

    # Event-Daten extrahieren
    start_time = STA_LTA_time[trigger_window[0]]
    end_time = STA_LTA_time[trigger_window[-1]]

    start_data_idx = trigger_window[0]
    end_data_idx   = trigger_window[-1] + 1

    # Time slice & data slice
    time_save = time[start_data_idx:end_data_idx]
    data_raw_dyn = y[:, start_data_idx:end_data_idx]
    
    if data_raw_dyn.size == 0 or np.isnan(data_raw_dyn).all():
        logging.debug(f"Skipping event {save_event_path.name}: empty or all-NaN dynamic window")
        return

    data_red = identify_and_zero_outliers(
        data_raw_dyn, threshold=CC_threshold_zeroing, taper_frac=0, clipping=True, clipping_factor=5
    )
    time_save = time[start_data_idx:end_data_idx]

    # Speichern
    np.save(save_event_path / "times.npy", np.array(time_save, dtype=object))
    np.save(save_event_path / 'data_red.npy', data_red)

    if plotting:
        plt.figure(figsize=(12, 6))
        for i in range(data_red.shape[0]):
            plt.plot(time_save, data_red[i], label=f"Station {i}")
        plt.title(f"Event {save_event_path.name} - Day {str(STA_LTA_time[0])[:10]}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()
                   


