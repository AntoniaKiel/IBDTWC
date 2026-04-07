# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:17:38 2025

@author: anton
"""

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path

def generate_time_windows(t_start:str, t_end:str):
    """Generate daily time window between both input timestamps

    Parameters
    ----------
    t_start : str
        Start time in ISO format 'YYYY-MM-DDTHH:MM:SS', e.g., '2022-01-12T00:00:00'.
    t_end : str
        End time in ISO format 'YYYY-MM-DDTHH:MM:SS'.

    Returns
    -------
    list of tuple
        A list of pairs (tuples), each containing the start and end time of one day.
        Start times are at 00:00:00 UTC, end times are at 23:59:59 UTC.
        Returned times are datetime objects with UTC timezone.
    """
    # turn string into datetime
    start_time = datetime.strptime(t_start, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
    end_time = datetime.strptime(t_end, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)

    # pre-allocation of empty lists 
    start_times, end_times = [], []
    current_time = start_time
    while current_time < end_time:
        start_times.append(current_time.isoformat())
        window_end = current_time + timedelta(days=1) - timedelta(seconds=1)
        end_times.append(window_end.isoformat())
        current_time += timedelta(days=1)   # go to next day

    logging.debug("list of daily tuples within desired timeframe created")
    return list(zip(start_times, end_times))

def is_window_unprocessed(t_start:str, t_end:str, 
                          base_path:Path, proc_path:Path) -> bool:
    """    Check whether a given time window has not yet been processed.

    This function checks whether the raw data folder for the specified time window
    exists and contains data, and whether detected events already exist for that day.
    If neither raw data nor event detections are present, the window is considered unprocessed.

    Parameters
    ----------
    t_start : str
        Start of the time window in ISO 8601 format, e.g. '2022-01-12T00:00:00+00:00'.
    t_end : str
        End of the time window in ISO 8601 format, e.g. '2022-01-12T23:59:59+00:00'.
    base_path : Path
        Base directory where raw data folders are stored.
    proc_path : Path
        Directory where processed event data is stored.

    Returns
    -------
    bool
        True if the time window has not yet been processed (no raw data and no detected events),
        False otherwise.

    Notes
    -----
    - The function assumes that raw data folders are organized as:
      `{base_path}/{year}/month{MM}/{t_start}_{t_end}/`
    - Detected events are assumed to be in:
      `{proc_path}/{year}/month{MM}/{YYYY-MM-DD}/`
    """
    logging.debug("startin is_window_unprocessed...")
    logging.debug(f"t_start is {t_start} of type {type(t_start)}")
    logging.debug(f"t_end is {t_end} of type {type(t_end)}")
    logging.debug(f"base_path is {base_path} of type {type(base_path)}")
    logging.debug(f"proc_path is {proc_path} of type {type(proc_path)}")

    save_path = (
        f'{base_path}/{t_start[:4]}/month{t_start[5:7]}/'
        f'{t_start.replace(":", "-")}_{t_end.replace(":", "-")}/'
    )
    
    logging.debug(f"t_start is {t_start} of type {type(t_start)}")
    t_start= datetime.fromisoformat(t_start).astimezone(timezone.utc)
    
    # check if there are event folders for the considered day -> do not re-download raw data in that case
    event_paths=str(proc_path) + '/' +  str(t_start.year) + '/month'+ t_start.strftime("%m") + '/' + t_start.strftime("%Y-%m-%d")
    
    if not os.path.exists(save_path) or not os.listdir(save_path):
        #logging.info(f'save path folder exists but is empty for {t_start}')
        if not os.path.exists(event_paths):     # if not, check if events were already detected for that day
            #logging.info(f'no detected events exist for {t_start}')
            return True







