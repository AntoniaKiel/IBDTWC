import os
import numpy as np
from typing import Union
import logging
from pathlib import Path
from config import  PROCESSED_DATA_DIR
from src.data_download.time_windows import generate_time_windows, is_window_unprocessed
from src.data_download.retry_utils import retry_on_connection_error
from src.data_download.fetch_GFZ_data import get_available_stations, process_seismic_data

def download_data(t_start:str,
                  t_end:str,
                  network_code:str,
                  stations_to_remove:list,
                  fmin:Union[int, float],
                  fmax:Union[int, float],
                  clipping:Union[int, float],
                  response_output,
                  rms_window:Union[int,float],
                  raw_data_dir:Path,
                  processed_data_dir:Path,
                  retry:dict):
    """This function downloads seismic data from the GFZ server and saves it itnto a raw_data path

    Parameters
    ----------
    t_start : str
        Start time string in the format "YYYY-MM-DDTHH:MM:SS", e.g., "2022-01-12T00:00:00".
        Must include leading zeros and a literal 'T' between date and time.
    t_end : str
        end time string in the format "YYYY-MM-DDTHH:MM:SS", e.g., "2022-01-12T00:00:00".
        Must include leading zeros and a literal 'T' between date and time.
    network_code : str
        string of the valid network code
    stations_to_remove : list
        list of valid station strings, which are skipped for further investigation
    fmin : Union[int, float]
        lower frequency band filter boundary [Hz]
    fmax : Union[int, float]
        upper frequency band filter boundary [Hz]
    clipping: Union [int, float]
        factor of the std(windowed rms(raw signal)) to clip the spikes to 
    response_output: 

    rms_window: Union [int,float]
        time for windowed rms-calculation used to clip spike events [s]
    raw_data_dir: Path
        pathlib Path to the data/raw_data folder
    processed_data_dir: Path
        pathlib Path to saving folder data/detected_events
    retry: dict
        dictionary stating the exact parameters for connection retry to server
    """


    # get daily time windows to download daily raw data from server
    time_windows = generate_time_windows(t_start, t_end)
    unprocessed = [
        (start, end)
        for start, end in time_windows
        if is_window_unprocessed(start, end, raw_data_dir, processed_data_dir)
    ]
    logging.info(f"{len(unprocessed)} unprocessed days between {t_start} and {t_end}")

    for (start, end) in unprocessed:
        logging.info(f"Downloading of {start} started...")
        
        save_path = (
            f'{raw_data_dir}/{start[:4]}/month{start[5:7]}/'
            f'{start.replace(":", "-")}_{end.replace(":", "-")}/'
        )
        os.makedirs(save_path, exist_ok=True)

        # Stationen laden
        stations = retry_on_connection_error(
            get_available_stations,
            max_retries=retry["max_retries"],
            retry_delay=retry["retry_delay"],
            network_code=network_code,
            t_start=start,
            t_end=end
        )

        stations = [s for s in stations if s not in stations_to_remove]
        # Daten verarbeiten
        try:
          y, times = retry_on_connection_error(
              process_seismic_data,
              max_retries=retry["max_retries"],
              retry_delay=retry["retry_delay"],
              station_list=stations,
              t_start=start,
              t_end=end,
              fmin=fmin,
              fmax=fmax,
              clipping=clipping,
              response_output=response_output,
              rms_window=rms_window
          )
        except ConnectionError:
              logging.error(f"Stations for {start} – {end} could not be downloaded after all retries. Day skipped")
              continue

        if len(y) != 0:
            np.save(save_path + 'data.npy', y)
            np.save(save_path + 'time.npy', times)
            np.save(save_path + 'station_list.npy', stations)
            print(f"Raw data saved for {start} until {end}", flush=True)

            
            
            
            
            
            


