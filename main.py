# -*- coding: utf-8 -*-

# Thread-Limits GANZ OBEN
# os.environ["OPENBLAS_NUM_THREADS"] = "128"
# os.environ["MKL_NUM_THREADS"] = "128"
# os.environ["NUMEXPR_NUM_THREADS"] = "128"
# os.environ["OMP_NUM_THREADS"] = "128"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "128"

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="obspy.core.util.base")

import logging

# =============================================================================
# Imports 
# =============================================================================
from config import (DOWNLOAD_ACTIVE,STA_LTA_ACTIVE,BEAMFORMING_ACTIVE,CATALOG_CREATION_ACTIVE,DTW_CLUSTERING_ACTIVE,
                    PLOTTING_DTW,DOWNLOAD,RETRY,CLIPPING_RAW,RESPONSE_OUTPUT,RMS_WINDOW,LOG_LEVEL,
                    RAW_DATA_DIR,PROCESSED_DATA_DIR,START,END,T_ADD_START,T_ADD_END,FMIN,FMAX,WIN_STA,WIN_LTA,DEAD_TIME,TRIGGER_THRESHOLD, THRESHOLD_OFF,
                    MIN_STATIONS,N_KERNELS,SIGMA_FACTOR,BUFFER_SIZE,MAX_BUFFER_AGE,
                    SLOWNESS_MIN,SLOWNESS_MAX,SLOWNESS_SPACING,BACKAZIMUTH_SPACING,STATION_METADATA_PATH,RE_PROCESS,PLOT_BEAMFORMING,
                    CATALOG_PATH,SAKOE_CHIBA_RADIUS_TIME,N_CLUSTERS,N_MIN_CLUSTER,N_MAX_CLUSTER,INITIAL_SAMPLE_SIZE,
                    SAMPLING_FREQ,SAVE_DATA,RECALCULATING,TEST_N_CLUSTERS,SUBSET,N_SEED,BEAMPOWER_THRESHOLD,
                    LOAD_PATH_CLUSTERING,WAVELETNAME,N_EXAMPLES_PER_CLUSTER,N_BINS,REEXTRACTING_INIT_BEAMS,SLOWNESS_THRESHOLD)



log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug.log",mode='w')
    ]
)

def main():
    # DOWNLOAD
    if DOWNLOAD_ACTIVE:
        from src.data_download.download import download_data
        logging.info('--------------------Starting the download process-------------------')
        download_data(t_start=DOWNLOAD['t_start'],
                      t_end=DOWNLOAD['t_end'],
                      network_code=DOWNLOAD['network_code'],
                      stations_to_remove=DOWNLOAD['stations_to_remove'],
                      fmin=DOWNLOAD['fmin'],fmax=DOWNLOAD['fmax'],
                      clipping=CLIPPING_RAW,
                      response_output=RESPONSE_OUTPUT,
                      rms_window=RMS_WINDOW,
                      raw_data_dir=RAW_DATA_DIR,
                      processed_data_dir=PROCESSED_DATA_DIR,
                      retry=RETRY)

    # STA/LTA
    if STA_LTA_ACTIVE:
        logging.info('----------------Starting STA/LTA trigger-----------------------')
        from src.sta_lta_trigger.event_processing import sta_lta_pipeline
        sta_lta_pipeline(
            raw_data_path=str(RAW_DATA_DIR) + "/",
            event_base_path=str(PROCESSED_DATA_DIR) + "/",
            start_date=START,
            end_date=END,
            t_add_start=T_ADD_START,
            t_add_end=T_ADD_END,
            fmin=FMIN,
            fmax=FMAX,
            win_STA=WIN_STA,
            win_LTA=WIN_LTA,
            dead_time=DEAD_TIME,
            trigger_threshold=TRIGGER_THRESHOLD,
            threshold_off=THRESHOLD_OFF,
            min_stations=MIN_STATIONS,
            n_kernels=N_KERNELS
        )

    # BEAMFORMING
    if BEAMFORMING_ACTIVE:
        logging.info('--------------------Starting beamforming----------------------------')
        from src.beamforming.beam import beamforming_pipeline
        import torch

        # Stelle sicher, dass CUDA verfügbar ist und GPU 0 benutzen
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Running on device: {device}")
        
        
        beamforming_pipeline(
            START, END,
            FMIN, FMAX, device,
            SLOWNESS_MIN, SLOWNESS_MAX,
            SLOWNESS_SPACING, BACKAZIMUTH_SPACING, N_KERNELS,
            PROCESSED_DATA_DIR, STATION_METADATA_PATH, 
            RE_PROCESS, PLOT_BEAMFORMING
        )

    # CATALOG CREATION
    if CATALOG_CREATION_ACTIVE:
        logging.info('--------------------CATALOG CREATION -----------------------')
        from src.clean_catalog.catalog_cleaning import clean_catalog_func
        clean_catalog_func(PROCESSED_DATA_DIR,SLOWNESS_THRESHOLD,BEAMPOWER_THRESHOLD)

    # DTW CLUSTERING
    if DTW_CLUSTERING_ACTIVE:
        logging.info('-----------------------DTW CLUSTERING -------------------------')
        from src.clustering.dtw import dtw_function

        dtw_function(
            CATALOG_PATH,
            sakoe_chiba_radius_time=SAKOE_CHIBA_RADIUS_TIME,
            n_clusters=N_CLUSTERS,
            init_sample_size=INITIAL_SAMPLE_SIZE,
            n_min_cluster=N_MIN_CLUSTER,
            n_max_cluster=N_MAX_CLUSTER,
            sampling_freq=SAMPLING_FREQ,
            sigma_factor=SIGMA_FACTOR,buffer_size=BUFFER_SIZE,max_buffer_age=MAX_BUFFER_AGE,
            save_data=SAVE_DATA,
            recalculating=RECALCULATING,
            reextracting_init_beams=REEXTRACTING_INIT_BEAMS,
            test_n_clusters=TEST_N_CLUSTERS,
            subset=SUBSET,
            n_seed=N_SEED,
        )


    # PLOTTING
    if PLOTTING_DTW:
        logging.info('--------------PLOTTING DTW RESULTS---------------------')
        from src.plotting.plots import plotting_func

        SAVE_PLT_PATH = LOAD_PATH_CLUSTERING + "/plots"
        plotting_func(
            CATALOG_PATH, LOAD_PATH_CLUSTERING, SAVE_PLT_PATH,
            SAKOE_CHIBA_RADIUS_TIME, N_CLUSTERS, N_SEED,
            FMIN, FMAX, SAMPLING_FREQ, WAVELETNAME,
            N_EXAMPLES_PER_CLUSTER, N_BINS
        )


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
