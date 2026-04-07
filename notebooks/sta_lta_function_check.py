# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:48:57 2025

@author: anton
"""
import sys
from pathlib import Path
import pickle
import numpy as np
#%% SETTING

# Füge src zum Python-Pfad hinzu
project_root = Path().resolve().parent  # wenn das Notebook in notebooks/ liegt
sys.path.append(str(project_root / "src"))

# Jetzt kannst du Module importieren
from sta_lta_trigger.utils_sta_lta import find_folders_to_process, find_events_to_process

project_root = Path().resolve().parent  # wenn das Notebook in notebooks/ liegt
sys.path.append(str(project_root ))
from config import (RAW_DATA_DIR, PROCESSED_DATA_DIR) 

raw_data_path=str(RAW_DATA_DIR) + "/"
event_base_path=str(PROCESSED_DATA_DIR) + "/"

#%% TESTING OF COMPARISON FUNCTION
folders_to_process=find_folders_to_process(raw_data_path,event_base_path,'2010-01-02','2010-01-30')

print('folders to process are:',folders_to_process[:5])
print('folders to process are:',folders_to_process[-5:])

#%% TESTING EVENT FOLDERS TO PROCESS DETECTION

folders_to_process=find_events_to_process(event_base_path,'2010-01-02','2010-01-30')

#%%
# event=Path('../../data/detected_events/2010/month01/2010-01-01')
# with open(str(event) + '/STA_LTA_results.pkl', 'rb') as f:
#     results = pickle.load(f)

# STA_LTA_vec_all = np.load(str(event) + '/STA_LTA_vec_all.npy')
# STA_LTA_vec_time = results["STA_LTA_time"]
# events_all_stations = results["events_all_stations"]

