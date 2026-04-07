# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:37:22 2026

@author: anton

"""
import glob
import numpy as np
import os

from clustering_labeled_func import bandpass_filter


load_path_list=glob.glob('../../../DATA/MVO/ASCII/*')


event_names=['hybrid','longperiod','lr','mi','rockfall','tremor']
component='Z'


    
events={'hybrid':{},'longperiod':{},'lr':{},'mi':{},'rockfall':{}, 'tremor':{}}
station_name='MBGA'


time_list=list()
data_list=list()

for idx in range(len(event_names)):
    data_dir = load_path_list[idx]
    ldir = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    str_event = event_names[idx]
    
    component = "Z"
    
    for dir_event in ldir:
        
        f = os.path.join(data_dir, dir_event, f'{station_name}_{component}.ASC')
        meta = open(f).readlines()[0:5]

        dt = float(meta[0][6:-1])
        data = bandpass_filter(np.loadtxt(f, skiprows=5), dt, 1, 30)
        
        time_list.append(meta[-1][7:-2])
        
        
