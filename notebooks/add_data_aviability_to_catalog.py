# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:48:53 2025

@author: anton
"""
import pandas as pd

# catalog_path='../../data/detected_events/EventCatalogs/catalog_data.csv'

# df = pd.read_csv(catalog_path)

# # add data available column (ONLY ONCE, NEVER RUN THIS AGAIN)
# #df['data_available']=True

# # alter column order
# # column_order=['ID','data_available','beam','cwt','used','notes']

# # df = df[column_order]

# # delete data available statement again since its only for days, not single events
# # df = df.drop(columns=['data_available'])

# df.to_csv(catalog_path,index=False)

import glob
from pathlib import Path

raw_data_list=glob.glob('../../data/raw_data/*/month*/*')
raw_data_list=[Path(f) for f in raw_data_list]

raw_data_list=["./"+str(f)[6:] for f in raw_data_list if any(f.iterdir())]

print(raw_data_list[:5])

catalog_path='../../data/detected_events/EventCatalogs/download_catalog.csv'

df = pd.DataFrame({
    "ID":raw_data_list,
    "data_available": [True] * len(raw_data_list)
    })

# add data available column (ONLY ONCE, NEVER RUN THIS AGAIN)
#df['data_available']=True

# alter column order
# column_order=['ID','data_available','beam','cwt','used','notes']

# df = df[column_order]

# delete data available statement again since its only for days, not single events
# df = df.drop(columns=['data_available'])

df.to_csv(catalog_path,index=False)