# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:03:54 2025

@author: anton
"""
import os
import shutil

# Pfad zu deinem Root-Ordner
root_path = "../../data/raw_data"  # Windows-Beispiel
# root_path = "/scratch/u/u301067/phd/data/raw_data"  # Linux-Beispiel

# 1 Alle Ordner sammeln, die "event" im Namen haben
event_dirs = []
for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
    for dirname in dirnames:
        if "event" in dirname.lower():  # ignore case
            full_path = os.path.join(dirpath, dirname)
            event_dirs.append(full_path)

# 2 Optional: Liste prüfen
print("Folgende Ordner werden gelöscht:")
for d in event_dirs:
    print(d)

#%%
# 3 Löschen
for d in event_dirs:
    try:
        shutil.rmtree(d)
        print(f"Gelöscht: {d}")
    except Exception as e:
        print(f"Fehler beim Löschen von {d}: {e}")