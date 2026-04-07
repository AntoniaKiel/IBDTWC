# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:33:52 2025

@author: anton
"""
import glob
import pandas as pd
import os
import numpy as np
import torch

from src.clean_catalog.catalog_utils import is_mostly_zero, check_for_noisy_event

def clean_catalog_func(processed_path, slowness_threshold=0.2, beam_power_threshold=1e6):
    """
    Prüft alle Event-Ordner in processed_path auf Datenqualität.
    Filtert Events nach:
      1. Slowness-Magnitude (slowness_best)
      2. Beam Power am Punkt der besten Slowness
      3. Optional: Dominierende 0-Traces, zu viele Null-Traces, SNR
    
    Speichert das bereinigte Catalog als CSV.
    """
    import glob, os
    import pandas as pd
    import numpy as np

    paths_events = glob.glob(str(processed_path) + '/*/month*/*/event*/')

    # Create DataFrame
    df = pd.DataFrame(paths_events, columns=["ID"])
    df['beam'] = None
    df['cwt'] = None
    df['used'] = None
    df['notes'] = None

    processed_times = []  # Store times of the last 20 processed events

    for idx, path in enumerate(paths_events):
        try:
            # --- Check if beam exists ---
            has_beam = os.path.exists(path + '/BEAM_results.npy') and os.path.exists(path + '/BEAM_results_15s.npy')
            df.loc[df['ID'] == path, "beam"] = has_beam

            # --- Check if cwt is calculated ---
            df.loc[df['ID'] == path, "cwt"] = os.path.exists(path + '/cwt_data.npy') and os.path.exists(path + '/BEAM_results_15s.npy')

            # --- Skip if no beam file ---
            if not has_beam:
                continue

            RESULTS = np.load(path + '/BEAM_results.npy', allow_pickle=True)
            traces = np.array(RESULTS[0]['shifted_traces'])
            beam = np.mean(traces, axis=0)
            zero_traces = np.sum(np.all(traces == 0, axis=1))

            df.loc[df['ID'] == path, "used"] = True  # Default True

            # === Step 1: Basic checks ===
            if is_mostly_zero(beam):
                df.loc[df['ID'] == path, "used"] = False
                df.loc[df['ID'] == path, "notes"] = "Beam dominated by 0s"
            elif zero_traces > 5:
                df.loc[df['ID'] == path, "used"] = False
                df.loc[df['ID'] == path, "notes"] = "More than 5 stations set to 0"
            else:
                # --- Slowness-Magnitude check ---
                slowness_best = RESULTS[0]['slowness_best']
                if isinstance(slowness_best, np.ndarray) == False and "torch" in str(type(slowness_best)):
                    slowness_best = slowness_best.detach().cpu().numpy()
                slowness_mag = np.linalg.norm(slowness_best)

                if slowness_mag < slowness_threshold:
                    df.loc[df['ID'] == path, "used"] = False
                    df.loc[df['ID'] == path, "notes"] = "Low slowness magnitude"
                else:
                    # --- Beam Power at best Slowness check ---
                    try:
                        ss = RESULTS[0]['slowness_space']
                        sp = RESULTS[0]['slowness power']

                        def to_np(x):
                            return x.detach().cpu().numpy() if isinstance(x, np.ndarray) == False and "torch" in str(type(x)) else np.array(x)

                        ss, sp = to_np(ss), to_np(sp).flatten()
                        # Index des Punktes, der am nächsten zu slowness_best liegt
                        best_idx = np.argmin(np.linalg.norm(ss - slowness_best, axis=1))
                        best_power = sp[best_idx]

                        if best_power < beam_power_threshold:
                            df.loc[df['ID'] == path, "used"] = False
                            note = df.loc[df['ID'] == path, "notes"].values[0]
                            if note is None:
                                note = ""
                            df.loc[df['ID'] == path, "notes"] = f"{note} Beam power < {beam_power_threshold}".strip()
                    except Exception as e:
                        print(f"Error computing best_power for {path}: {e}")

                    # --- Optional: SNR check ---
                    sim = check_for_noisy_event(beam, n_sections=15)
                    if sim['snr'] < 3:
                        df.loc[df['ID'] == path, "used"] = False
                        note = df.loc[df['ID'] == path, "notes"].values[0]
                        if note is None:
                            note = ""
                        df.loc[df['ID'] == path, "notes"] = f"{note} Low SNR".strip()

            # keep time window info for overlap check
            event_times = RESULTS[0]['times_date']
            processed_times.extend(event_times)
            processed_times = processed_times[-20:]

            del RESULTS, traces, beam, zero_traces

        except Exception as e:
            print(f"Error processing {path}: {e}")

        print(f"{idx} beam processed, ({((idx/len(paths_events))*100):.2f}%) "
              f"used: {df.loc[df['ID']==path,'used'].values[0]}")

    # --- Save cleaned catalog ---
    save_dir = processed_path / "EventCatalogs"
    os.makedirs(save_dir, exist_ok=True)
    save_file = save_dir / "catalog_data.csv"
    df.to_csv(save_file, index=False)

    beam_paths_usable = len(df.loc[(df["beam"] == True) & (df["used"] == True), "ID"].unique().tolist())
    print(f"\n✅ {beam_paths_usable} paths usable for clustering after cleaning.")
    print(f"📄 Catalog saved to: {save_file}")


