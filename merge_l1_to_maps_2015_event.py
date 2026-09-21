"""
L1 (ACE) to Ionosphere Map Matching — 2015 March 10-20 Event
=============================================================

Variant of merge_l1_to_maps_even_minutes.py for the 2015 ACE dataset:
  - B-field in GSM coordinates (consistent with DSCOVR pipeline)
  - Real ACE orbit distance from X_(@_x_)_km column (no fixed distance needed)
  - Speed is already positive (SW_H_SPEED, not vx_gsm component)
  - Data already interpolated upstream (v2 file)

Author: Francesco Ramunno
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CONFIGURATION
# ============================================================================

MAP_DIR     = '/capstor/scratch/cscs/framunno/ionosphere_data/event_maps/2015_march_10_20/'
L1_FILE     = '/users/framunno/data/ionosphere/combined_ACE_1min_gsm_2015.csv'
OUTPUT_FILE = '/users/framunno/data/ionosphere/l1_to_map_matched_2015_march_10_20.csv'

MAX_MATCH_TOLERANCE_S = 60
N_WORKERS             = 8
BATCH_SIZE            = 10_000

# ============================================================================
# HELPERS
# ============================================================================

def extract_datetime_from_filename(filename):
    parts = filename.replace('.npy', '').split('_')
    try:
        if parts[0] in ('map', ''):
            parts = parts[1:]
        year, month, day, hour, minute, second = parts[:6]
        return datetime(int(year), int(month), int(day),
                        int(hour), int(minute), int(second))
    except Exception:
        return None


def is_valid_row(row):
    for col in ['speed_kms', 'bx_gsm', 'by_gsm', 'bz_gsm']:
        if pd.isna(row[col]):
            return False
    if pd.isna(row['x_ace_km']) or row['x_ace_km'] <= 0:
        return False
    if row['speed_kms'] <= 0:
        return False
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("L1-TO-MAP MATCHING  —  2015 MARCH EVENT  (real ACE orbit distance)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Load L1 data
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading ACE L1 data (GSM, interpolated)...")
    df = pd.read_csv(L1_FILE)
    df = df.rename(columns={
        'Time':                             'time',
        'BX_(GSM)_(@_x_component_)_nT':    'bx_gsm',
        'BY_(GSM)_(@_y_component_)_nT':    'by_gsm',
        'BZ_(GSM)_(@_z_component_)_nT':    'bz_gsm',
        'SW_H_SPEED_km/s':                 'speed_kms',
        'X_(@_x_)_km':                     'x_ace_km',
    })
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    print(f"  Records loaded : {len(df):,}")
    print(f"  Time range     : {df['time'].min()} → {df['time'].max()}")
    print(f"  NaN counts     : { {c: int(df[c].isna().sum()) for c in ['speed_kms','bx_gsm','by_gsm','bz_gsm','x_ace_km']} }")

    # -------------------------------------------------------------------------
    # STEP 2: Scan and filter map files (even minutes only)
    # -------------------------------------------------------------------------
    print("\n[2/4] Scanning ionosphere map files...")
    all_files = [f for f in os.listdir(MAP_DIR) if f.endswith('.npy')]
    print(f"  Found {len(all_files):,} .npy files")

    map_data = []
    for fname in tqdm(all_files, desc="  Parsing filenames"):
        dt = extract_datetime_from_filename(fname)
        if dt is not None and dt.minute % 2 == 0:
            map_data.append({'filename': fname, 'map_time': pd.Timestamp(dt)})

    map_df = pd.DataFrame(map_data).sort_values('map_time').reset_index(drop=True)
    map_times = map_df['map_time'].values
    print(f"  Even-minute maps : {len(map_df):,}")
    print(f"  Map time range   : {map_df['map_time'].min()} → {map_df['map_time'].max()}")

    # -------------------------------------------------------------------------
    # STEP 3: Match each L1 point to the nearest map
    # -------------------------------------------------------------------------
    print("\n[3/4] Matching L1 data to ionosphere maps...")

    def process_row(idx):
        row = df.iloc[idx]
        if not is_valid_row(row):
            return None

        travel_s   = row['x_ace_km'] / row['speed_kms']
        earth_time = row['time'] + timedelta(seconds=travel_s)
        earth_np   = pd.Timestamp(earth_time).to_numpy()

        i = map_times.searchsorted(earth_np)
        candidates = [c for c in (i - 1, i) if 0 <= c < len(map_times)]
        if not candidates:
            return None

        best_i, best_diff = min(
            ((c, abs((pd.Timestamp(map_times[c]) - earth_time).total_seconds())) for c in candidates),
            key=lambda x: x[1]
        )

        if best_diff > MAX_MATCH_TOLERANCE_S:
            return None

        map_row  = map_df.iloc[best_i]
        map_time = map_row['map_time']

        return {
            'time':       row['time'].tz_localize('UTC'),
            'speed_kms':  row['speed_kms'],
            'bx_gsm':     row['bx_gsm'],
            'by_gsm':     row['by_gsm'],
            'bz_gsm':     row['bz_gsm'],
            'x_gse_km':   row['x_ace_km'],
            'delay_min':  travel_s / 60,
            'earth_time': pd.Timestamp(earth_time).tz_localize('UTC'),
            'time_map':   map_time,
            'dt_diff':    map_time - earth_time,
            'filename':   map_row['filename'],
        }

    matched = []
    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(process_row, i): i for i in range(start, end)}
            for fut in tqdm(as_completed(futures), total=end - start,
                            desc=f"  Batch {start//BATCH_SIZE + 1}"):
                r = fut.result()
                if r is not None:
                    matched.append(r)
        print(f"  Matches so far: {len(matched):,}")

    # -------------------------------------------------------------------------
    # STEP 4: Save
    # -------------------------------------------------------------------------
    print("\n[4/4] Saving results...")
    matched_df = pd.DataFrame(matched)
    matched_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"  L1 records total : {len(df):,}")
    print(f"  Matched          : {len(matched_df):,}  ({len(matched_df)/len(df)*100:.1f}%)")
    print(f"  Output           : {OUTPUT_FILE}")

    if len(matched_df) > 0:
        print(f"\n  Delay  mean: {matched_df['delay_min'].mean():.1f} min  "
              f"std: {matched_df['delay_min'].std():.1f} min  "
              f"range: [{matched_df['delay_min'].min():.1f}, {matched_df['delay_min'].max():.1f}]")
        dt_s = matched_df['dt_diff'].dt.total_seconds().abs()
        print(f"  Match error mean: {dt_s.mean():.1f} s   max: {dt_s.max():.1f} s")

    print("=" * 80)


if __name__ == "__main__":
    main()
