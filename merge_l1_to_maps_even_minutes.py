"""
L1 Solar Wind to Ionosphere Map Matching
=========================================

Physics-based matching that accounts for propagation delay from L1 to Earth.

Approach:
1. Filter both L1 data and map files for EVEN minutes only (00, 02, 04, ..., 58)
2. For each L1 measurement, calculate when it arrives at Earth
3. Find the nearest ionosphere map to that arrival time
4. Output matched pairs with propagation info

Author: Francesco Ramunno
Date: 2025
"""

import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
from IPython import embed
# ============================================================================
# CONFIGURATION
# ============================================================================

MAP_DIR = './data_root/ionosphere_data/all_maps/'
SOLAR_WIND_FILE = './data/ionosphere/combined_f1m_m1m_2020_2025_interpolated.csv'
DSCOVR_FILE = './data/ionosphere/DSCOVR_ORBIT_PRE_2020_2025.csv'
OUTPUT_FILE = './data/ionosphere/l1_to_map_matched_2020_2025.csv'

# Matching parameters
MAX_MATCH_TOLERANCE_SECONDS = 60  # Accept matches within 5 minutes
N_WORKERS = 8  # Parallel processing workers
BATCH_SIZE = 10000  # Process L1 data in batches

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_datetime_from_filename(filename):
    """
    Extract datetime from ionosphere map filename.

    Format: map_2024_9_1_13_48_0.npy or _2024_9_1_13_48_0.npy
    Returns: datetime object or None if parsing fails
    """
    parts = filename.replace('.npy', '').split('_')
    try:
        # Skip 'map' or empty prefix
        if parts[0] == 'map' or parts[0] == '':
            parts = parts[1:]

        year, month, day, hour, minute, second = parts[0:6]
        return datetime(int(year), int(month), int(day),
                       int(hour), int(minute), int(second))
    except:
        return None


def calculate_earth_arrival_time(l1_time, velocity_km_s, x_gse_km):
    """
    Calculate when L1 solar wind measurement arrives at Earth.

    Args:
        l1_time: Time of L1 measurement
        velocity_km_s: Solar wind velocity (negative in GSM coordinates)
        x_gse_km: Actual DSCOVR satellite distance along Sun-Earth line (X_GSE)

    Returns:
        datetime of Earth arrival, or None if velocity or distance is invalid
    """
    velocity = abs(velocity_km_s)

    # Skip invalid velocities
    if velocity <= 0 or velocity == 99999 or velocity == -99999 or np.isnan(velocity):
        return None

    if np.isnan(x_gse_km) or x_gse_km <= 0:
        return None

    # Calculate propagation time: actual distance / velocity
    travel_time_seconds = x_gse_km / velocity

    return l1_time + timedelta(seconds=travel_time_seconds)


def is_valid_condition_row(row):
    """
    Check if all 4 solar wind conditions are valid.

    Returns: True if all valid, False otherwise
    """
    for cond in ['proton_vx_gsm', 'bx_gsm', 'by_gsm', 'bz_gsm']:
        val = row[cond]
        if pd.isna(val) or val == 99999 or val == -99999:
            return False
    return True


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("=" * 80)
    print("L1-TO-MAP MATCHING (EVEN MINUTES ONLY)")
    print("=" * 80)

    # ========================================================================
    # STEP 1: LOAD L1 DATA AND JOIN WITH DSCOVR ORBIT
    # ========================================================================
    print("\n[1/4] Loading L1 solar wind data...")
    solar_wind_df = pd.read_csv(SOLAR_WIND_FILE)
    solar_wind_df['time'] = pd.to_datetime(solar_wind_df['time'])
    solar_wind_df['time'] = solar_wind_df['time'].dt.tz_localize(None)
    solar_wind_df = solar_wind_df.sort_values('time').reset_index(drop=True)
    print(f"Total L1 records: {len(solar_wind_df):,}")
    print(f"Time range: {solar_wind_df['time'].min()} to {solar_wind_df['time'].max()}")

    print("\nLoading DSCOVR orbit data...")
    dscovr_df = pd.read_csv(DSCOVR_FILE)
    dscovr_df = dscovr_df.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ': 'time'})
    dscovr_df['time'] = pd.to_datetime(dscovr_df['time']).dt.tz_localize(None)
    dscovr_df = dscovr_df[['time', 'X_GSE_km']].sort_values('time').reset_index(drop=True)
    print(f"Total DSCOVR records: {len(dscovr_df):,}")

    # Merge on nearest minute
    solar_wind_df = pd.merge_asof(
        solar_wind_df, dscovr_df,
        on='time', direction='nearest', tolerance=pd.Timedelta('1min')
    )
    missing_x = solar_wind_df['X_GSE_km'].isna().sum()
    print(f"L1 rows with no DSCOVR match: {missing_x:,}")
    print(f"L1 records after join: {len(solar_wind_df):,}")

    # ========================================================================
    # STEP 2: SCAN AND FILTER MAP FILES (EVEN MINUTES ONLY)
    # ========================================================================
    print("\n[2/4] Scanning ionosphere map files...")
    all_files = [f for f in os.listdir(MAP_DIR) if f.endswith('.npy')]
    print(f"Found {len(all_files):,} .npy files")

    # Parse filenames and filter for even minutes
    print("Filtering map files for even minutes (00, 02, 04, ..., 58)...")
    map_data = []
    for filename in tqdm(all_files, desc="Parsing filenames"):
        dt = extract_datetime_from_filename(filename)
        if dt is not None and dt.minute % 2 == 0:  # Only even minutes
            map_data.append({
                'filename': filename,
                'map_time': pd.Timestamp(dt)
            })

    map_df = pd.DataFrame(map_data)
    map_df = map_df.sort_values('map_time').reset_index(drop=True)

    print(f"Map files with even minutes: {len(map_df):,}")
    print(f"Time range: {map_df['map_time'].min()} to {map_df['map_time'].max()}")

    # Create numpy array of map times for fast searching
    map_times = map_df['map_time'].values

    # ========================================================================
    # STEP 3: MATCH EACH L1 POINT TO NEAREST MAP
    # ========================================================================
    print("\n[3/4] Matching L1 data to ionosphere maps...")
    print(f"Using {N_WORKERS} parallel workers, batch size {BATCH_SIZE:,}")

    def process_l1_row(idx):
        """
        For each L1 measurement:
        1. Calculate Earth arrival time
        2. Find nearest map file
        3. Return matched pair if within tolerance
        """
        # embed()
        row = solar_wind_df.iloc[idx]
        l1_time = row['time']
        velocity = row['proton_vx_gsm']
        x_gse_km = row['X_GSE_km']

        # Check if all conditions are valid
        if not is_valid_condition_row(row):
            return None

        # Calculate Earth arrival time using actual DSCOVR distance
        earth_arrival_time = calculate_earth_arrival_time(l1_time, velocity, x_gse_km)

        if earth_arrival_time is None:
            return None

        # Find nearest map to earth arrival time
        earth_arrival_np = pd.Timestamp(earth_arrival_time).to_numpy()

        # Use binary search to find closest map
        idx_after = map_times.searchsorted(earth_arrival_np)

        # Check both the map before and after the arrival time
        candidates = []
        if idx_after > 0:
            candidates.append(idx_after - 1)
        if idx_after < len(map_times):
            candidates.append(idx_after)

        if not candidates:
            return None

        # Find the closest map
        best_idx = None
        best_diff = None

        for map_idx in candidates:
            map_time = pd.Timestamp(map_times[map_idx])
            time_diff_seconds = abs((map_time - earth_arrival_time).total_seconds())

            if best_idx is None or time_diff_seconds < best_diff:
                best_idx = map_idx
                best_diff = time_diff_seconds

        # Accept match if within tolerance
        if best_diff is not None and best_diff <= MAX_MATCH_TOLERANCE_SECONDS:
            map_row = map_df.iloc[best_idx]
            map_time = map_row['map_time']

            # Calculate speed and delay
            speed_kms = abs(velocity)
            delay_min = (earth_arrival_time - l1_time).total_seconds() / 60
            dt_diff = map_time - earth_arrival_time

            # Match old CSV format with timezone-aware times
            return {
                'time': l1_time.tz_localize('UTC'),
                'proton_vx_gsm': row['proton_vx_gsm'],
                'bx_gsm': row['bx_gsm'],
                'by_gsm': row['by_gsm'],
                'bz_gsm': row['bz_gsm'],
                'x_gse_km': x_gse_km,
                'speed_kms': speed_kms,
                'delay_min': delay_min,
                'earth_time': pd.Timestamp(earth_arrival_time).tz_localize('UTC'),
                'time_map': map_time,
                'dt_diff': dt_diff,
                'filename': map_row['filename']
            }

        return None

    # Process in batches with parallel execution
    matched_data = []
    total_rows = len(solar_wind_df)

    for batch_start in range(0, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_indices = range(batch_start, batch_end)

        print(f"Processing batch {batch_start:,}-{batch_end:,}...")

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(process_l1_row, idx): idx for idx in batch_indices}

            batch_results = []
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Batch {batch_start//BATCH_SIZE + 1}"):
                result = future.result()
                if result is not None:
                    batch_results.append(result)

            matched_data.extend(batch_results)

        print(f"  Matches so far: {len(matched_data):,}")

    # ========================================================================
    # STEP 4: SAVE RESULTS
    # ========================================================================
    print("\n[4/4] Saving results...")
    matched_df = pd.DataFrame(matched_data)
    matched_df.to_csv(OUTPUT_FILE, index=False)

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("MATCHING COMPLETE")
    print("=" * 80)
    print(f"Total L1 records (even min):  {len(solar_wind_df):,}")
    print(f"Total map files (even min):   {len(map_df):,}")
    print(f"Successfully matched:         {len(matched_df):,}")
    print(f"Match rate:                   {len(matched_df)/len(solar_wind_df)*100:.2f}%")
    print(f"\nSaved to: {OUTPUT_FILE}")

    if len(matched_df) > 0:
        print("\n" + "-" * 80)
        print("PROPAGATION DELAY STATISTICS")
        print("-" * 80)
        print(f"  Mean:  {matched_df['delay_min'].mean():.2f} minutes")
        print(f"  Std:   {matched_df['delay_min'].std():.2f} minutes")
        print(f"  Min:   {matched_df['delay_min'].min():.2f} minutes")
        print(f"  Max:   {matched_df['delay_min'].max():.2f} minutes")

        print("\n" + "-" * 80)
        print("MATCHING ACCURACY")
        print("-" * 80)
        # Convert dt_diff to seconds for accuracy metrics
        matched_df['time_diff_seconds'] = matched_df['dt_diff'].dt.total_seconds().abs()
        print(f"  Mean error: {matched_df['time_diff_seconds'].mean():.2f} seconds")
        print(f"  Max error:  {matched_df['time_diff_seconds'].max():.2f} seconds")

        # Show distribution of matching errors
        within_1min = (matched_df['time_diff_seconds'] <= 60).sum()
        within_2min = (matched_df['time_diff_seconds'] <= 120).sum()
        within_5min = (matched_df['time_diff_seconds'] <= 300).sum()

        print(f"\n  Matches within 1 minute: {within_1min:,} ({within_1min/len(matched_df)*100:.2f}%)")
        print(f"  Matches within 2 minutes: {within_2min:,} ({within_2min/len(matched_df)*100:.2f}%)")
        print(f"  Matches within 5 minutes: {within_5min:,} ({within_5min/len(matched_df)*100:.2f}%)")

        print("\n" + "-" * 80)
        print("SAMPLE OUTPUT (first 5 rows)")
        print("-" * 80)
        print(matched_df.head().to_string())

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
