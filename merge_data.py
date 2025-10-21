import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import numpy as np

print("Script started!", flush=True)

# Check file sizes first
params_file = "/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/params.csv"
solar_wind_file = "/mnt/nas05/data01/francesco/combined_f1m_m1m_2024.csv"

print(f"Params file size: {os.path.getsize(params_file) / 1024 / 1024:.2f} MB", flush=True)
print(f"Solar wind file size: {os.path.getsize(solar_wind_file) / 1024 / 1024:.2f} MB", flush=True)

# Load the params data
print('Loading params data...', flush=True)
params_df = pd.read_csv(params_file)
print(f'Loaded params data: {len(params_df)} rows', flush=True)

print('Loading solar wind data...', flush=True)
# Load the solar wind data in chunks if it's too large
try:
    solar_wind_df = pd.read_csv(solar_wind_file)
    print(f'Loaded solar wind data: {len(solar_wind_df)} rows', flush=True)
except MemoryError:
    print("Memory error loading solar wind data, trying chunked approach...", flush=True)
    # Load in chunks
    chunk_list = []
    for chunk in pd.read_csv(solar_wind_file, chunksize=100000):
        chunk_list.append(chunk)
    solar_wind_df = pd.concat(chunk_list, ignore_index=True)
    print(f'Loaded solar wind data in chunks: {len(solar_wind_df)} rows', flush=True)

print('Converting time column...', flush=True)
solar_wind_df['time'] = pd.to_datetime(solar_wind_df['time'])
print('Time conversion completed', flush=True)

print('Preparing to merge data...', flush=True)
# Create a sorted index for faster searching
print('Sorting solar wind data by time...', flush=True)
solar_wind_df = solar_wind_df.sort_values('time').reset_index(drop=True)
print('Creating time array...', flush=True)
solar_wind_times = solar_wind_df['time'].values

# Constants for L1-Earth propagation
L1_EARTH_DISTANCE_KM = 1.5e6  # ~1.5 million km (typical L1 distance)
print(f'L1-Earth distance: {L1_EARTH_DISTANCE_KM/1e6:.2f} million km', flush=True)
print('Data preparation completed', flush=True)

def extract_datetime_from_filename(filename):
    """Extract datetime from filename like 'map_2024_1_1_13_48_0.npy'"""
    parts = filename.replace('.npy', '').split('_')
    if len(parts) >= 6 and parts[0] == 'map':
        year, month, day, hour, minute, second = parts[1:7]
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    return None

def calculate_earth_arrival_time(l1_time, velocity_km_s):
    """
    Calculate when L1 solar wind data arrives at Earth.

    Args:
        l1_time: L1 measurement time
        velocity_km_s: Solar wind velocity in km/s (will be made positive)

    Returns:
        Earth arrival time
    """
    # Use absolute value of velocity (it's negative in GSM coordinates)
    velocity = abs(velocity_km_s)

    # Skip invalid velocities
    if velocity <= 0 or velocity == 99999 or np.isnan(velocity):
        return None

    # Calculate travel time from L1 to Earth
    travel_time_seconds = L1_EARTH_DISTANCE_KM / velocity

    return l1_time + timedelta(seconds=travel_time_seconds)

def process_row(row_data):
    """
    Process a single map file to find matching L1 solar wind data.

    Strategy:
    1. Get map time (Earth observation time)
    2. For each L1 measurement, calculate when it arrives at Earth
    3. Find L1 measurement whose Earth arrival time is closest to map time
    """
    idx, row = row_data
    filename = row['filename']
    file_datetime = extract_datetime_from_filename(filename)

    if file_datetime is None:
        return None

    # This is the map time at Earth (ionosphere observation)
    map_time = pd.Timestamp(file_datetime)

    # Search window: L1 data from ~30 min to ~2 hours before map time
    # (typical propagation time is 1 hour, but can vary based on velocity)
    search_start = map_time - timedelta(hours=2)
    search_end = map_time - timedelta(minutes=30)

    # Find indices in this time range
    search_start_np = search_start.to_numpy()
    search_end_np = search_end.to_numpy()

    start_idx = solar_wind_times.searchsorted(search_start_np)
    end_idx = solar_wind_times.searchsorted(search_end_np)

    # Find best match by calculating Earth arrival time for each L1 measurement
    best_match_idx = None
    best_time_diff = None

    for i in range(start_idx, min(end_idx + 1, len(solar_wind_df))):
        l1_data = solar_wind_df.iloc[i]
        l1_time = pd.Timestamp(solar_wind_times[i])
        velocity = l1_data['proton_vx_gsm']

        # Calculate when this L1 measurement arrives at Earth
        earth_arrival_time = calculate_earth_arrival_time(l1_time, velocity)

        if earth_arrival_time is None:
            continue

        # Calculate time difference from map time
        time_diff_seconds = abs((earth_arrival_time - map_time).total_seconds())

        # Keep track of best match
        if best_match_idx is None or time_diff_seconds < best_time_diff:
            best_match_idx = i
            best_time_diff = time_diff_seconds

    # Accept match if within 5 minutes
    if best_match_idx is not None and best_time_diff <= 300:  # 5 minutes
        l1_data = solar_wind_df.iloc[best_match_idx]
        l1_time = pd.Timestamp(solar_wind_times[best_match_idx])
        velocity = abs(l1_data['proton_vx_gsm'])
        delay_minutes = (L1_EARTH_DISTANCE_KM / velocity) / 60

        return {
            'filename': filename,
            'float1': row['float1'],
            'float2': row['float2'],
            'float3': row['float3'],
            'float4': row['float4'],
            'proton_vx_gsm': l1_data['proton_vx_gsm'],
            'bx_gsm': l1_data['bx_gsm'],
            'by_gsm': l1_data['by_gsm'],
            'bz_gsm': l1_data['bz_gsm'],
            'l1_time': l1_time,
            'map_time': map_time,
            'propagation_delay_minutes': delay_minutes,
            'time_diff_seconds': best_time_diff
        }

    return None

print('Merging data...', flush=True)
# Process rows concurrently in batches to avoid memory issues
merged_data = []
batch_size = 1000  # Process 1000 rows at a time
total_rows = len(params_df)
processed = 0

print(f'Processing {total_rows} rows in batches of {batch_size} with 8 workers...', flush=True)

for batch_start in range(0, total_rows, batch_size):
    batch_end = min(batch_start + batch_size, total_rows)
    batch_df = params_df.iloc[batch_start:batch_end]
    
    print(f'Processing batch {batch_start}-{batch_end} ({len(batch_df)} rows)...', flush=True)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_row, (idx, row)): idx for idx, row in batch_df.iterrows()}
        
        batch_results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_start//batch_size + 1}"):
            result = future.result()
            if result is not None:
                batch_results.append(result)
        
        merged_data.extend(batch_results)
        processed += len(batch_df)
        
    print(f'Completed batch. Total processed: {processed}/{total_rows}, Total matches: {len(merged_data)}', flush=True)

# Create merged DataFrame
merged_df = pd.DataFrame(merged_data)

# Save the merged data
output_path = "/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/merged_params_solar_wind_v2.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged DataFrame created with {len(merged_df)} rows")
print(f"Original params data: {len(params_df)} rows")
print(f"Successfully matched: {len(merged_df)} rows")
print(f"Saved to: {output_path}")

if len(merged_df) > 0:
    print("\nPropagation delay statistics:")
    print(f"  Mean delay: {merged_df['propagation_delay_minutes'].mean():.2f} minutes")
    print(f"  Min delay: {merged_df['propagation_delay_minutes'].min():.2f} minutes")
    print(f"  Max delay: {merged_df['propagation_delay_minutes'].max():.2f} minutes")
    print(f"  Mean matching error: {merged_df['time_diff_seconds'].mean():.2f} seconds")

print("\nFirst few rows:")
print(merged_df.head())