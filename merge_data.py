import pandas as pd
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

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
print('Data preparation completed', flush=True)

def extract_datetime_from_filename(filename):
    """Extract datetime from filename like 'map_2024_1_1_13_48_0.npy'"""
    parts = filename.replace('.npy', '').split('_')
    if len(parts) >= 6 and parts[0] == 'map':
        year, month, day, hour, minute, second = parts[1:7]
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    return None

def process_row(row_data):
    """Process a single row to find matching solar wind data"""
    idx, row = row_data
    filename = row['filename']
    file_datetime = extract_datetime_from_filename(filename)
    
    if file_datetime is None:
        return None
    
    # Convert to pandas timestamp for comparison
    target_time = pd.Timestamp(file_datetime)
    
    # Find closest time using searchsorted for O(log n) complexity
    # Convert target_time to numpy datetime64 to match solar_wind_times array type
    target_time_np = target_time.to_numpy()
    insert_idx = solar_wind_times.searchsorted(target_time_np)
    
    # Check nearby indices to find the closest match
    candidates = []
    for i in [insert_idx - 1, insert_idx]:
        if 0 <= i < len(solar_wind_times):
            # Convert both to pandas timestamps for comparison
            solar_time = pd.Timestamp(solar_wind_times[i])
            time_diff = abs(solar_time - target_time)
            candidates.append((i, time_diff))
    
    if not candidates:
        return None
    
    # Find the closest match within 5 minutes
    closest_idx, min_diff = min(candidates, key=lambda x: x[1])
    if min_diff.total_seconds() / 60 <= 5:  # Within 5 minutes
        solar_wind_data = solar_wind_df.iloc[closest_idx]
        return {
            'filename': filename,
            'float1': row['float1'],
            'float2': row['float2'],
            'float3': row['float3'],
            'float4': row['float4'],
            'proton_vx_gsm': solar_wind_data['proton_vx_gsm'],
            'bx_gsm': solar_wind_data['bx_gsm'],
            'by_gsm': solar_wind_data['by_gsm'],
            'bz_gsm': solar_wind_data['bz_gsm'],
            'time': solar_wind_data['time']
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
output_path = "/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/merged_params_solar_wind.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged DataFrame created with {len(merged_df)} rows")
print(f"Original params data: {len(params_df)} rows")
print(f"Successfully matched: {len(merged_df)} rows")
print(f"Saved to: {output_path}")
print("\nFirst few rows:")
print(merged_df.head())