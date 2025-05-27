import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

data_dir = "/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/pickled_maps"
files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

def extract_metadata(fname):
    try:
        fpath = os.path.join(data_dir, fname)
        arr = np.load(fpath, allow_pickle=True)
        if isinstance(arr, (list, tuple)) and len(arr) == 5:
            return (fname, arr[1], arr[2], arr[3], arr[4])
    except Exception as e:
        print(f"Error loading {fname}: {e}")
    return None



results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(extract_metadata, fname): fname for fname in files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
        result = future.result()
        if result is not None:
            results.append(result)

# Create DataFrame
df = pd.DataFrame(results, columns=["filename", "float1", "float2", "float3", "float4"])

# Save to CSV
df.to_csv(
    "/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/params.csv",
    index=False
)
