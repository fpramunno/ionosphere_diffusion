# Ionosphere Diffusion - Data Pipeline

## Overview

The pipeline prepares paired (L1 solar wind conditions, ionosphere map) samples for model training.
Each sample links a solar wind measurement at L1 to the ionosphere map that corresponds to when
that solar wind arrived at Earth, accounting for the actual propagation delay.

---

## Data Sources

### Ionosphere Maps
- Location: `/capstor/scratch/cscs/framunno/ionosphere_data/all_maps/`
- Format: `.npy` files, named `map_YYYY_M_D_H_MM_0.npy`
- Coverage: 2020-2025, even minutes only (00, 02, 04, ..., 58)
- Count: ~1,574,960 files

### L1 Solar Wind (raw)
- Location: `/users/framunno/data/ionosphere/combined_f1m_m1m_YYYY.csv`
- Years: 2020, 2021, 2022, 2023, 2024, 2025
- Resolution: 1-minute
- Columns: `time`, `proton_vx_gsm`, `bx_gsm`, `by_gsm`, `bz_gsm`
- Missing values marked as `-99999` / `99999`

### DSCOVR Orbit (actual satellite position)
- Location: `/users/framunno/data/ionosphere/DSCOVR_ORBIT_PRE_YYYY.csv`
- Years: 2020-2025
- Key column: `X_GSE_km` — actual distance along the Sun-Earth line, used to compute propagation delay

---

## Pipeline Steps

### Step 1 - Interpolate L1 gaps
**Script:** `/users/framunno/projects/ionosphere_diffusion/interpolate_l1_gaps.py`

Replaces `-99999` sentinel values with NaN and linearly interpolates gaps up to `MAX_GAP = 10` minutes.
Gaps longer than 10 minutes remain as NaN.

```bash
python interpolate_l1_gaps.py combined_f1m_m1m_YYYY.csv
# output: combined_f1m_m1m_YYYY_interpolated.csv
```

Output per year:
- `/users/framunno/data/ionosphere/combined_f1m_m1m_2020_interpolated.csv`
- `/users/framunno/data/ionosphere/combined_f1m_m1m_2021_interpolated.csv`
- `/users/framunno/data/ionosphere/combined_f1m_m1m_2022_interpolated.csv`
- `/users/framunno/data/ionosphere/combined_f1m_m1m_2023_interpolated.csv`
- `/users/framunno/data/ionosphere/combined_f1m_m1m_2024_interpolated.csv`
- `/users/framunno/data/ionosphere/combined_f1m_m1m_2025_interpolated.csv`

### Step 2 - Concatenate per-year files
Merge all 6 interpolated L1 CSVs and all 6 DSCOVR orbit CSVs into single files.

Output:
- `/users/framunno/data/ionosphere/combined_f1m_m1m_2020_2025_interpolated.csv`
- `/users/framunno/data/ionosphere/DSCOVR_ORBIT_PRE_2020_2025.csv`

### Step 3 - Match L1 to ionosphere maps
**Script:** `/users/framunno/projects/ionosphere_diffusion/merge_l1_to_maps_even_minutes.py`
**Job script:** `/users/framunno/projects/ionosphere_diffusion/scripts/merge_l1_maps.sh`

For each valid L1 measurement:
1. Joins with DSCOVR orbit to get the real `X_GSE_km` distance at that timestamp
2. Computes Earth arrival time: `t_arrival = t_l1 + X_GSE_km / |vx_gsm|`
3. Finds the nearest even-minute ionosphere map to `t_arrival` (binary search)
4. Accepts the match if the map is within 60 seconds of the arrival time

Only rows where all 4 conditions (`proton_vx_gsm`, `bx_gsm`, `by_gsm`, `bz_gsm`) are valid are matched.

```bash
sbatch /users/framunno/projects/ionosphere_diffusion/scripts/merge_l1_maps.sh
```

Output: `/users/framunno/data/ionosphere/l1_to_map_matched_2020_2025.csv`

Output columns:
| Column | Description |
|--------|-------------|
| `time` | L1 measurement timestamp |
| `proton_vx_gsm` | Solar wind velocity (km/s) |
| `bx_gsm`, `by_gsm`, `bz_gsm` | IMF components (nT) |
| `x_gse_km` | Actual DSCOVR distance used for delay |
| `speed_kms` | `abs(proton_vx_gsm)` |
| `delay_min` | Propagation delay in minutes |
| `earth_time` | When solar wind arrived at Earth |
| `time_map` | Timestamp of matched map file |
| `dt_diff` | Difference between `earth_time` and `time_map` |
| `filename` | `.npy` map file to load for training |

---

## Reference results (2024 only, fixed 1.5e6 km distance)
- L1 records: 522,720
- Matched: 496,235 (94.93% match rate)
- Mean propagation delay: 60.61 min (std: 9.25 min, range: 30-99 min)
- Mean matching error: 30 seconds

---

## Gap Analysis Notebook
**Location:** `/users/framunno/data/ionosphere/gap_analysis.ipynb`

Analyzes missing data and gap statistics per year and parameter before interpolation.
Sections: missing value counts, gap counts, gap width distributions, interpolation recovery curves, gap timelines.
