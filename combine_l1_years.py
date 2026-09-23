"""
Combine per-year interpolated L1 solar wind CSVs into one multi-year file.

interpolate_l1_gaps.py is run once per year (raw yearly downloads are
per-year); merge_l1_to_maps_even_minutes.py then expects a single combined
multi-year CSV as its SOLAR_WIND_FILE. This concatenates the per-year
interpolated files into that combined file.

Usage
-----
python combine_l1_years.py \
    --inputs ./data/ionosphere/combined_f1m_m1m_2020_interpolated.csv \
             ./data/ionosphere/combined_f1m_m1m_2021_interpolated.csv \
             ./data/ionosphere/combined_f1m_m1m_2022_interpolated.csv \
             ./data/ionosphere/combined_f1m_m1m_2023_interpolated.csv \
             ./data/ionosphere/combined_f1m_m1m_2024_interpolated.csv \
             ./data/ionosphere/combined_f1m_m1m_2025_interpolated.csv \
    --output ./data/ionosphere/combined_f1m_m1m_2020_2025_interpolated.csv
"""

import argparse
import pandas as pd


def combine(input_paths, output_path):
    frames = []
    for path in input_paths:
        df = pd.read_csv(path)
        print(f"{path}: {len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    if 'time' in combined.columns:
        combined['time'] = pd.to_datetime(combined['time'])
        combined = combined.sort_values('time').reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    print(f"\nCombined: {len(combined):,} rows total")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=str, nargs='+', required=True, help="Per-year interpolated CSVs, in any order")
    p.add_argument("--output", type=str, required=True, help="Path to save the combined multi-year CSV")
    args = p.parse_args()
    combine(args.inputs, args.output)
