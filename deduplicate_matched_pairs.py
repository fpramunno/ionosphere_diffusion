"""
Deduplicate L1-to-map matched pairs.

merge_l1_to_maps_even_minutes.py / merge_l1_to_maps_2015_event.py produce one
row per L1 measurement (~1 min cadence), each matched to its nearest
ionosphere map (~2 min cadence). Since maps are coarser than L1, several L1
rows legitimately match the same map file, so the same map appears multiple
times in the raw matched output. This keeps, for each unique map filename,
only the single L1 match with the smallest |dt_diff| (map_time - L1 Earth-
arrival time) — i.e. each map is associated with exactly one, best-matching
L1 row.

Usage
-----
python deduplicate_matched_pairs.py \
    --input ./data/ionosphere/l1_to_map_matched_2020_2025.csv \
    --output ./data/ionosphere/l1_to_map_matched_2020_2025_deduplicated.csv
"""

import argparse
import pandas as pd


def deduplicate(input_path, output_path):
    df = pd.read_csv(input_path)
    before = len(df)

    dt_diff_seconds = pd.to_timedelta(df['dt_diff']).dt.total_seconds().abs()
    df = df.assign(_abs_dt_diff=dt_diff_seconds)
    df = df.sort_values('_abs_dt_diff').drop_duplicates(subset='filename', keep='first')
    df = df.drop(columns='_abs_dt_diff').sort_values('time').reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Rows before: {before:,}")
    print(f"Rows after (one per unique map): {len(df):,}")
    print(f"Removed {before - len(df):,} duplicate-map rows")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=str, required=True, help="Matched-pairs CSV from a merge_l1_to_maps_*.py script")
    p.add_argument("--output", type=str, required=True, help="Path to save the deduplicated CSV")
    args = p.parse_args()
    deduplicate(args.input, args.output)
