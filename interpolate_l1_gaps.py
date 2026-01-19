"""
L1 Solar Wind Data Gap Interpolation
=====================================

Script to interpolate missing values in L1 solar wind condition data.
Uses linear interpolation for gaps up to a maximum duration.

Input: Raw L1 data CSV with missing/invalid values
Output: Cleaned L1 data CSV with interpolated values + statistics report

Author: Francesco Ramunno
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = '/users/framunno/data/ionosphere/combined_f1m_m1m_2024.csv'
OUTPUT_FILE = '/users/framunno/data/ionosphere/combined_f1m_m1m_2024_interpolated.csv'
MAX_GAP_MINUTES = 10  # Maximum gap size to interpolate across

# L1 condition columns to process
L1_CONDITIONS = ['proton_vx_gsm', 'bx_gsm', 'by_gsm', 'bz_gsm']

# Invalid value markers
INVALID_VALUES = [99999, -99999]

# ============================================================================
# INTERPOLATION FUNCTIONS
# ============================================================================

def identify_gaps(series, time_series, show_progress=False):
    """
    Identify gaps (consecutive missing values) in a time series.

    Args:
        series: Data series with NaN for missing values
        time_series: Corresponding timestamps
        show_progress: Whether to show progress bar

    Returns:
        List of dictionaries with gap information
    """
    gaps = []
    in_gap = False
    gap_start_idx = None

    iterator = tqdm(range(len(series)), desc="  Identifying gaps", leave=False) if show_progress and len(series) > 100000 else range(len(series))

    for idx in iterator:
        is_missing = pd.isna(series.iloc[idx])

        if is_missing and not in_gap:
            # Start of a new gap
            in_gap = True
            gap_start_idx = idx

        elif not is_missing and in_gap:
            # End of gap
            gap_end_idx = idx - 1

            # Calculate gap duration
            if gap_start_idx > 0 and gap_end_idx < len(series) - 1:
                gap_start_time = time_series.iloc[gap_start_idx]
                gap_end_time = time_series.iloc[gap_end_idx]
                gap_duration = (gap_end_time - gap_start_time).total_seconds() / 60

                gaps.append({
                    'start_idx': gap_start_idx,
                    'end_idx': gap_end_idx,
                    'num_points': gap_end_idx - gap_start_idx + 1,
                    'duration_minutes': gap_duration
                })

            in_gap = False
            gap_start_idx = None

    # Handle case where series ends in a gap
    if in_gap and gap_start_idx is not None:
        gap_end_idx = len(series) - 1
        if gap_start_idx < gap_end_idx:
            gap_start_time = time_series.iloc[gap_start_idx]
            gap_end_time = time_series.iloc[gap_end_idx]
            gap_duration = (gap_end_time - gap_start_time).total_seconds() / 60

            gaps.append({
                'start_idx': gap_start_idx,
                'end_idx': gap_end_idx,
                'num_points': gap_end_idx - gap_start_idx + 1,
                'duration_minutes': gap_duration
            })

    return gaps


def interpolate_with_gap_limit(df, column, max_gap_minutes):
    """
    Interpolate missing values in a column, but only for gaps <= max_gap_minutes.

    Args:
        df: DataFrame with time-sorted data
        column: Column name to interpolate
        max_gap_minutes: Maximum gap duration to interpolate

    Returns:
        series: Interpolated series
        stats: Dictionary with interpolation statistics
    """
    # Make a copy of the column
    series = df[column].copy()
    time_series = df['time']

    # Count missing before
    missing_before = series.isna().sum()

    if missing_before == 0:
        return series, {
            'missing_before': 0,
            'interpolated': 0,
            'still_missing': 0,
            'longest_gap_filled': 0,
            'gaps_too_large': 0,
            'gaps_filled': 0
        }

    # Identify all gaps
    gaps = identify_gaps(series, time_series, show_progress=True)

    # Separate gaps into fillable and too-large
    fillable_gaps = [g for g in gaps if g['duration_minutes'] <= max_gap_minutes]
    too_large_gaps = [g for g in gaps if g['duration_minutes'] > max_gap_minutes]

    # Interpolate only fillable gaps
    for gap in tqdm(fillable_gaps, desc=f"  Filling gaps", leave=False, disable=len(fillable_gaps) < 10):
        start_idx = gap['start_idx']
        end_idx = gap['end_idx']

        # Get the valid values before and after the gap
        if start_idx > 0 and end_idx < len(series) - 1:
            before_idx = start_idx - 1
            after_idx = end_idx + 1

            before_val = series.iloc[before_idx]
            after_val = series.iloc[after_idx]

            before_time = time_series.iloc[before_idx]
            after_time = time_series.iloc[after_idx]

            # Linear interpolation based on time
            for idx in range(start_idx, end_idx + 1):
                current_time = time_series.iloc[idx]
                # Calculate interpolation weight based on time position
                time_fraction = (current_time - before_time).total_seconds() / \
                               (after_time - before_time).total_seconds()

                interpolated_val = before_val + (after_val - before_val) * time_fraction
                series.iloc[idx] = interpolated_val

    # Calculate statistics
    missing_after = series.isna().sum()
    interpolated_count = missing_before - missing_after
    longest_gap_filled = max([g['duration_minutes'] for g in fillable_gaps]) if fillable_gaps else 0

    stats = {
        'missing_before': missing_before,
        'interpolated': interpolated_count,
        'still_missing': missing_after,
        'longest_gap_filled': longest_gap_filled,
        'gaps_too_large': len(too_large_gaps),
        'gaps_filled': len(fillable_gaps)
    }

    return series, stats


def interpolate_l1_data(df, max_gap_minutes=10):
    """
    Interpolate missing L1 solar wind condition values.

    Args:
        df: DataFrame with L1 data (must have 'time' column and be sorted by time)
        max_gap_minutes: Maximum gap duration to interpolate

    Returns:
        df_interpolated: DataFrame with interpolated values
        all_stats: Dictionary with statistics for each column
    """
    df = df.copy()

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)

    # Mark invalid values as NaN
    for col in L1_CONDITIONS:
        df.loc[df[col].isin(INVALID_VALUES), col] = np.nan

    # Interpolate each condition
    all_stats = {}
    print("\nInterpolating L1 conditions:")
    for col in tqdm(L1_CONDITIONS, desc="Processing conditions"):
        df[col], stats = interpolate_with_gap_limit(df, col, max_gap_minutes)
        all_stats[col] = stats

    return df, all_stats


def print_statistics(all_stats):
    """
    Print formatted interpolation statistics.
    """
    print("\n" + "=" * 80)
    print("L1 DATA INTERPOLATION STATISTICS")
    print("=" * 80)

    for col in L1_CONDITIONS:
        stats = all_stats[col]
        print(f"\n{col}:")
        print(f"  Missing before:           {stats['missing_before']:,}")
        print(f"  Interpolated:             {stats['interpolated']:,}")
        print(f"  Still missing:            {stats['still_missing']:,}")

        if stats['missing_before'] > 0:
            recovery_rate = stats['interpolated'] / stats['missing_before'] * 100
            print(f"  Recovery rate:            {recovery_rate:.2f}%")

        print(f"  Gaps filled:              {stats['gaps_filled']:,}")
        print(f"  Gaps too large (skipped): {stats['gaps_too_large']:,}")
        print(f"  Longest gap filled:       {stats['longest_gap_filled']:.2f} minutes")

    # Overall statistics
    total_missing_before = sum(s['missing_before'] for s in all_stats.values())
    total_interpolated = sum(s['interpolated'] for s in all_stats.values())
    total_still_missing = sum(s['still_missing'] for s in all_stats.values())

    print("\n" + "-" * 80)
    print("OVERALL:")
    print(f"  Total missing before:     {total_missing_before:,}")
    print(f"  Total interpolated:       {total_interpolated:,}")
    print(f"  Total still missing:      {total_still_missing:,}")

    if total_missing_before > 0:
        overall_recovery = total_interpolated / total_missing_before * 100
        print(f"  Overall recovery rate:    {overall_recovery:.2f}%")

    print("=" * 80)


def count_valid_rows(df):
    """
    Count how many rows have all L1 conditions valid.
    """
    valid_mask = pd.Series([True] * len(df), index=df.index)

    for col in L1_CONDITIONS:
        valid_mask &= ~df[col].isna()

    return valid_mask.sum()


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interpolate missing L1 solar wind data")
    parser.add_argument("--input", type=str, default=INPUT_FILE,
                       help="Input CSV file with raw L1 data")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                       help="Output CSV file with interpolated data")
    parser.add_argument("--max_gap", type=int, default=MAX_GAP_MINUTES,
                       help="Maximum gap size (minutes) to interpolate")
    args = parser.parse_args()

    print("=" * 80)
    print("L1 SOLAR WIND DATA GAP INTERPOLATION")
    print("=" * 80)
    print(f"Input file:       {args.input}")
    print(f"Output file:      {args.output}")
    print(f"Max gap duration: {args.max_gap} minutes")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading L1 data...")
    df = pd.read_csv(args.input)
    df['time'] = pd.to_datetime(df['time'])

    # Remove timezone if present
    if df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_localize(None)

    print(f"Loaded {len(df):,} records")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")

    # Mark invalid values as NaN first to get accurate "before" count
    df_temp = df.copy()
    for col in L1_CONDITIONS:
        df_temp.loc[df_temp[col].isin(INVALID_VALUES), col] = np.nan

    # Count valid rows before interpolation (after marking invalid as NaN)
    valid_before = count_valid_rows(df_temp)
    print(f"Rows with all conditions valid (before interpolation): {valid_before:,} ({valid_before/len(df)*100:.2f}%)")

    # Count total missing values
    total_missing_before = sum(df_temp[col].isna().sum() for col in L1_CONDITIONS)
    print(f"Total missing/invalid values: {total_missing_before:,}")

    # Interpolate
    print("\n[2/4] Interpolating missing values...")
    df_interpolated, stats = interpolate_l1_data(df, max_gap_minutes=args.max_gap)

    # Print statistics
    print("\n[3/4] Interpolation results:")
    print_statistics(stats)

    # Count valid rows after
    valid_after = count_valid_rows(df_interpolated)
    rows_recovered = valid_after - valid_before
    print(f"\nRows with all conditions valid (after interpolation): {valid_after:,} ({valid_after/len(df)*100:.2f}%)")
    print(f"Additional complete rows recovered: {rows_recovered:,}")

    if rows_recovered > 0:
        print(f"  → Interpolation recovered {rows_recovered:,} additional usable data points!")
    elif rows_recovered == 0:
        print(f"  → No additional complete rows recovered (gaps too large)")
    else:
        print(f"  → Note: Some rows had partial recovery (not all 4 conditions filled)")

    # Save
    print(f"\n[4/4] Saving interpolated data to {args.output}...")
    df_interpolated.to_csv(args.output, index=False)

    print("\n" + "=" * 80)
    print("INTERPOLATION COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
