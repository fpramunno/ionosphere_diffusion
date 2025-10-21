"""
Analyze current vs proposed sequence building strategy.

Current: Reject sequences with any missing frames, non-overlapping
Proposed: Fill missing frames with zeros, overlapping with min center distance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse


def extract_timestamp(filename):
    """Extract timestamp from filename.

    Expected format: map_2024_1_6_1_30_0.npy (year_month_day_hour_minute_second)
    or map_20240106_013000.npy (yearmonthday_hourminutesecond)
    """
    try:
        # Extract just the filename without path
        basename = filename.split('/')[-1]
        parts = basename.replace('.npy', '').split('_')

        # Try new format: map_2024_1_6_1_30_0
        if len(parts) == 7 and parts[0] == 'map':
            year, month, day, hour, minute, second = parts[1:7]
            return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

        # Try old format: map_20240106_013000
        elif len(parts) == 3 and parts[0] == 'map':
            date_str = parts[1]
            time_str = parts[2]
            dt_str = f"{date_str}_{time_str}"
            return datetime.strptime(dt_str, '%Y%m%d_%H%M%S')

        return None
    except Exception as e:
        return None


def build_current_sequences(timestamps, sequence_length=5):
    """Current approach: Non-overlapping, reject if any frame missing."""
    sequences = []
    n = len(timestamps)
    i = 0
    rejected = 0

    while i + sequence_length <= n:
        seq_times = timestamps[i:i+sequence_length]
        is_consistent = True

        for j in range(1, sequence_length):
            delta = (seq_times[j] - seq_times[j-1]).total_seconds()
            if delta != 120:  # 2 minutes = 120 seconds
                is_consistent = False
                break

        if is_consistent:
            sequences.append(i)  # Store center index
            i += sequence_length  # Non-overlapping
        else:
            rejected += 1
            i += 1

    return sequences, rejected


def build_proposed_sequences(timestamps, sequence_length=30, min_center_distance=10):
    """
    Proposed approach: Overlapping with minimum center distance, fill missing frames.

    Args:
        timestamps: List of available timestamps
        sequence_length: Length of each sequence
        min_center_distance: Minimum distance between sequence centers (in frames)
    """
    sequences = []
    n = len(timestamps)

    # For each potential starting position
    i = 0
    while i + sequence_length <= n:
        # Calculate center index
        center_idx = i + sequence_length // 2

        # Check if we have minimum distance from last sequence's center
        if not sequences or (center_idx - sequences[-1] >= min_center_distance):
            sequences.append(center_idx)
            i += min_center_distance  # Move by minimum distance
        else:
            i += 1

    return sequences


def count_missing_frames(timestamps, sequences, sequence_length):
    """Count how many frames would need to be filled with zeros."""
    total_missing = 0
    missing_per_seq = []

    for center_idx in sequences:
        start_idx = center_idx - sequence_length // 2
        end_idx = start_idx + sequence_length

        # Count missing frames in this sequence
        missing = 0
        for j in range(start_idx + 1, end_idx):
            if j < len(timestamps):
                delta = (timestamps[j] - timestamps[j-1]).total_seconds()
                if delta != 120:
                    # Calculate how many 2-min frames are missing
                    missing += int(delta / 120) - 1

        missing_per_seq.append(missing)
        total_missing += missing

    return total_missing, missing_per_seq


def analyze_strategies(csv_path, sequence_length=30, min_center_distance=10):
    """Compare current vs proposed sequence building strategies."""
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Extract timestamps
    df['timestamp'] = df['filename'].apply(extract_timestamp)
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    timestamps = df['timestamp'].tolist()
    total_files = len(timestamps)

    print(f"\nTotal available files: {total_files}")
    print(f"Sequence length: {sequence_length}")
    print("="*80)

    # Current approach
    print(f"\nCURRENT APPROACH:")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Strategy: Non-overlapping, reject if any frame missing")
    print("-"*80)

    current_seqs, rejected = build_current_sequences(timestamps, sequence_length)

    print(f"  Results:")
    print(f"    - Valid sequences: {len(current_seqs)}")
    print(f"    - Rejected sequences: {rejected}")
    print(f"    - Total frames used: {len(current_seqs) * sequence_length}")
    if total_files > 0:
        print(f"    - Frame usage: {(len(current_seqs) * sequence_length) / total_files * 100:.2f}%")
    else:
        print(f"    - Frame usage: N/A (no files)")

    # Proposed approach
    print(f"\nPROPOSED APPROACH:")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Strategy: Overlapping with min center distance = {min_center_distance}")
    print(f"  - Fill missing frames with zeros")
    print("-"*80)

    proposed_seqs = build_proposed_sequences(timestamps, sequence_length, min_center_distance)

    # Count unique frames used
    unique_frames = set()
    for center_idx in proposed_seqs:
        start_idx = center_idx - sequence_length // 2
        end_idx = start_idx + sequence_length
        for idx in range(max(0, start_idx), min(total_files, end_idx)):
            unique_frames.add(idx)

    total_missing, missing_per_seq = count_missing_frames(timestamps, proposed_seqs, sequence_length)

    print(f"  Results:")
    print(f"    - Valid sequences: {len(proposed_seqs)}")
    print(f"    - Rejected sequences: 0 (all included!)")
    print(f"    - Unique frames used: {len(unique_frames)}")
    print(f"    - Frame usage: {len(unique_frames) / total_files * 100:.2f}%")
    print(f"    - Frames that will be filled with zeros: {total_missing}")
    print(f"    - Avg missing frames per sequence: {np.mean(missing_per_seq):.2f}")
    print(f"    - Max missing frames in a sequence: {max(missing_per_seq) if missing_per_seq else 0}")

    # Comparison
    print(f"\n" + "="*80)
    print(f"COMPARISON (same sequence length = {sequence_length}):")
    print("-"*80)
    print(f"  Sequence gain: {len(proposed_seqs)} vs {len(current_seqs)} "
          f"({len(proposed_seqs) / len(current_seqs):.2f}x more)" if len(current_seqs) > 0 else "  Infinite gain (current has 0 sequences)")
    print(f"  Frame utilization: {len(unique_frames) / total_files * 100:.2f}% vs "
          f"{(len(current_seqs) * sequence_length) / total_files * 100:.2f}%")
    print(f"  Zero-filled frames: {total_missing} ({total_missing / (len(proposed_seqs) * sequence_length) * 100:.2f}% of total frames)" if len(proposed_seqs) > 0 else "  No sequences")

    # Split analysis
    print(f"\n" + "="*80)
    print(f"SPLIT ANALYSIS (by month):")
    print("-"*80)

    splits = {
        'Train (Jan-Aug)': range(1, 9),
        'Valid (Sep-Oct)': range(9, 11),
        'Test (Nov-Dec)': range(11, 13)
    }

    print(f"\n{'Split':<20} {'Current':<15} {'Proposed':<15} {'Improvement':<15}")
    print("-"*80)

    for split_name, month_range in splits.items():
        # Current approach
        current_split = sum(1 for idx in current_seqs if timestamps[idx].month in month_range)

        # Proposed approach
        proposed_split = sum(1 for idx in proposed_seqs if timestamps[idx].month in month_range)

        improvement = proposed_split / current_split if current_split > 0 else float('inf')

        print(f"{split_name:<20} {current_split:<15} {proposed_split:<15} {improvement:.2f}x")

    # Test different min_center_distances
    print(f"\n" + "="*80)
    print(f"EFFECT OF MIN CENTER DISTANCE (seq_length={sequence_length}):")
    print("-"*80)
    print(f"{'Distance':<15} {'Sequences':<15} {'Overlap %':<15} {'Notes':<30}")
    print("-"*80)

    test_distances = [1, 5, 10, 15, 20, sequence_length]
    for dist in test_distances:
        seqs = build_proposed_sequences(timestamps, sequence_length, dist)
        overlap_pct = ((sequence_length - dist) / sequence_length) * 100 if dist < sequence_length else 0

        notes = ""
        if dist == 1:
            notes = "Maximum overlap"
        elif dist == sequence_length:
            notes = "No overlap (like current)"
        elif dist == sequence_length // 2:
            notes = "50% overlap"

        print(f"{dist:<15} {len(seqs):<15} {overlap_pct:.1f}%{'':<10} {notes:<30}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze sequence building strategies')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
    parser.add_argument('--min-dist', type=int, default=10, help='Minimum center distance')

    args = parser.parse_args()

    analyze_strategies(args.csv, args.seq_len, args.min_dist)
