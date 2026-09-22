# -*- coding: utf-8 -*-
"""Pre-compute and cache per-sequence dynamics scores (mean abs frame-to-frame change).

Run this once before generation to avoid the slow scoring loop at dataset init time.
The output JSON maps str(center_idx) → float score (V/frame, raw physical units).

Usage
-----
python precompute_dynamics_scores.py \
    --csv-path ./data/ionosphere/l1_to_map_matched_2020_2025.csv \
    --cache-path ./data_root/dynamics_scores_val_seq50.json \
    --split valid \
    --sequence-length 50 \
    --num-workers 16

# Inspect which sequences pass a given threshold without running generation:
python precompute_dynamics_scores.py \
    --cache-path ./data_root/dynamics_scores_val_seq50.json \
    --inspect --dynamics-filter 0.75
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

MAP_DIR = './data_root/ionosphere_data/all_maps/'

# ---------------------------------------------------------------------------
# Per-worker globals — set once by the initializer, never re-pickled per task
# ---------------------------------------------------------------------------
_G_all_files      = None
_G_seq_len        = None
_G_cartesian      = None
_G_output_size    = None


def _worker_init(all_files, seq_len, cartesian, output_size):
    global _G_all_files, _G_seq_len, _G_cartesian, _G_output_size
    _G_all_files   = all_files
    _G_seq_len     = seq_len
    _G_cartesian   = cartesian
    _G_output_size = output_size


def _worker_score(center_idx):
    """Score one sequence; uses globals set by _worker_init.
    Returns 0.0 if any frame is missing or unloadable — only fully complete sequences get real scores.
    """
    from src.data.dataset import latlon_to_cartesian_grid
    start_idx = center_idx - _G_seq_len // 2
    frames = []
    for frame_offset in range(_G_seq_len):
        file_idx = start_idx + frame_offset
        if not (0 <= file_idx < len(_G_all_files)):
            return center_idx, 0.0
        try:
            data = np.load(MAP_DIR + _G_all_files[file_idx], allow_pickle=True)
            data_map = data[0].astype(np.float32)
            if _G_cartesian:
                data_map = latlon_to_cartesian_grid(data_map, output_size=_G_output_size)
            frames.append(data_map)
        except Exception:
            return center_idx, 0.0
    arr = np.stack(frames)
    return center_idx, float(np.abs(np.diff(arr, axis=0)).mean())


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv-path', type=str,
                   default='./data/ionosphere/l1_to_map_matched_2020_2025.csv')
    p.add_argument('--cache-path', type=str,
                   default='./data_root/dynamics_scores_val_seq50.json',
                   help='where to save (or load) the scores JSON')
    p.add_argument('--split',            type=str, default='valid')
    p.add_argument('--sequence-length',  type=int, default=50)
    p.add_argument('--cartesian-transform', action='store_true', default=True)
    p.add_argument('--no-cartesian-transform', dest='cartesian_transform', action='store_false')
    p.add_argument('--output-size',      type=int, default=64)
    p.add_argument('--num-workers',      type=int, default=os.cpu_count(),
                   help='parallel worker processes for scoring')
    p.add_argument('--chunksize',        type=int, default=32,
                   help='sequences per worker chunk (larger = less IPC overhead)')
    p.add_argument('--inspect',          action='store_true',
                   help='print stats from existing cache and exit')
    p.add_argument('--dynamics-filter',  type=float, default=None, metavar='QUANTILE',
                   help='with --inspect: show how many sequences pass this threshold')
    p.add_argument('--activity-filter',  type=str, default=None,
                   choices=['high', 'moderate', 'low'],
                   help='filter sequences by epsilon activity level before scoring')
    p.add_argument('--epsilon-high-quantile', type=float, default=0.75)
    p.add_argument('--epsilon-low-quantile',  type=float, default=0.25)
    args = p.parse_args()

    # ------------------------------------------------------------------
    # Inspect mode: just print stats from an existing cache
    # ------------------------------------------------------------------
    if args.inspect:
        if not os.path.exists(args.cache_path):
            print(f'Cache not found: {args.cache_path}')
            print('Run without --inspect first to compute scores.')
            sys.exit(1)
        with open(args.cache_path) as f:
            raw = json.load(f)
        scores = np.array(list(raw.values()), dtype=np.float32)
        print(f'Cache: {args.cache_path}')
        print(f'Total sequences scored: {len(scores)}')
        print('Score stats (V/frame):')
        for q in [0, 25, 50, 75, 90, 95, 99, 100]:
            print(f'  p{q:3d}: {np.percentile(scores, q):8.1f}')
        if args.dynamics_filter is not None:
            threshold = float(np.quantile(scores, args.dynamics_filter))
            n_pass = int((scores >= threshold).sum())
            pct = (1 - args.dynamics_filter) * 100
            print(f'\nFilter quantile={args.dynamics_filter} (top {pct:.0f}%): '
                  f'{n_pass}/{len(scores)} sequences pass  (threshold={threshold:.1f} V/frame)')
        return

    # ------------------------------------------------------------------
    # Load dataset (no dynamics filter — we just need sequences + all_files)
    # ------------------------------------------------------------------
    from src.data.dataset import get_sequence_data_objects_iterable

    print(f'Loading dataset ({args.split} split, seq_len={args.sequence_length})…')
    val_dataset, _, _ = get_sequence_data_objects_iterable(
        csv_path=args.csv_path,
        batch_size=1,
        num_data_workers=0,
        split=args.split,
        seed=42,
        sequence_length=args.sequence_length,
        normalization_type='absolute_max',
        use_l1_conditions=True,
        min_center_distance=30,
        cartesian_transform=args.cartesian_transform,
        output_size=args.output_size,
        only_complete_sequences=True,
        activity_filter=args.activity_filter,
        epsilon_high_quantile=args.epsilon_high_quantile,
        epsilon_low_quantile=args.epsilon_low_quantile,
    )
    sequences = val_dataset.sequences
    all_files = val_dataset.all_files
    print(f'Total sequences in dataset: {len(sequences)}')

    # ------------------------------------------------------------------
    # Load existing cache — skip already-scored sequences
    # ------------------------------------------------------------------
    cached = {}
    if os.path.exists(args.cache_path):
        with open(args.cache_path) as f:
            cached = {int(k): float(v) for k, v in json.load(f).items()}
        print(f'Cache exists with {len(cached)} entries — skipping already-scored.')

    to_score = [idx for idx in sequences if idx not in cached]
    print(f'Need to score: {len(to_score)} sequences  (workers={args.num_workers}, chunksize={args.chunksize})')

    # ------------------------------------------------------------------
    # Parallel scoring
    # ------------------------------------------------------------------
    new_scores = {}
    if to_score:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_worker_init,
            initargs=(all_files, args.sequence_length, args.cartesian_transform, args.output_size),
        ) as pool:
            futures = {pool.submit(_worker_score, idx): idx for idx in to_score}
            with tqdm(total=len(futures), desc='Scoring') as bar:
                for fut in as_completed(futures):
                    center_idx, score = fut.result()
                    new_scores[center_idx] = score
                    bar.update(1)

    combined = {**cached, **new_scores}
    os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)), exist_ok=True)
    with open(args.cache_path, 'w') as f:
        json.dump({str(k): v for k, v in combined.items()}, f)
    print(f'\nSaved {len(combined)} scores → {args.cache_path}')

    scores_arr = np.array([combined[idx] for idx in sequences if idx in combined], dtype=np.float32)
    print('Score stats (V/frame):')
    for q in [0, 25, 50, 75, 90, 95, 100]:
        print(f'  p{q:3d}: {np.percentile(scores_arr, q):8.1f}')


if __name__ == '__main__':
    main()
