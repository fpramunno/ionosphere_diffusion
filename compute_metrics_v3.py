# -*- coding: utf-8 -*-
import torch
torch.backends.cudnn.enabled = False  # required on H100 for Conv2d under bf16 (LPIPS)

"""Compute metrics from pre-saved .npy inference results (singlepass, short sequences).

Extends compute_metrics_v2.py's per-frame pixel metrics (reused as-is via
compute_metrics_batch: PSNR/SSIM/LPIPS/power-spectrum/...) with NRMSE of three
scalar summary statistics (mean, std, CPCP) of the field, computed the same way
as in notebooks/explore_2015_timeseries.ipynb:

  NRMSE = RMSE(GT, pred) / std(GT)

For models with multiple samples per sequence (diffusion: n_samples > 1), NRMSE
is computed separately PER SAMPLE (RMSE pooled over all seq_idx/frame_idx of a
group, normalized by that group's pooled GT std), then reported as mean +/- std
ACROSS SAMPLES. This isolates sample-to-sample (epistemic) spread from the
frame-to-frame variability the pooled RMSE already averages over. Models with a
single sample (UNET) get std=0 (nothing to average).

Same two-step aggregation for the pixel metrics kept from BAR_METRICS: average
over frames within a sample first, then mean/std across samples.

Summary CSVs (per results-dir group == "model")
------------------------------------------------
metrics_per_frame_v3.csv     -- one row per (model, seq_idx, frame_idx, sample_idx)
bar_metrics_overall_v3.csv   -- BAR_METRICS mean+-std across samples, per model
bar_metrics_by_frame_v3.csv  -- same, grouped by (model, frame_idx)
scalar_nrmse_overall_v3.csv  -- NRMSE(mean/std/cpcp) mean+-std across samples, per model
scalar_nrmse_by_frame_v3.csv -- same, grouped by (model, frame_idx)

Usage
-----
python compute_metrics_v3.py \\
    --results-dirs \\
        /capstor/.../results_diffusion_classic_singlepass_step200k_dynhigh_acthigh \\
        /capstor/.../results_diffusion_nocond_singlepass_step200k_dynhigh_acthigh \\
        /capstor/.../results_unet_singlepass_step200k_dynhigh_acthigh \\
    --labels classic nocond unet \\
    --output-dir ./res_iono/metrics_v3_singlepass_dynhigh_acthigh
"""

import argparse
import os

import lpips as lpips_lib
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_metrics_v2 import unnormalize, compute_metrics_batch

BAR_METRICS  = ['ps_log_mse', 'lpips', 'ssim']   # presi cosi' come sono dall'output di compute_metrics_batch
SCALAR_STATS = ['mean', 'std', 'cpcp']


def compute_cpcp(field):
    return float(np.percentile(field, 75) - np.percentile(field, 25))


def scalar_stats(field):
    return {'mean': float(field.mean()), 'std': float(field.std()), 'cpcp': compute_cpcp(field)}


def process_results_dir(results_dir, label, lpips_fn, device):
    gen_dir = os.path.join(results_dir, 'generated_imgs')
    gt_dir  = os.path.join(results_dir, 'ground_truth')

    indices = sorted(
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(gen_dir)
        if f.startswith('pred_') and f.endswith('.npy')
    )

    records = []
    for k in tqdm(indices, desc=label):
        pred_path   = os.path.join(gen_dir, f'pred_{k}.npy')
        target_path = os.path.join(gt_dir,  f'target_{k}.npy')
        if not os.path.exists(target_path):
            continue

        pred   = np.load(pred_path).astype(np.float32)    # (n_samples, T, H, W), normalizzato [-1,1]
        target = np.load(target_path).astype(np.float32)  # (T, H, W)
        n_samples, T, H, W = pred.shape

        for t in range(T):
            batch_metrics = compute_metrics_batch(pred[:, t], target[t], lpips_fn, device)  # una entry per sample

            gt_stats = scalar_stats(unnormalize(target[t]))

            for s, m in enumerate(batch_metrics):
                pred_stats = scalar_stats(unnormalize(pred[s, t]))
                records.append({
                    'model': label, 'seq_idx': k, 'frame_idx': t, 'sample_idx': s,
                    **{f'{stat}_pred': v for stat, v in pred_stats.items()},
                    **{f'{stat}_gt':   v for stat, v in gt_stats.items()},
                    **{c: m[c] for c in BAR_METRICS if c in m},
                })

    return records


def compute_bar_metrics_summary(df, group_cols):
    """Media sui frame dentro ogni sample, poi mean+-std sui sample."""
    per_sample = df.groupby(group_cols + ['sample_idx'])[BAR_METRICS].mean()
    mean_ = per_sample.groupby(group_cols).mean()
    std_  = per_sample.groupby(group_cols).std().fillna(0.0)
    mean_.columns = [f'{c}_mean' for c in mean_.columns]
    std_.columns  = [f'{c}_std' for c in std_.columns]
    return pd.concat([mean_, std_], axis=1).reset_index()


def compute_scalar_nrmse_summary(df, group_cols):
    """NRMSE = RMSE(GT, pred) / std(GT), computed on the ENSEMBLE MEAN of the samples: for each
    (seq_idx, frame_idx) point the prediction used is the average across the n_samples trajectories,
    *then* RMSE/NRMSE is computed on that averaged series (not per-sample followed by averaging the
    NRMSE values). This is a deliberate choice for these scalar summary statistics -- averaging a
    handful of numbers per point is not the same kind of "cheat" as averaging whole images before
    computing a perceptual/structural metric (kept per-sample in compute_bar_metrics_summary above).

    Two different "std" columns are reported, because they answer different questions:

    - nrmse_{stat}_std ("smoothed"): for each sample individually, compute ITS OWN NRMSE pooled over
      every (seq_idx, frame_idx) point in the group (i.e. the old per-sample NRMSE), then take the std
      across the n_samples per-sample NRMSE values. Each per-sample number is already an average over
      hundreds/thousands of points, so this is smooth and answers "how much would the reported mean
      bounce around with a different set of samples?" -- the statistically appropriate error bar for a
      bar chart comparing aggregate model performance.
    - nrmse_{stat}_std_raw: the RMS (root-mean-square, the correct way to pool several independent
      standard deviations, since variances -- not standard deviations -- add) of the PER-POINT
      sample-to-sample std, i.e. how much the n_samples disagree with each other on any ONE given
      instance, unsmoothed. Useful for a separate per-instance calibration/uncertainty figure, not as
      the error bar on the main comparison bar charts (it can be much larger and is a different
      quantity).

    Models with a single sample (UNET/PERSISTENCE) get 0 for both, automatically."""
    rows = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for stat in SCALAR_STATS:
            per_point = grp.groupby(['seq_idx', 'frame_idx'])[f'{stat}_pred']
            ens_mean  = per_point.mean()
            ens_std   = per_point.std().fillna(0.0)   # NaN quando c'e' un solo sample (UNET/PERSISTENCE)
            gt        = grp.groupby(['seq_idx', 'frame_idx'])[f'{stat}_gt'].mean()

            g = gt.values.astype(np.float64)
            p = ens_mean.values.astype(np.float64)
            gt_std = g.std() + 1e-8

            rmse = float(np.sqrt(np.mean((g - p) ** 2)))
            row[f'nrmse_{stat}_mean'] = rmse / gt_std
            row[f'nrmse_{stat}_std_raw'] = float(np.sqrt(np.mean(ens_std.values ** 2))) / gt_std

            # "smussato": NRMSE per-sample (ognuno gia' pooled su tutti i punti), poi std tra i sample
            per_sample_nrmse = []
            for _, sample_grp in grp.groupby('sample_idx'):
                sp = sample_grp.groupby(['seq_idx', 'frame_idx'])[f'{stat}_pred'].mean()
                sp = sp.reindex(gt.index)
                rmse_s = float(np.sqrt(np.mean((g - sp.values.astype(np.float64)) ** 2)))
                per_sample_nrmse.append(rmse_s / gt_std)
            row[f'nrmse_{stat}_std'] = float(np.std(per_sample_nrmse)) if len(per_sample_nrmse) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--results-dirs', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--lpips-net', type=str, default='alex', choices=['alex', 'vgg'])
    args = p.parse_args()

    if args.labels is None:
        args.labels = [os.path.basename(d.rstrip('/')) for d in args.results_dirs]
    if len(args.labels) != len(args.results_dirs):
        p.error('--labels must have the same length as --results-dirs')

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading LPIPS ({args.lpips_net}) on {device}...')
    lpips_fn = lpips_lib.LPIPS(net=args.lpips_net, verbose=False).to(device)
    lpips_fn.eval()

    all_records = []
    for d, label in zip(args.results_dirs, args.labels):
        records = process_results_dir(d, label, lpips_fn, device)
        all_records.extend(records)
        print(f'{label}: {len(records)} record')

    df = pd.DataFrame(all_records)

    full_csv = os.path.join(args.output_dir, 'metrics_per_frame_v3.csv')
    df.to_csv(full_csv, index=False)
    print(f'\nSaved: {full_csv}')

    bar_overall  = compute_bar_metrics_summary(df, ['model'])
    bar_by_frame = compute_bar_metrics_summary(df, ['model', 'frame_idx'])
    bar_overall.to_csv(os.path.join(args.output_dir, 'bar_metrics_overall_v3.csv'), index=False)
    bar_by_frame.to_csv(os.path.join(args.output_dir, 'bar_metrics_by_frame_v3.csv'), index=False)

    nrmse_overall  = compute_scalar_nrmse_summary(df, ['model'])
    nrmse_by_frame = compute_scalar_nrmse_summary(df, ['model', 'frame_idx'])
    nrmse_overall.to_csv(os.path.join(args.output_dir, 'scalar_nrmse_overall_v3.csv'), index=False)
    nrmse_by_frame.to_csv(os.path.join(args.output_dir, 'scalar_nrmse_by_frame_v3.csv'), index=False)

    print('\n=== BAR_METRICS overall (mean +/- std sui sample) ===')
    print(bar_overall.to_string(index=False))

    print('\n=== NRMSE overall (mean +/- std sui sample) ===')
    print(nrmse_overall.to_string(index=False))


if __name__ == '__main__':
    main()
