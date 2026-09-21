# -*- coding: utf-8 -*-
import torch
torch.backends.cudnn.enabled = False  # required on H100 for Conv2d under bf16 (LPIPS)

"""Compute metrics from pre-saved .npy inference results.

Reads directly from results directories — no dataset loading needed.
Supports diffusion (n_samples, T, H, W) and UNet (1, T, H, W) outputs.

Metrics per frame
-----------------
Image quality   : PSNR, SSIM, LPIPS
Pixel           : MSE, MAE, RMSE, correlation
Physics         : rel_l2, nrmse, mean_error, std_ratio  (on unnormalized V)
CPCP            : cpcp_pred, cpcp_gt, cpcp_pct_var      (99th-1st percentile)
Dynamic range   : PCPC (% variation of max-min)
Morphology      : dist_max_px, dist_min_px
Power spectrum  : ps_log_mse, ps_log_rel_l2, ps_rel_error_filtered

Summary CSVs
------------
metrics_per_frame.csv  — one row per (model, seq_idx, frame_idx, sample)
metrics_by_frame.csv   — mean ± std grouped by (model, frame_idx) ← "per frame"
metrics_overall.csv    — mean ± std grouped by model

Usage
-----
python compute_metrics_v2.py \\
    --results-dirs \\
        /capstor/.../results_diffusion_classic_AR50_step200k \\
        /capstor/.../results_diffusion_nocond_AR50_step200k \\
        /capstor/.../results_unet_AR50_step200k \\
    --labels classic nocond unet \\
    --output-dir /capstor/.../metrics_comparison
"""

import argparse
import json
import os

import lpips as lpips_lib
import numpy as np
import pandas as pd
from scipy.signal.windows import tukey
from powerbox import get_power
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

PHYS_MIN = -80000.0
PHYS_MAX =  80000.0


def unnormalize(x):
    return ((x + 1) / 2) * (PHYS_MAX - PHYS_MIN) + PHYS_MIN


def compute_efield_metrics(pred_phys, tgt_phys):
    gy_gt, gx_gt = np.gradient(tgt_phys)
    gy_pr, gx_pr = np.gradient(pred_phys)
    mag_gt = np.sqrt(gx_gt**2 + gy_gt**2)
    mag_pr = np.sqrt(gx_pr**2 + gy_pr**2)
    rms_gt = float(np.sqrt(np.mean(mag_gt**2)))
    rms_pr = float(np.sqrt(np.mean(mag_pr**2)))
    return {
        'efield_rms_pred':  rms_pr,
        'efield_rms_gt':    rms_gt,
        'efield_rms_ratio': float(rms_pr / (rms_gt + 1e-8)),
        'efield_mag_mae':   float(np.mean(np.abs(mag_pr - mag_gt))),
        'efield_mag_corr':  float(np.corrcoef(mag_pr.flatten(), mag_gt.flatten())[0, 1]),
    }


def compute_isotropic_power(x, boxlength=None, apply_window=True):
    *batch_shape, H, W = x.shape
    x = x.reshape(-1, H, W)
    window2d = 1.0
    if apply_window:
        window2d = np.outer(tukey(H, alpha=0.5), tukey(W, alpha=0.5))
    x = x * window2d
    spectra = []
    for i in range(x.shape[0]):
        pk, k = get_power(x[i], boxlength=boxlength or [H, W], bins_upto_boxlen=False)
        spectra.append(pk)
    return np.array(spectra).reshape(batch_shape + [-1]), k


def compute_power_spectrum_metrics(pred_ps, gt_ps, epsilon=1e-10):
    log_mse = float(np.mean((np.log10(pred_ps + epsilon) - np.log10(gt_ps + epsilon)) ** 2))
    log_rel_l2 = float(
        np.sqrt(np.sum((np.log10(pred_ps + epsilon) - np.log10(gt_ps + epsilon)) ** 2)) /
        np.sqrt(np.sum((np.log10(gt_ps + epsilon)) ** 2))
    )
    threshold = np.max(gt_ps) * 1e-3
    mask = gt_ps > threshold
    rel_error_filtered = float(np.mean(np.abs(pred_ps[mask] - gt_ps[mask]) / gt_ps[mask])) if mask.sum() > 0 else float('nan')
    ps_nmse = float(np.mean((pred_ps - gt_ps) ** 2) / (np.var(gt_ps) + epsilon))
    return {'ps_log_mse': log_mse, 'ps_log_rel_l2': log_rel_l2,
            'ps_rel_error_filtered': rel_error_filtered, 'ps_nmse': ps_nmse}


def compute_cpcp(field_renorm):
    return float(np.percentile(field_renorm, 75) - np.percentile(field_renorm, 25))


def compute_metrics_batch(preds, target, lpips_fn, device):
    """
    preds  : (n_samples, H, W) float32, normalized [-1, 1]
    target : (H, W) float32
    Returns list of n_samples metric dicts.
    """
    n_samples, H, W = preds.shape
    tgt_phys = unnormalize(target)
    cpcp_gt  = compute_cpcp(tgt_phys)
    dr_gt    = float(tgt_phys.max() - tgt_phys.min())
    data_range = float(target.max() - target.min()) if target.max() != target.min() else 1.0

    tgt_max_pos = np.unravel_index(np.argmax(target), target.shape)
    tgt_min_pos = np.unravel_index(np.argmin(target), target.shape)

    # gt power spectrum (once per frame)
    gt_ps = compute_isotropic_power(target[None], boxlength=[H, W])[0][0]

    # batch power spectra for all samples at once
    all_pred_ps = compute_isotropic_power(preds, boxlength=[H, W])[0]  # (n_samples, F)

    # batch LPIPS: (n_samples, 1, H, W)
    t_preds = torch.tensor(preds[:, None]).float().to(device)
    t_tgt   = torch.tensor(target[None, None]).float().to(device).expand(n_samples, -1, -1, -1)
    with torch.no_grad():
        lpips_vals = lpips_fn(t_preds, t_tgt).squeeze().cpu().numpy()
    if n_samples == 1:
        lpips_vals = np.array([float(lpips_vals)])

    results = []
    for s in range(n_samples):
        pred = preds[s]
        pred_f = pred.flatten().astype(np.float64)
        tgt_f  = target.flatten().astype(np.float64)

        mse  = float(np.mean((pred_f - tgt_f) ** 2))
        mae  = float(np.mean(np.abs(pred_f - tgt_f)))
        rmse = float(np.sqrt(mse))
        corr = float(np.corrcoef(pred_f, tgt_f)[0, 1])
        ssim_val = float(ssim_fn(target, pred, data_range=data_range))
        psnr_val = float(psnr_fn(target, pred, data_range=data_range))
        lpips_val = float(lpips_vals[s])

        pred_phys = unnormalize(pred)
        rel_l2     = float(np.linalg.norm(pred_phys - tgt_phys) / (np.linalg.norm(tgt_phys) + 1e-8))
        nrmse      = float(np.sqrt(np.mean((pred_phys - tgt_phys) ** 2)) / (np.std(tgt_phys) + 1e-8))
        mean_error = float((np.mean(pred_phys) - np.mean(tgt_phys)) / (np.std(tgt_phys) + 1e-8))
        std_ratio  = float(np.std(pred_phys) / (np.std(tgt_phys) + 1e-8))

        cpcp_pred    = compute_cpcp(pred_phys)
        cpcp_pct_var = float(abs(cpcp_gt - cpcp_pred) / abs(cpcp_gt) * 100) if cpcp_gt != 0 else 0.0

        dr_pred = float(pred_phys.max() - pred_phys.min())
        pcpc    = float(abs(dr_pred - dr_gt) / (dr_gt + 1e-8) * 100)

        pred_max_pos = np.unravel_index(np.argmax(pred), pred.shape)
        pred_min_pos = np.unravel_index(np.argmin(pred), pred.shape)
        dist_max = float(np.sqrt((tgt_max_pos[0]-pred_max_pos[0])**2 + (tgt_max_pos[1]-pred_max_pos[1])**2))
        dist_min = float(np.sqrt((tgt_min_pos[0]-pred_min_pos[0])**2 + (tgt_min_pos[1]-pred_min_pos[1])**2))

        ps_metrics = compute_power_spectrum_metrics(all_pred_ps[s], gt_ps)
        ef_metrics = compute_efield_metrics(pred_phys, tgt_phys)

        results.append({
            'mse': mse, 'mae': mae, 'rmse': rmse, 'correlation': corr,
            'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val,
            'rel_l2': rel_l2, 'nrmse': nrmse, 'mean_error': mean_error, 'std_ratio': std_ratio,
            'cpcp_pred': cpcp_pred, 'cpcp_gt': cpcp_gt, 'cpcp_pct_var': cpcp_pct_var,
            'pcpc': pcpc, 'dr_pred_V': dr_pred, 'dr_gt_V': dr_gt,
            'dist_max_px': dist_max, 'dist_min_px': dist_min,
            **ps_metrics,
            **ef_metrics,
        })
    return results


def process_results_dir(results_dir, label, lpips_fn, device):
    gen_dir  = os.path.join(results_dir, 'generated_imgs')
    gt_dir   = os.path.join(results_dir, 'ground_truth')
    meta_dir = os.path.join(results_dir, 'metadata')

    indices = sorted(
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(gen_dir)
        if f.startswith('pred_') and f.endswith('.npy')
    )

    records = []
    for k in tqdm(indices, desc=label):
        pred_path   = os.path.join(gen_dir,  f'pred_{k}.npy')
        target_path = os.path.join(gt_dir,   f'target_{k}.npy')
        meta_path   = os.path.join(meta_dir, f'meta_{k}.json')

        if not os.path.exists(target_path):
            continue

        pred   = np.load(pred_path).astype(np.float32)   # (n_samples, T, H, W)
        target = np.load(target_path).astype(np.float32) # (T, H, W)
        n_samples, T, H, W = pred.shape

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        gt_mean_t0   = float(unnormalize(target[0]).mean())
        pred_mean_t0 = [float(unnormalize(pred[s, 0]).mean()) for s in range(n_samples)]

        for t in range(T):
            sample_metrics = compute_metrics_batch(pred[:, t], target[t], lpips_fn, device)
            gt_mean_drift = float(unnormalize(target[t]).mean()) - gt_mean_t0
            for s, m in enumerate(sample_metrics):
                pred_mean_drift = float(unnormalize(pred[s, t]).mean()) - pred_mean_t0[s]
                records.append({
                    'model':       label,
                    'seq_idx':     k,
                    'frame_idx':   t,
                    'sample_idx':  s,
                    'center_time': meta.get('center_time', ''),
                    'mean_drift_gt':   gt_mean_drift,
                    'mean_drift_pred': pred_mean_drift,
                    **m,
                })

    return records


METRIC_COLS = [
    'mse', 'mae', 'rmse', 'correlation',
    'psnr', 'ssim', 'lpips',
    'rel_l2', 'nrmse', 'mean_error', 'std_ratio',
    'cpcp_pred', 'cpcp_gt', 'cpcp_pct_var',
    'pcpc', 'dr_pred_V', 'dr_gt_V',
    'dist_max_px', 'dist_min_px',
    'ps_log_mse', 'ps_log_rel_l2', 'ps_rel_error_filtered', 'ps_nmse',
    'efield_rms_pred', 'efield_rms_gt', 'efield_rms_ratio', 'efield_mag_mae', 'efield_mag_corr',
    'mean_drift_gt', 'mean_drift_pred',
]


def summarize(df, group_cols):
    rows = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for col in METRIC_COLS:
            if col in grp.columns:
                row[f'{col}_mean'] = grp[col].mean()
                row[f'{col}_std']  = grp[col].std()
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
        print(f'{label}: {len(records)} records')

    df = pd.DataFrame(all_records)

    # Full per-frame per-sample CSV
    full_csv = os.path.join(args.output_dir, 'metrics_per_frame.csv')
    df.to_csv(full_csv, index=False)
    print(f'\nSaved: {full_csv}')

    # Summary by (model, frame_idx) — shows degradation over rollout steps
    by_frame = summarize(df, ['model', 'frame_idx'])
    by_frame_csv = os.path.join(args.output_dir, 'metrics_by_frame.csv')
    by_frame.to_csv(by_frame_csv, index=False)
    print(f'Saved: {by_frame_csv}')

    # Overall summary per model
    overall = summarize(df, ['model'])
    overall_csv = os.path.join(args.output_dir, 'metrics_overall.csv')
    overall.to_csv(overall_csv, index=False)
    print(f'Saved: {overall_csv}')

    print('\n=== OVERALL SUMMARY ===')
    key = ['psnr_mean', 'ssim_mean', 'lpips_mean', 'correlation_mean',
           'pcpc_mean', 'cpcp_pct_var_mean', 'dist_max_px_mean', 'dist_min_px_mean',
           'ps_log_mse_mean']
    print(overall[['model'] + [c for c in key if c in overall.columns]].to_string(index=False))

    print('\n=== BY FRAME INDEX ===')
    print(by_frame[['model', 'frame_idx', 'psnr_mean', 'ssim_mean', 'lpips_mean',
                    'correlation_mean', 'ps_log_mse_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
