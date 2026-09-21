# -*- coding: utf-8 -*-
"""Inference / evaluation script.

Supports:
  --model-type diffusion   : stochastic ViT diffusion model (uses generate_samples)
  --model-type unet        : deterministic UNet baseline (direct MSE prediction)

  --autoregressive         : slide the 15-frame conditioning window by 7 each step,
                             feeding back model output until the full sequence is covered.
                             Requires sequence_length > 22 (e.g. 15 + 7*N).
                             L1 conditions for future frames come from the CSV (real data).

Usage examples
--------------
# Single-pass diffusion, 10 samples
python generate_data_v4.py \
    --model-type diffusion \
    --config configs/forecast_iono_15_big_cosine_solar_classic.json \
    --ckpt /capstor/.../model_step_XXXXXXX.pth \
    --output-dir /capstor/.../results_diffusion \
    --cartesian-transform --activity-filter high

# Autoregressive UNet rollout over 50-frame sequences (5 steps of 7)
python generate_data_v4.py \
    --model-type unet \
    --ckpt /capstor/.../unet_step_XXXXXXX.pth \
    --output-dir /capstor/.../results_unet_ar \
    --sequence-length 50 \
    --autoregressive \
    --cartesian-transform --activity-filter high
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

torch.backends.cudnn.enabled = False   # required on H100 for Conv2d under bf16

import src as K
from src.data.dataset import get_sequence_data_objects_iterable
from src.unet_simple import UNetSimple
from util import generate_samples


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_diffusion_model(config_path, ckpt_path, device):
    config = K.config.load_config(config_path)
    inner_model = K.config.make_model(config)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_ema.inner_model.load_state_dict(ckpt['model_ema'])
    model_ema.to(device).eval()
    return model_ema


def load_unet_model(ckpt_path, device, cond_frames, pred_frames,
                    base_channels, channel_mults, num_res_blocks):
    model = UNetSimple(
        in_channels=1, out_channels=1,
        cond_frames=cond_frames, pred_frames=pred_frames,
        base_channels=base_channels,
        channel_mults=tuple(channel_mults),
        num_res_blocks=num_res_blocks,
        total_frames=cond_frames + pred_frames,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_ema'])
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Single-step predictors
# ---------------------------------------------------------------------------

def predict_diffusion(model, cond_img, l1_window, n_samples, sampler,
                      spatial_shape, no_mapping_cond, steps=50):
    """
    cond_img  : (1, cL, H, W)
    l1_window : (1, cL+pS, 4)
    returns     (n_samples, pS, H, W)  cpu
    """
    cond_rep  = cond_img.repeat(n_samples, 1, 1, 1)          # (n, cL, H, W)
    label_rep = None if no_mapping_cond else l1_window.repeat(n_samples, 1, 1)  # (n, 22, 4)
    return generate_samples(
        model, n_samples, cond_img.device,
        cond_label=label_rep,
        sampler=sampler,
        step=steps,
        cond_img=cond_rep,
        num_pred_frames=l1_window.shape[1] - cond_img.shape[1],   # pS = total_frames - cL = 7
    ).cpu()   # (n_samples, pS, H, W)


def predict_unet(model, cond_img, l1_window):
    """
    cond_img  : (1, cL, H, W)
    l1_window : (1, cL+pS, 4)
    returns     (1, pS, H, W)  cpu
    """
    return model(cond_img, l1_window).cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- model type ---
    p.add_argument('--model-type', choices=['diffusion', 'unet'], required=True)
    p.add_argument('--ckpt', type=str, required=True, help='path to checkpoint (.pth)')

    # --- diffusion-only ---
    p.add_argument('--config', type=str, default=None,
                   help='JSON config (required for --model-type diffusion)')
    p.add_argument('--no-mapping-cond', action='store_true',
                   help='disable L1 mapping conditioning (diffusion only)')
    p.add_argument('--n-samples', type=int, default=10,
                   help='stochastic samples per step (diffusion, single-pass only)')
    p.add_argument('--sampler', type=str, default='dpmpp_2m_sde',
                   help='diffusion sampler name')
    p.add_argument('--diffusion-steps', type=int, default=50,
                   help='number of diffusion denoising steps')

    # --- UNet architecture (must match training) ---
    p.add_argument('--base-channels',  type=int,         default=256)
    p.add_argument('--channel-mults',  type=int, nargs='+', default=[1, 2, 4, 4])
    p.add_argument('--num-res-blocks', type=int,         default=8)

    # --- sequence / data ---
    p.add_argument('--csv-path', type=str,
                   default='/users/framunno/data/ionosphere/l1_to_map_matched_2020_2025.csv')
    p.add_argument('--norm-csv', type=str, default=None,
                   metavar='PATH',
                   help='CSV used to compute cond_min/cond_max for L1 normalisation. '
                        'Defaults to --csv-path. Pass the training CSV when evaluating '
                        'on out-of-distribution data (e.g. 2015) so the model sees the '
                        'same normalisation scale it was trained with.')
    p.add_argument('--conditioning-length', type=int, default=15)
    p.add_argument('--predict-steps',       type=int, default=7)
    p.add_argument('--sequence-length',     type=int, default=22,
                   help='frames loaded per sample; must be >= cond+pred (22). '
                        'For AR rollout use 15 + 7*N, e.g. 50 for 5 rollout steps.')
    p.add_argument('--normalization-type',  type=str, default='absolute_max')
    p.add_argument('--cartesian-transform', action='store_true')
    p.add_argument('--activity-filter',     type=str, default=None,
                   choices=['high', 'low', 'moderate'],
                   help='filter sequences by geomagnetic activity level')
    p.add_argument('--dynamics-filter', type=str, default=None,
                   choices=['high', 'low'],
                   help='"high": keep top 25%% most dynamic; "low": keep bottom 25%% least dynamic')
    p.add_argument('--dynamics-high-quantile', type=float, default=0.75,
                   help='quantile threshold for --dynamics-filter high (default: 0.75)')
    p.add_argument('--dynamics-low-quantile',  type=float, default=0.25,
                   help='quantile threshold for --dynamics-filter low (default: 0.25)')
    p.add_argument('--dynamics-cache', type=str, default=None,
                   metavar='PATH',
                   help='path to pre-computed dynamics scores JSON (computed and saved '
                        'on first run, loaded on subsequent runs)')
    p.add_argument('--time-ranges', type=str, nargs='+', default=None,
                   metavar='START:END',
                   help='keep only sequences whose center timestamp falls in one of '
                        'these ranges, e.g. 2024-10-01:2024-10-31 2025-10-16:2025-12-31')
    p.add_argument('--split',              type=str, default='valid',
                   choices=['train', 'valid', 'test', 'eval'])
    p.add_argument('--min-center-distance', type=int, default=30)
    p.add_argument('--num-workers',         type=int, default=2)

    # --- rollout ---
    p.add_argument('--autoregressive', action='store_true',
                   help='autoregressive rollout: slide conditioning window by '
                        'predict-steps until sequence end. Diffusion uses 1 '
                        'committed sample per step (mean); UNet is deterministic.')
    p.add_argument('--rollout-end-time', type=str, default=None,
                   metavar='TIMESTAMP',
                   help='extend AR rollout beyond the loaded sequence until this UTC timestamp '
                        '(e.g. "2015-03-19T00:00:00+00:00"). Requires --autoregressive.')

    # --- output ---
    p.add_argument('--output-dir',  type=str, required=True)
    p.add_argument('--max-batches', type=int, default=None,
                   help='stop after this many sequences (useful for quick tests)')

    args = p.parse_args()

    cL = args.conditioning_length
    pS = args.predict_steps
    total_frames = cL + pS   # 22

    if args.sequence_length < total_frames:
        p.error(f'--sequence-length must be >= {total_frames}')

    if args.model_type == 'diffusion' and args.config is None:
        p.error('--config is required when --model-type is diffusion')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    if args.model_type == 'diffusion':
        model = load_diffusion_model(args.config, args.ckpt, device)
    else:
        model = load_unet_model(
            args.ckpt, device,
            cond_frames=cL, pred_frames=pS,
            base_channels=args.base_channels,
            channel_mults=args.channel_mults,
            num_res_blocks=args.num_res_blocks,
        )

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    val_dataset, _, val_dl = get_sequence_data_objects_iterable(
        csv_path=args.csv_path,
        batch_size=1,
        num_data_workers=args.num_workers,
        split=args.split,
        seed=42,
        sequence_length=args.sequence_length,
        normalization_type=args.normalization_type,
        use_l1_conditions=True,
        min_center_distance=args.min_center_distance,
        cartesian_transform=args.cartesian_transform,
        output_size=128,
        only_complete_sequences=True,
        activity_filter=args.activity_filter,
        epsilon_high_quantile=0.75,
        dynamics_filter_quantile=(
            args.dynamics_high_quantile if args.dynamics_filter == 'high' else
            args.dynamics_low_quantile  if args.dynamics_filter == 'low'  else None
        ),
        dynamics_filter_mode=args.dynamics_filter if args.dynamics_filter else 'high',
        dynamics_cache_path=args.dynamics_cache,
    )

    # Override L1 normalisation stats with the training CSV when evaluating OOD data
    if args.norm_csv is not None:
        import pandas as _pd
        _l1_cols = ['bx_gsm', 'by_gsm', 'bz_gsm', 'proton_vx_gsm']
        _df_norm = _pd.read_csv(args.norm_csv)
        _df_clean = _df_norm[_l1_cols].replace(-99999, float('nan'))
        val_dataset.cond_min = _df_clean.min().values.astype('float32')
        val_dataset.cond_max = _df_clean.max().values.astype('float32')
        print(f'[norm-csv] cond_min overridden from {args.norm_csv}')
        print(f'  cond_min = {val_dataset.cond_min}')
        print(f'  cond_max = {val_dataset.cond_max}')

    spatial_shape = (128, 128) if args.cartesian_transform else (24, 360)

    # -----------------------------------------------------------------------
    # Optional time-range filter (applied before iteration begins)
    # -----------------------------------------------------------------------
    if args.time_ranges:
        import pandas as pd
        def _to_naive(ts_str):
            t = pd.Timestamp(ts_str)
            return t.tz_localize(None) if t.tzinfo is None else t.tz_convert(None)

        parsed_ranges = []
        for r in args.time_ranges:
            # timestamps like "2015-03-16T00:20" contain ":" so can't split naively;
            # ISO datetime is always 16 chars "YYYY-MM-DDTHH:MM", separator at pos 16
            if 'T' in r:
                start_str, end_str = r[:16], r[17:]
            else:
                start_str, end_str = r.split(':')
            parsed_ranges.append((_to_naive(start_str), _to_naive(end_str)))
        before = len(val_dataset.sequences)
        val_dataset.sequences = [
            idx for idx in val_dataset.sequences
            if any(s <= _to_naive(val_dataset.all_timestamps[idx]) <= e
                   for s, e in parsed_ranges)
        ]
        print(f'Time-range filter: {before} → {len(val_dataset.sequences)} sequences')

    # -----------------------------------------------------------------------
    # Build timestamp index from dataset (no changes to dataset needed)
    # sequences is a list of center_idx values in the order they'll be yielded
    # -----------------------------------------------------------------------
    half = args.sequence_length // 2
    timestamp_index = []   # one entry per sequence, in iteration order
    for center_idx in val_dataset.sequences:
        start_idx = center_idx - half
        center_ts = val_dataset.all_timestamps[center_idx]
        start_ts  = val_dataset.all_timestamps[max(0, start_idx)]
        end_idx   = min(start_idx + args.sequence_length - 1, len(val_dataset.all_timestamps) - 1)
        end_ts    = val_dataset.all_timestamps[end_idx]
        timestamp_index.append({
            'center_idx':  center_idx,
            'start_idx':   center_idx - half,
            'center_time': str(center_ts),
            'start_time':  str(start_ts),
            'end_time':    str(end_ts),
        })

    # -----------------------------------------------------------------------
    # Output dirs
    # -----------------------------------------------------------------------
    for d in ('input_imgs', 'generated_imgs', 'ground_truth', 'conditions', 'metadata'):
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    # -----------------------------------------------------------------------
    # Inference loop
    # -----------------------------------------------------------------------
    with torch.no_grad():
        for k, batch in enumerate(tqdm(val_dl, desc='Inference')):
            if args.max_batches is not None and k >= args.max_batches:
                break
            if os.path.exists(os.path.join(args.output_dir, f'generated_imgs/pred_{k}.npy')):
                continue

            # batch[0]: (1, T, 1, H, W) → squeeze channel → (1, T, H, W)
            inpt  = batch[0].contiguous().float().to(device).squeeze(2)  # (1, T, H, W)
            l1_all = batch[1].float().to(device)                          # (1, T, 4)
            T = inpt.shape[1]

            if args.autoregressive:
                # ----------------------------------------------------------
                # Autoregressive rollout
                # ----------------------------------------------------------
                n_steps = (T - cL) // pS
                if n_steps == 0:
                    print(f'Warning: sequence_length={T} too short for AR rollout, skipping.')
                    continue

                if args.model_type == 'diffusion':
                    # n_samples independent trajectories run in parallel.
                    # current_cond: (n_samples, cL, H, W) — each sample has its own window
                    n = args.n_samples
                    current_cond = inpt[:, :cL].repeat(n, 1, 1, 1)  # (n, cL, H, W)
                else:
                    # UNet is deterministic — one trajectory is enough
                    n = 1
                    current_cond = inpt[:, :cL]   # (1, cL, H, W)

                all_preds = []   # list of (n, pS, H, W) cpu tensors

                for step in range(n_steps):
                    l1_window = l1_all[:, step * pS : step * pS + total_frames, :]  # (1, 22, 4)

                    if args.model_type == 'diffusion':
                        # Each of the n samples gets its own independent noise draw.
                        # current_cond is already (n, cL, H, W) — pass directly.
                        label_rep = None if args.no_mapping_cond else l1_window.repeat(n, 1, 1)
                        pred = generate_samples(
                            model, n, device,
                            cond_label=label_rep,
                            sampler=args.sampler,
                            step=args.diffusion_steps,
                            cond_img=current_cond,
                            num_pred_frames=pS,
                        ).cpu()   # (n, pS, H, W)
                        pred_committed = pred.to(device)
                    else:
                        pred_committed = predict_unet(model, current_cond, l1_window).to(device)
                        pred = pred_committed.cpu()

                    all_preds.append(pred.cpu())
                    # Each sample slides its own window independently
                    current_cond = torch.cat([current_cond[:, pS:], pred_committed], dim=1)

                # ---- Extended chained rollout beyond loaded sequence ----------
                if args.rollout_end_time and k < len(timestamp_index):
                    import pandas as pd
                    rollout_end = pd.Timestamp(args.rollout_end_time)
                    if rollout_end.tzinfo is None:
                        rollout_end = rollout_end.tz_localize('UTC')
                    seq_start_idx = timestamp_index[k]['start_idx']
                    ext_step = 0
                    while True:
                        g_start = seq_start_idx + (n_steps + ext_step) * pS
                        g_end   = g_start + total_frames
                        if g_end > len(val_dataset.all_files):
                            break
                        ts_raw = val_dataset.all_timestamps[g_start + cL]
                        ts = pd.Timestamp(ts_raw)
                        if ts.tzinfo is None:
                            ts = ts.tz_localize('UTC')
                        if ts >= rollout_end:
                            break
                        l1_rows = []
                        for fi in range(g_start, g_end):
                            fname = os.path.basename(val_dataset.all_files[fi])
                            cond  = val_dataset.filename_to_conditions.get(fname)
                            if cond is not None:
                                cond = 2 * (cond - val_dataset.cond_min) / (val_dataset.cond_max - val_dataset.cond_min) - 1
                            else:
                                cond = np.zeros(4, dtype=np.float32)
                            l1_rows.append(cond)
                        l1_window = torch.tensor([l1_rows], dtype=torch.float32).to(device)
                        if args.model_type == 'diffusion':
                            label_rep = None if args.no_mapping_cond else l1_window.repeat(n, 1, 1)
                            pred = generate_samples(
                                model, n, device,
                                cond_label=label_rep,
                                sampler=args.sampler,
                                step=args.diffusion_steps,
                                cond_img=current_cond,
                                num_pred_frames=pS,
                            ).cpu()
                            pred_committed = pred.to(device)
                        else:
                            pred_committed = predict_unet(model, current_cond, l1_window).to(device)
                            pred = pred_committed.cpu()
                        all_preds.append(pred.cpu())
                        current_cond = torch.cat([current_cond[:, pS:], pred_committed], dim=1)
                        ext_step += 1
                    print(f'  Extended rollout: {ext_step} extra steps '
                          f'(total {n_steps + ext_step} steps, '
                          f'{(n_steps + ext_step) * pS * 2} min predicted)')

                generated = torch.cat(all_preds, dim=1)   # (n, total_steps*pS, H, W)
                target    = inpt[:, cL : cL + n_steps * pS]

            else:
                # ----------------------------------------------------------
                # Single-pass prediction
                # ----------------------------------------------------------
                cond_img  = inpt[:, :cL]
                target    = inpt[:, cL : cL + pS]
                l1_window = l1_all[:, :total_frames, :]

                if args.model_type == 'diffusion':
                    generated = predict_diffusion(
                        model, cond_img, l1_window,
                        n_samples=args.n_samples, sampler=args.sampler,
                        spatial_shape=spatial_shape,
                        no_mapping_cond=args.no_mapping_cond,
                        steps=args.diffusion_steps,
                    )   # (n_samples, pS, H, W)
                else:
                    generated = predict_unet(model, cond_img, l1_window)  # (1, pS, H, W)

            # ---------------------------------------------------------------
            # Save
            # ---------------------------------------------------------------
            np.save(os.path.join(args.output_dir, f'input_imgs/input_{k}.npy'),
                    inpt[0, :cL].cpu().numpy())
            np.save(os.path.join(args.output_dir, f'generated_imgs/pred_{k}.npy'),
                    generated.numpy())
            np.save(os.path.join(args.output_dir, f'ground_truth/target_{k}.npy'),
                    target[0].cpu().numpy())
            np.save(os.path.join(args.output_dir, f'conditions/cond_{k}.npy'),
                    l1_all[0].cpu().numpy())
            if k < len(timestamp_index):
                import json
                meta = {**timestamp_index[k], 'sample_idx': k,
                        'model_type': args.model_type,
                        'autoregressive': args.autoregressive,
                        'n_rollout_steps': (T - cL) // pS if args.autoregressive else 1}
                with open(os.path.join(args.output_dir, f'metadata/meta_{k}.json'), 'w') as f:
                    json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
