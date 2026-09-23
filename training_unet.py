# -*- coding: utf-8 -*-
"""Deterministic UNet baseline training script.

Trains a simple conv UNet with MSE loss directly on ionosphere frame prediction.
No diffusion, no noise schedule — pure regression benchmark.
"""

import os
import torch
torch.backends.cudnn.enabled = False


def main():
    import argparse
    import json
    from pathlib import Path

    import torch
    from torch import optim
    from tqdm.auto import tqdm
    import accelerate

    from src.unet_simple import UNetSimple
    import src as K
    from src.data.dataset import get_sequence_data_objects_iterable

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size',          type=int,   default=1)
    p.add_argument('--num-workers',         type=int,   default=8)
    p.add_argument('--max-steps',           type=int,   default=2000000)
    p.add_argument('--evaluate-every',      type=int,   default=5000)
    p.add_argument('--save-every',          type=int,   default=10000)
    p.add_argument('--val-steps',           type=int,   default=200)
    p.add_argument('--lr',                  type=float, default=1e-4)
    p.add_argument('--weight-decay',        type=float, default=1e-3)
    p.add_argument('--mixed-precision',     type=str,   default='bf16')
    p.add_argument('--grad-accum-steps',    type=int,   default=1)
    p.add_argument('--sequence-length',     type=int,   default=22)
    p.add_argument('--predict-steps',       type=int,   default=7)
    p.add_argument('--csv-path',            type=str,   required=True)
    p.add_argument('--saving-path',         type=str,   default='./data_root/models_results')
    p.add_argument('--dir-name',            type=str,   default='unet_benchmark')
    p.add_argument('--normalization-type',  type=str,   default='absolute_max')
    p.add_argument('--base-channels',       type=int,   default=128)
    p.add_argument('--channel-mults',       type=int,   nargs='+', default=[1, 2, 4, 4])
    p.add_argument('--num-res-blocks',      type=int,   default=2)
    p.add_argument('--dropout',             type=float, default=0.0)
    p.add_argument('--cartesian-transform', action='store_true')
    p.add_argument('--only-complete-sequences', action='store_true')
    p.add_argument('--use-wandb',           action='store_true')
    p.add_argument('--wandb-runname',       type=str,   default=None)
    p.add_argument('--wandb-runid',         type=str,   default=None)
    p.add_argument('--name',                type=str,   default='model')
    p.add_argument('--resume',              type=str,   default=None)

    args = p.parse_args()
    args.conditioning_length = args.sequence_length - args.predict_steps

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}')
        print(f'Conditioning frames: {args.conditioning_length}, Predict frames: {args.predict_steps}')

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    spatial_shape = (128, 128) if args.cartesian_transform else (24, 360)
    model = UNetSimple(
        in_channels=1,
        out_channels=1,
        cond_frames=args.conditioning_length,
        pred_frames=args.predict_steps,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
    )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {n_params:,}')

    # -------------------------------------------------------------------------
    # Optimizer & scheduler
    # -------------------------------------------------------------------------
    opt = optim.AdamW(model.param_groups(args.lr), lr=args.lr,
                      betas=(0.95, 0.999), eps=1e-6, weight_decay=args.weight_decay)

    total_steps = args.max_steps
    warmup_steps = 1000
    warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=args.lr / 1000, T_max=total_steps - warmup_steps)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    ema_sched = K.utils.EMAWarmup(power=0.6667, max_value=0.9999)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    train_dataset, _, train_dl = get_sequence_data_objects_iterable(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_data_workers=args.num_workers,
        split='train',
        seed=42,
        sequence_length=args.sequence_length,
        normalization_type=args.normalization_type,
        use_l1_conditions=True,
        min_center_distance=15,
        cartesian_transform=args.cartesian_transform,
        output_size=128,
        only_complete_sequences=args.only_complete_sequences,
        persistent_workers=True,
        prefetch_factor=4,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    val_dataset, _, val_dl = get_sequence_data_objects_iterable(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_data_workers=args.num_workers,
        split='valid',
        seed=42,
        sequence_length=args.sequence_length,
        normalization_type=args.normalization_type,
        use_l1_conditions=True,
        min_center_distance=30,
        cartesian_transform=args.cartesian_transform,
        output_size=128,
        only_complete_sequences=args.only_complete_sequences,
        persistent_workers=True,
        prefetch_factor=4,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    # -------------------------------------------------------------------------
    # Prepare
    # -------------------------------------------------------------------------
    model, opt = accelerator.prepare(model, opt)
    from copy import deepcopy
    model_ema = deepcopy(accelerator.unwrap_model(model))

    dir_path_mdl = os.path.join(args.saving_path, f'models_{args.dir_name}')
    dir_path_res = os.path.join(args.saving_path, f'results_{args.dir_name}')
    os.makedirs(dir_path_mdl, exist_ok=True)
    os.makedirs(dir_path_res, exist_ok=True)

    # -------------------------------------------------------------------------
    # WandB
    # -------------------------------------------------------------------------
    if accelerator.is_main_process and args.use_wandb:
        import wandb
        wandb.init(project='ionosphere', id=args.wandb_runid, name=args.wandb_runname,
                   resume='allow', config=vars(args), dir=os.getcwd())

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    state_path = Path(f'{args.name}_state_{args.dir_name}.json')
    step = 0
    epoch = 0

    if state_path.exists() or args.resume:
        ckpt_path = args.resume if args.resume else json.load(open(state_path))['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(ckpt['model'])
        model_ema.load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        step  = ckpt['step'] + 1
        epoch = ckpt['epoch'] + 1
        del ckpt

    # -------------------------------------------------------------------------
    # Save helper
    # -------------------------------------------------------------------------
    def save():
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        filename = os.path.join(dir_path_mdl, f'{args.name}_step_{step:07}.pth')
        obj = {
            'model':     accelerator.get_state_dict(accelerator.unwrap_model(model)),
            'model_ema': model_ema.state_dict(),
            'opt':       opt.state_dict(),
            'sched':     sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'step':      step,
            'epoch':     epoch,
        }
        torch.save(obj, filename)
        json.dump({'latest_checkpoint': filename}, open(state_path, 'w'))
        tqdm.write(f'Saved {filename}')

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def infinite_loader(dl):
        while True:
            yield from dl

    model.train()
    train_iter = infinite_loader(train_dl)
    losses_buf = []
    pbar = tqdm(total=args.max_steps, initial=step, smoothing=0.1,
                disable=not accelerator.is_main_process)

    while step < args.max_steps:
        batch = next(train_iter)

        with accelerator.accumulate(model):
            inpt = batch[0].contiguous().float().to(device, non_blocking=True).squeeze(2)
            cond_img   = inpt[:, :args.conditioning_length]
            target_img = inpt[:, args.conditioning_length:args.conditioning_length + args.predict_steps]
            l1_cond    = batch[1].float().to(device, non_blocking=True)   # (B, total_frames, 4)

            pred = model(cond_img, l1_cond)           # (B, 7, H, W)
            loss = ((pred - target_img) ** 2).mean()

            losses_buf.append(loss.item())
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                K.utils.ema_update(accelerator.unwrap_model(model), model_ema, ema_sched.get_value())
                ema_sched.step()

        if accelerator.sync_gradients:
            step += 1
            pbar.update(1)

            if step % 25 == 0 and accelerator.is_main_process:
                avg = sum(losses_buf) / len(losses_buf)
                losses_buf.clear()
                tqdm.write(f'step: {step}, loss: {avg:.6f}, lr: {sched.get_last_lr()[0]:.2e}')
                if args.use_wandb:
                    import wandb
                    wandb.log({'train/loss': avg, 'lr': sched.get_last_lr()[0]}, step=step)

            # -----------------------------------------------------------------
            # Validation
            # -----------------------------------------------------------------
            if step % args.evaluate_every == 0:
                accelerator.wait_for_everyone()
                model.eval()
                local_val_loss = torch.tensor(0.0, device=device)
                val_iter_tmp = iter(val_dl)
                with torch.no_grad():
                    for _ in tqdm(range(args.val_steps), desc='Validation',
                                  disable=not accelerator.is_main_process, leave=False):
                        try:
                            vb = next(val_iter_tmp)
                        except StopIteration:
                            break
                        inpt_v = vb[0].contiguous().float().to(device, non_blocking=True).squeeze(2)
                        cond_v   = inpt_v[:, :args.conditioning_length]
                        target_v = inpt_v[:, args.conditioning_length:args.conditioning_length + args.predict_steps]
                        l1_v     = vb[1].float().to(device, non_blocking=True)
                        pred_v   = model(cond_v, l1_v)
                        local_val_loss += ((pred_v - target_v) ** 2).mean().detach()

                accelerator.wait_for_everyone()
                gathered = accelerator.gather(local_val_loss)
                if accelerator.is_main_process:
                    val_loss = gathered.mean().item()
                    tqdm.write(f'step: {step}, val_loss: {val_loss:.6f}')
                    if args.use_wandb:
                        import wandb
                        wandb.log({'val_loss': val_loss}, step=step)
                model.train()
                accelerator.wait_for_everyone()

            # -----------------------------------------------------------------
            # Save
            # -----------------------------------------------------------------
            if step % args.save_every == 0:
                save()

    save()
    if accelerator.is_main_process:
        print(f'Training complete at step {step}.')


if __name__ == '__main__':
    main()
