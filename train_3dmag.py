# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:47:31 2025

@author: pio-r
"""

# import debugpy

# debugpy.connect(("v000675", 5678))  # VS Code listens on login node
# print("✅ Connected to VS Code debugger!")
# debugpy.wait_for_client()
# print("🎯 Debugger attached!")

def main():

    import argparse
    import os
    from copy import deepcopy
    import json
    from pathlib import Path
    import time
                
    import matplotlib.pyplot as plt
    import imageio
    import io
    import accelerate
    import safetensors.torch as safetorch
    import torch
    import torch._dynamo
    from torch import optim
    from tqdm.auto import tqdm

    import src as K

    from util import generate_samples
    import torch
    from src.data.dataset import get_data_objects

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=32,
                help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                help='enable gradient checkpointing')
    p.add_argument('--compile', action='store_true',
                help='compile the model')
    p.add_argument('--config', type=str, required=True,
                help='the configuration file')
    p.add_argument('--data-path', type=str, default="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/pickled_maps",
                help='the path of the dataset')
    p.add_argument('--saving-path', type=str, default="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/", 
                help='the path where to save the model')
    p.add_argument('--dir-name', type=str, default='cond_generation_batch32',
                help='the directory name to use')  # <---- Added this line
    p.add_argument('--end-step', type=int, default=None,
                help='the step to end training at')
    p.add_argument('--evaluate-every', type=int, default=5,
                help='How often to evaluate the model in epochs') 
    p.add_argument('--evaluate-only', action='store_true',
                help='evaluate instead of training')
    p.add_argument('--gns', action='store_true',
                help='measure the gradient noise scale (DDP only, disables stratified sampling)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                help='the mixed precision type')
    p.add_argument('--name', type=str, default='model',
                help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                help='the number of data loader workers')
    p.add_argument('--reset-ema', action='store_true',
                help='reset the EMA')
    p.add_argument('--resume', type=str,
                help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                help='the inference checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                help='the number of images to sample for demo grids')
    p.add_argument('--save-every', type=int, default=10000,
                help='save every this many steps')
    p.add_argument('--seed', type=int,
                help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                choices=['fork', 'forkserver', 'spawn'],
                help='the multiprocessing start method')
    p.add_argument('--use_wandb', type=bool, default=True,
                help='Use wandb')

    p.add_argument('--wandb-entity', type=str,
                help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                help='save model to wandb')

    args = p.parse_args(["--config", "./configs/3dmag.json"])

    dir_path_res = f"results_{args.dir_name}"
    dir_path_mdl = f"model_{args.dir_name}"

    if not os.path.exists(dir_path_res):
        os.makedirs(dir_path_res, exist_ok=True)
        
    if not os.path.exists(dir_path_mdl):
        os.makedirs(dir_path_mdl, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    config = K.config.load_config(args.config)
    model_config = config['model']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']

    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps,
                                            mixed_precision=args.mixed_precision, 
                kwargs_handlers=[accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=True)])

    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
    elapsed = 0.0

    # Model definition,
    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)

    if args.compile:
        inner_model.compile()
        # inner_model_ema.compile()

    if accelerator.is_main_process:
        print(f'Parameters: {K.utils.n_params(inner_model):,}')


    # WANDB LOGGING
    use_wandb = args.use_wandb # accelerator.is_main_process and args.wandb_project
    if accelerator.is_main_process and use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(inner_model)
        wandb.init(project="ionosphere", entity="francescopio", config=log_config, save_code=True)

    lr = opt_config['lr'] if args.lr is None else args.lr
    groups = inner_model.param_groups(lr)
    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(groups,
                        lr=lr,
                        betas=tuple(opt_config['betas']),
                        eps=opt_config['eps'],
                        weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'adam8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(groups,
                                lr=lr,
                                betas=tuple(opt_config['betas']),
                                eps=opt_config['eps'],
                                weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(groups,
                        lr=lr,
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                inv_gamma=sched_config['inv_gamma'],
                                power=sched_config['power'],
                                warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                    num_steps=sched_config['num_steps'],
                                    decay=sched_config['decay'],
                                    warmup=sched_config['warmup'])
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                max_value=ema_sched_config['max_value'])
    ema_stats = {}

    # Load the dataset

    train_dataset, train_sampler, train_dl = get_data_objects(
        path=args.data_path,
        batch_size=args.batch_size,
        distributed=False,
        num_data_workers=args.num_workers,
        split='train',
        seed=42
    )

    val_dataset, val_sampler, val_dl = get_data_objects(
        path=args.data_path,
        batch_size=args.batch_size,
        distributed=False,
        num_data_workers=args.num_workers,
        split='valid',
        seed=42
    )

    print('Train loader and Valid loader are up!')

    # Prepare the model, optimizer, and dataloaders with the accelerator
    inner_model, inner_model_ema, opt, train_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl)

    use_wandb = args.use_wandb

    if use_wandb and accelerator.is_main_process:
        wandb.watch(inner_model)
    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    # Define the model 
    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    state_path = Path(f'{args.name}_state_{args.dir_name}.json')

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unwrap(model.inner_model).load_state_dict(ckpt['model'])
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                    max_value=ema_sched_config['max_value'])
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt

    def save():
        accelerator.wait_for_everyone()
        filename = os.path.join(args.saving_path, dir_path_mdl, f"{args.name}_{step:08}.pth") 
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        inner_model = unwrap(model.inner_model)
        inner_model_ema = unwrap(model_ema.inner_model)
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
            'elapsed': elapsed,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    losses_since_last_print = []
    model = model.to(device)
    try:
        while True:
            # Training Loop
            epoch_train_loss = 0  # Track total training loss
            num_train_batches = len(train_dl)  # Number of batches
            model.train()
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                if device.type == 'cuda':
                    start_timer = torch.cuda.Event(enable_timing=True)
                    end_timer = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_timer.record()
                else:
                    start_timer = time.time()
                
                print('Here')
                with accelerator.accumulate(model):
                    # Track memory before processing
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                    reserved_before = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB
                    print(f"Memory before processing: Allocated: {mem_before:.2f} GB, Reserved: {reserved_before:.2f} GB")

                    inpt = batch[0].contiguous().float().to(device, non_blocking=True)
                    cond_label = batch[1].to(device, non_blocking=True)

                    extra_args = {}
                    noise = torch.randn_like(inpt).to(device)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([inpt.shape[0]], device=device)

                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss(inpt, None, noise, sigma, mapping_cond=cond_label, **extra_args)

                    # Evita NCCL timeout: non fare gather durante il training!
                    loss = losses.mean().item()
                    losses_since_last_print.append(loss)
                    epoch_train_loss += loss  # Accumulate loss

                    # Backward pass
                    accelerator.backward(losses.mean())

                    # Track memory after backward pass
                    torch.cuda.synchronize()
                    mem_after_backward = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                    reserved_after_backward = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB
                    print(f"Memory after backward pass: Allocated: {mem_after_backward:.2f} GB, Reserved: {reserved_after_backward:.2f} GB")

                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, inpt.shape[0], inpt.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    # Track memory after optimizer step
                    torch.cuda.synchronize()
                    mem_after_step = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                    reserved_after_step = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB
                    print(f"Memory after optimizer step: Allocated: {mem_after_step:.2f} GB, Reserved: {reserved_after_step:.2f} GB")

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if device.type == 'cuda':
                    end_timer.record()
                    torch.cuda.synchronize()
                    elapsed += start_timer.elapsed_time(end_timer) / 1000
                else:
                    elapsed += time.time() - start_timer

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')

                step += 1

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    # return
            
            epoch_train_loss /= num_train_batches 

            # **Validation Loop (After Training, Before wandb Logging)**
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dl, desc="Validation", disable=not accelerator.is_main_process):
                    inpt = batch[0].contiguous().float().to(device, non_blocking=True)
                    cond_label = batch[1].to(device, non_blocking=True)

                    extra_args = {}
                    noise = torch.randn_like(inpt).to(device)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([inpt.shape[0]], device=device)

                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss(inpt, None, noise, sigma, mapping_cond=cond_label, **extra_args)

                    # Make sure we only gather scalar loss (not batch tensor)
                    loss_value = losses.mean().detach()
                    gathered_loss = accelerator.gather_for_metrics(loss_value)

                    # Accumulate average across ranks only from main process
                    if accelerator.is_main_process:
                        val_loss += gathered_loss.mean().item()

            # Final averaging
            if accelerator.is_main_process:
                val_loss /= len(val_dl)

            # Print validation loss
            if accelerator.is_main_process:
                tqdm.write(f"Epoch {epoch}, Train Loss: {epoch_train_loss:.6f}, Validation Loss: {val_loss:.6f}")

            # Sampling
            
            if epoch % args.evaluate_every == 0 and accelerator.is_main_process:
                
                # Test sampling 
                samples = generate_samples(model_ema, 1, device, cond_label=cond_label[0].unsqueeze(0), sampler="dpmpp_2m_sde")
                
                # Remove batch dim
                target_tensor = samples[0][0].cpu().numpy() # shape: (5, 3, 256, 256)
                
                fig, (ax1) = plt.subplots(1, 1)
                # Plot the original image in the first subplot
                ax1.imshow(target_tensor, origin='lower')
                ax1.set_title('Predicted Image')
                # Add a big title in the middle of all subplots
                fig.suptitle('Cond_1: {}, Cond_2: {}, Cond_3: {}'.format(cond_label[0][0].item(), cond_label[0][1].item(), cond_label[0][2].item()))
                # Adjust the spacing between subplots
                plt.tight_layout()
                # Show the plot
                plt.show()
                
            # **wandb Logging (Now Includes Validation Loss)**
            if use_wandb and accelerator.is_main_process:
                log_dict = {
                    'epoch': epoch,
                    'loss': epoch_train_loss,
                    'val_loss': val_loss,
                    'lr': sched.get_last_lr()[0],
                    'ema_decay': ema_decay,
                    'Sampled images': wandb.Image(plt)
                }
                if args.gns:
                    log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                
                plt.close()
            save()
            epoch += 1  # Move to the next epoch

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()