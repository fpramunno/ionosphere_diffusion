# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:47:31 2025

@author: pio-r
"""

# ------------------------------------------------------------------
# ✅ Pre-FSDP enum coercion for all distributed ranks
#     must run BEFORE importing accelerate or torch.distributed
# ------------------------------------------------------------------
import os

if os.environ.get("ACCELERATE_USE_FSDP", "false").lower() in {"true", "1"}:
    from torch.distributed.fsdp import ShardingStrategy, StateDictType, BackwardPrefetch

    def _coerce_accelerate_enum(env_name: str, enum_cls):
        value = os.environ.get(env_name)
        if not value or value.isdigit():
            return
        upper = value.upper()
        if upper in enum_cls.__members__:
            os.environ[env_name] = str(enum_cls[upper].value)

    for prefix in ["ACCELERATE_FSDP_", "ACCELERATE_FSDP_FSDP_"]:
        _coerce_accelerate_enum(f"{prefix}SHARDING_STRATEGY", ShardingStrategy)
        _coerce_accelerate_enum(f"{prefix}STATE_DICT_TYPE", StateDictType)
        _coerce_accelerate_enum(f"{prefix}BACKWARD_PREFETCH", BackwardPrefetch)

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
    from IPython import embed 

    import src as K

    from util import generate_samples
    import torch
    from src.data.dataset import get_data_objects, get_sequence_data_objects, get_sequence_data_objects_iterable

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
    p.add_argument('--data-path', type=str, default="/users/framunno/data/ionosphere/ionosphere_data/pickled_maps",
                help='the path of the dataset')
    p.add_argument('--saving-path', type=str, default="/users/framunno/models_results", 
                help='the path where to save the model')
    p.add_argument('--dir-name', type=str, default='cond_forecasting_cfg_oneframe_nonorm',
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
    p.add_argument('--max-epochs', type=int, default=None,
                help='maximum number of epochs to train for')
    p.add_argument('--sequence-length', type=int, default=30,
                help='the total length of the sequence (conditioning + prediction)')
    p.add_argument('--predict-steps', type=int, default=1,
                help='number of future steps to predict')
    p.add_argument('--csv-path', type=str, default="/users/framunno/data/ionosphere/l1_earth_associated_with_maps.csv",
                help='path to the main CSV file with metrics')
    p.add_argument('--transform-cond-csv', type=str, default="/users/framunno/data/ionosphere/params.csv",
                help='path to the transform condition CSV file')
    p.add_argument('--normalization-type', type=str, default="absolute_max",
                choices=["absolute_max", "mean_sigma_tanh", "ionosphere_preprocess"],
                help='type of normalization to use: absolute_max (original), mean_sigma_tanh, or ionosphere_preprocess (SDO-style)')
    p.add_argument('--preprocess-scaling', type=str, default=None,
                choices=[None, "log10", "sqrt", "symlog"],
                help='scaling method for ionosphere_preprocess: None, log10, sqrt, or symlog')
    p.add_argument('--preprocess-scale-factor', type=float, default=10000,
                help='scale factor for symlog preprocessing (only used when --preprocess-scaling=symlog)')
    p.add_argument('--min-center-distance', type=int, default=5,
                help='minimum distance between sequence centers (in frames) to avoid overlap')

    p.add_argument('--wandb-entity', type=str,
                help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                help='save model to wandb')
    p.add_argument('--wandb-runname', type=str,
                help='the run name')
    p.add_argument('--use-wandb', action='store_true', help='Enable wandb logging')
    p.add_argument('--no-wandb', dest='use_wandb', action='store_false', help='Disable wandb logging')
    p.set_defaults(use_wandb=True)  # or False, depending on your preference
    p.add_argument('--debug', action='store_true',
                help='enable debug mode: disables wandb, sets batch_size=4, num_workers=2, evaluate_every=1')
    p.add_argument('--cartesian-transform', action='store_true',
                help='apply Cartesian transform to data')
    p.add_argument('--overfit-single', action='store_true',
                help='overfit on a single trajectory for debugging')
    p.add_argument('--only-complete-sequences', action='store_true',
                help='only use sequences with no missing frames (no zero-padded frames)')

    # FSDP and scaling arguments
    p.add_argument('--use-fsdp', action='store_true',
                help='use FSDP config (make sure to use configs/accelerate_config_fsdp.yaml)')
    p.add_argument('--activation-checkpointing', action='store_true',
                help='enable activation checkpointing (saves memory, slight compute overhead)')
    p.add_argument('--compile-model', action='store_true',
                help='use torch.compile for 20-30%% speedup (requires PyTorch 2.0+)')
    p.add_argument('--use-iterable-dataset', action='store_true',
                help='use IterableDataset for proper multi-GPU/multi-worker sharding (RECOMMENDED for scaling!)')

    args = p.parse_args()

    # Apply debug mode settings
    if args.debug:
        print("🐛 DEBUG MODE ENABLED")
        print("  - Disabling wandb")
        print("  - Setting batch_size=4")
        print("  - Setting num_workers=2")
        print("  - Setting evaluate_every=1")
        print("  - Setting max_epochs=5 (if not already set)")
        args.use_wandb = False
        args.batch_size = 4
        args.num_workers = 2
        args.evaluate_every = 1
        if args.max_epochs is None:
            args.max_epochs = 5

    # Calculate conditioning length from total sequence length and prediction steps
    args.conditioning_length = args.sequence_length - args.predict_steps
    
    if args.conditioning_length <= 0:
        raise ValueError(f"Conditioning length ({args.conditioning_length}) must be positive. "
                        f"sequence_length ({args.sequence_length}) must be greater than predict_steps ({args.predict_steps})")
    
    print(f"📊 Sequence Configuration:")
    print(f"  Total sequence length: {args.sequence_length}")
    print(f"  Conditioning frames: {args.conditioning_length}")
    print(f"  Prediction frames: {args.predict_steps}")

    dir_path_res = os.path.join(args.saving_path, f"results_{args.dir_name}")
    dir_path_mdl = os.path.join(args.saving_path, f"models_{args.dir_name}")

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

    assert len(model_config['input_size']) == 2 # and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    # # ------------------------------------------------------------------
    # # Accelerate configuration tweaks
    # # ------------------------------------------------------------------
    # def _env_flag_enabled(name: str) -> bool:
    #     value = os.environ.get(name, "")
    #     if not value:
    #         return False
    #     return value.lower() in {"1", "true", "yes", "y", "on"}

    # # ✅ ensure the flag is correctly set for subprocesses
    # if "ACCELERATE_USE_FSDP" in os.environ and os.environ["ACCELERATE_USE_FSDP"].lower() in {"true", "1"}:
    #     os.environ["ACCELERATE_USE_FSDP"] = "true"

    # if _env_flag_enabled("ACCELERATE_USE_FSDP"):
    #     from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy, StateDictType

    #     def _coerce_accelerate_enum(env_name: str, enum_cls) -> None:
    #         value = os.environ.get(env_name)
    #         if not value or value.isdigit():
    #             return
    #         upper = value.upper()
    #         if upper in enum_cls.__members__:
    #             os.environ[env_name] = str(enum_cls[upper].value)

    #     # Apply to both possible key formats (old/new Accelerate)
    #     for prefix in ["ACCELERATE_FSDP_", "ACCELERATE_FSDP_FSDP_"]:
    #         _coerce_accelerate_enum(f"{prefix}SHARDING_STRATEGY", ShardingStrategy)
    #         _coerce_accelerate_enum(f"{prefix}STATE_DICT_TYPE", StateDictType)
    #         _coerce_accelerate_enum(f"{prefix}BACKWARD_PREFETCH", BackwardPrefetch)


    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps,
                                            mixed_precision=args.mixed_precision,
                kwargs_handlers=[accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=False)])

    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size per GPU: {args.batch_size}', flush=True)
        print(f'Global batch size: {args.batch_size * accelerator.num_processes}', flush=True)
        print(f'Effective batch size: {args.batch_size * accelerator.num_processes * args.grad_accum_steps}', flush=True)
        print(f'Mixed precision: {args.mixed_precision}', flush=True)
        if args.use_fsdp:
            print(f'Using FSDP: ✅')
        if args.activation_checkpointing:
            print(f'Activation checkpointing: ✅')
        if args.compile_model:
            print(f'Torch compile: ✅')

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
    elapsed = 0.0

    # Model definition
    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)

    # Enable activation checkpointing if requested (saves memory)
    if args.activation_checkpointing or args.checkpointing:
        # This will use gradient checkpointing in transformer blocks
        if hasattr(inner_model, 'set_gradient_checkpointing'):
            inner_model.set_gradient_checkpointing(True)
            if accelerator.is_main_process:
                print("✅ Enabled gradient checkpointing on model")

    # Apply torch.compile for speedup
    if args.compile or args.compile_model:
        if accelerator.is_main_process:
            print("Compiling model with torch.compile...")
        inner_model = torch.compile(inner_model, mode='max-autotune')
        # Don't compile EMA model (not used in training loop)

    if accelerator.is_main_process:
        print(f'Parameters: {K.utils.n_params(inner_model):,}')


    # WANDB LOGGING
    use_wandb = args.use_wandb # accelerator.is_main_process and args.wandb_project
    if accelerator.is_main_process and use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(inner_model)
        wandb.init(project="ionosphere", entity="francescopio", name=args.wandb_runname, config=log_config, save_code=True)
    
    # MODEL SUMMARY
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in inner_model.parameters())
        trainable_params = sum(p.numel() for p in inner_model.parameters() if p.requires_grad)
        
        print(f'{"="*80}')
        print(f'🤖 MODEL ARCHITECTURE SUMMARY')
        print(f'{"="*80}')
        
        # Layer-by-layer breakdown
        print(f'📋 LAYER-BY-LAYER BREAKDOWN:')
        print(f'{"Layer Name":<40} {"Shape":<20} {"Parameters":<15} {"Trainable":<10}')
        print(f'{"-"*80}')
        
        layer_count = 0
        for name, param in inner_model.named_parameters():
            layer_count += 1
            shape_str = "x".join(map(str, param.shape))
            param_count = param.numel()
            trainable = "Yes" if param.requires_grad else "No"
            
            # Truncate long layer names
            display_name = name if len(name) <= 39 else name[:36] + "..."
            
            print(f'{display_name:<40} {shape_str:<20} {param_count:<15,} {trainable:<10}')
        
        print(f'{"-"*80}')
        
        # Summary by module type
        print(f'\n📊 SUMMARY BY MODULE TYPE:')
        module_stats = {}
        for name, module in inner_model.named_modules():
            if len(list(module.parameters(recurse=False))) > 0:  # Only modules with direct parameters
                module_type = type(module).__name__
                if module_type not in module_stats:
                    module_stats[module_type] = {'count': 0, 'params': 0}
                module_stats[module_type]['count'] += 1
                module_stats[module_type]['params'] += sum(p.numel() for p in module.parameters(recurse=False))
        
        print(f'{"Module Type":<25} {"Count":<8} {"Parameters":<15} {"% of Total":<12}')
        print(f'{"-"*65}')
        for module_type, stats in sorted(module_stats.items(), key=lambda x: x[1]['params'], reverse=True):
            percentage = 100.0 * stats['params'] / total_params
            print(f'{module_type:<25} {stats["count"]:<8} {stats["params"]:<15,} {percentage:<12.1f}%')
        
        print(f'\n💾 OVERALL SUMMARY:')
        print(f'Total layers with parameters: {layer_count}')
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        print(f'Non-trainable parameters: {total_params - trainable_params:,}')
        print(f'Model size (MB): {total_params * 4 / (1024**2):.2f}')
        print(f'{"="*80}')


    # Load the dataset

    # Create preprocessing config if using ionosphere_preprocess normalization
    preprocess_config = None
    if args.normalization_type == "ionosphere_preprocess":
        preprocess_config = {
            "min": -80000,
            "max": 80000,
            "scaling": args.preprocess_scaling,
        }
        if args.preprocess_scaling == "symlog":
            preprocess_config["scale_factor"] = args.preprocess_scale_factor

        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print("PREPROCESSING CONFIG (SDO-style)")
            print(f"{'='*80}")
            print(f"  Min/Max clipping: [{preprocess_config['min']}, {preprocess_config['max']}]")
            print(f"  Scaling method: {preprocess_config['scaling']}")
            if args.preprocess_scaling == "symlog":
                print(f"  Scale factor: {preprocess_config['scale_factor']}")
            print(f"{'='*80}\n")

    # Choose between regular Dataset and IterableDataset
    if args.use_iterable_dataset:
        print("✅ Using IterableDataset for proper multi-GPU/multi-worker sharding!")
        get_data_fn = get_sequence_data_objects_iterable
    else:
        print("Using regular Dataset (map-style)")
        get_data_fn = get_sequence_data_objects

    train_dataset, train_sampler, train_dl = get_data_fn(
        csv_path=args.csv_path,
        transform_cond_csv=args.transform_cond_csv,
        batch_size=args.batch_size,
        num_data_workers=args.num_workers,
        split='train',
        seed=42,
        sequence_length=args.sequence_length,
        normalization_type=args.normalization_type,
        preprocess_config=preprocess_config,
        use_l1_conditions=True,
        min_center_distance=args.min_center_distance,
        cartesian_transform=args.cartesian_transform,
        output_size=64,
        only_complete_sequences=args.only_complete_sequences,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_dataset, val_sampler, val_dl = get_data_fn(
        csv_path=args.csv_path,
        transform_cond_csv=args.transform_cond_csv,
        batch_size=args.batch_size,
        num_data_workers=args.num_workers,
        split='valid',
        seed=42,
        sequence_length=args.sequence_length,
        normalization_type=args.normalization_type,
        preprocess_config=preprocess_config,
        use_l1_conditions=True,
        min_center_distance=args.min_center_distance,
        cartesian_transform=args.cartesian_transform,
        output_size=64,
        only_complete_sequences=args.only_complete_sequences,
        persistent_workers=True,
        prefetch_factor=4,
    )

    print(f'Train loader and Valid loader are up! Lengths: {len(train_dl)}, {len(val_dl)}')
    print(f'Using normalization method: {args.normalization_type}')

    # Overfitting on a single trajectory
    if args.overfit_single:
        if accelerator.is_main_process:
            print("🎯 OVERFITTING MODE: Using only 1 trajectory")

        # Get one batch from the training set (batch_images, batch_conditions)
        batch_images, batch_conditions = next(iter(train_dl))

        if accelerator.is_main_process:
            print(f"   Original batch shapes: images={batch_images.shape}, conditions={batch_conditions.shape}")

        # Extract the FIRST sample from the batch (remove batch dimension)
        # batch_images: [batch_size, seq_len, C, H, W] -> [seq_len, C, H, W]
        # batch_conditions: [batch_size, seq_len, num_cond] -> [seq_len, num_cond]
        single_image = batch_images[0]
        single_condition = batch_conditions[0]

        if accelerator.is_main_process:
            print(f"   Single sample shapes: images={single_image.shape}, conditions={single_condition.shape}")

        # Create a dataset that repeats this single sample
        class SingleSampleDataset(torch.utils.data.Dataset):
            def __init__(self, image, condition, repeat=1000):
                self.image = image
                self.condition = condition
                self.repeat = repeat

            def __len__(self):
                return self.repeat

            def __getitem__(self, idx):
                return self.image, self.condition

        # Repeat the single sample enough times for a reasonable epoch
        repeat_count_train = 10000 * args.batch_size
        repeat_count_val = repeat_count_train // 10  # Validation set is 1/10 of training set

        single_dataset_train = SingleSampleDataset(single_image, single_condition, repeat=repeat_count_train)
        single_dataset_val = SingleSampleDataset(single_image, single_condition, repeat=repeat_count_val)

        if accelerator.is_main_process:
            print(f"   Train dataset size: {repeat_count_train} samples = {repeat_count_train // args.batch_size} batches per epoch (total)")
            print(f"   Train per-GPU batches: {repeat_count_train // args.batch_size // accelerator.num_processes}")
            print(f"   Val dataset size: {repeat_count_val} samples = {repeat_count_val // args.batch_size} batches per epoch (total)")
            print(f"   Val per-GPU batches: {repeat_count_val // args.batch_size // accelerator.num_processes}")

        # Create DistributedSampler to shard across GPUs
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            single_dataset_train,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False
        )

        # Create new dataloader with the single sample
        # Use original batch_size so the model sees the expected batch dimension
        train_dl = torch.utils.data.DataLoader(
            single_dataset_train,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )

        # Create validation dataloader (1/10 the size of training)
        val_sampler = DistributedSampler(
            single_dataset_val,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False
        )
        val_dl = torch.utils.data.DataLoader(
            single_dataset_val,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
        )

        if accelerator.is_main_process:
            print(f"   New dataloader will produce batches of size {args.batch_size} with the same trajectory repeated")

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
    elif sched_config['type'] == 'cosine':
        # Calculate total steps
        if args.max_epochs is None:
            raise ValueError("max_epochs must be specified when using cosine scheduler")

        epoch_size = len(train_dl)  # steps per epoch
        warmup_epochs = sched_config.get('warmup_epochs', 0)
        warmup_steps = (warmup_epochs * epoch_size) // args.grad_accum_steps
        total_steps = (args.max_epochs * epoch_size) // args.grad_accum_steps
        eta_min_factor = sched_config.get('eta_min_factor', 100)  # LR will decay to lr/100

        if warmup_steps > 0:
            # Cosine with linear warmup
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=0.01,  # Start at 1% of base LR
                end_factor=1.0,     # Reach 100% of base LR
                total_iters=warmup_steps
            )
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                eta_min=lr / eta_min_factor,  # Min LR = lr/100
                T_max=total_steps - warmup_steps
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup, decay],
                milestones=[warmup_steps]
            )

            if accelerator.is_main_process:
                print(f"📊 Cosine LR Schedule:")
                print(f"   Warmup steps: {warmup_steps} (epochs: {warmup_epochs})")
                print(f"   Total steps: {total_steps} (epochs: {args.max_epochs})")
                print(f"   Base LR: {lr:.2e}")
                print(f"   Min LR: {lr/eta_min_factor:.2e}")
        else:
            # Cosine without warmup
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                eta_min=lr / eta_min_factor,
                T_max=total_steps
            )

            if accelerator.is_main_process:
                print(f"📊 Cosine LR Schedule (no warmup):")
                print(f"   Total steps: {total_steps}")
                print(f"   Base LR: {lr:.2e}")
                print(f"   Min LR: {lr/eta_min_factor:.2e}")
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                max_value=ema_sched_config['max_value'])
    ema_stats = {}


    # Prepare the model, optimizer, and dataloaders with the accelerator
    if not args.overfit_single:
        inner_model, inner_model_ema, opt, train_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl)
    else:
        # For overfitting mode, don't prepare dataloaders (they're already simple)
        inner_model, inner_model_ema, opt = accelerator.prepare(inner_model, inner_model_ema, opt)

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
        wandb_step = ckpt.get('wandb_step', 0)  # Load wandb_step if resuming
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)

        del ckpt
    else:
        epoch = 0
        step = 0
        wandb_step = 0  # Start from 0 for new runs

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
        filename = os.path.join(args.saving_path, dir_path_mdl, f"{args.name}_epoch_{epoch:04}.pth") 
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
            'wandb_step': wandb_step,
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

    # PROFILING: Track timing for different parts of training loop
    timing_stats = {
        'data_loading': [],
        'data_transfer': [],
        'forward': [],
        'backward': [],
        'optimizer': [],
        'ema': [],
    }

    model = model.to(device)
    try:
        while args.max_epochs is None or epoch < args.max_epochs:
            # Training Loop
            epoch_train_loss = 0  # Track total training loss
            num_train_batches = len(train_dl)  # Number of batches
            model.train()

            batch_start_time = time.perf_counter()
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                # TIMING: Data loading time
                data_load_time = time.perf_counter() - batch_start_time
                timing_stats['data_loading'].append(data_load_time)

                with accelerator.accumulate(model):
                    # TIMING: Data transfer to GPU
                    transfer_start = time.perf_counter()
                    inpt = batch[0].contiguous().float().to(device, non_blocking=True)
                    inpt = inpt.squeeze(2)  # shape: (batch_size, sequence_length, 24, 360)
                    cond_img = inpt[:, :args.conditioning_length, :, :]    # first conditioning_length time steps
                    target_img = inpt[:, args.conditioning_length:args.conditioning_length+args.predict_steps, :, :]  # next predict_steps time steps
                    cond_label = batch[1].to(device, non_blocking=True)
                    cond_label_inp = cond_label[:, :args.conditioning_length+args.predict_steps, :]  # :16
                    torch.cuda.synchronize()  # Wait for transfers to complete
                    transfer_time = time.perf_counter() - transfer_start
                    timing_stats['data_transfer'].append(transfer_time)

                    # TIMING: Forward pass
                    forward_start = time.perf_counter()
                    extra_args = {}
                    noise = torch.randn_like(target_img).to(device)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([target_img.shape[0]], device=device)

                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss(target_img, cond_img, noise, sigma, mapping_cond=cond_label_inp, **extra_args)
                    torch.cuda.synchronize()  # Wait for forward to complete
                    forward_time = time.perf_counter() - forward_start
                    timing_stats['forward'].append(forward_time)

                    # Evita NCCL timeout: non fare gather durante il training!
                    loss = losses.mean().item()
                    losses_since_last_print.append(loss)
                    epoch_train_loss += loss  # Accumulate loss

                    # TIMING: Backward pass
                    backward_start = time.perf_counter()
                    accelerator.backward(losses.mean())
                    torch.cuda.synchronize()  # Wait for backward to complete
                    backward_time = time.perf_counter() - backward_start
                    timing_stats['backward'].append(backward_time)

                    # TIMING: Optimizer step
                    opt_start = time.perf_counter()
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, inpt.shape[0], inpt.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    torch.cuda.synchronize()  # Wait for optimizer to complete
                    opt_time = time.perf_counter() - opt_start
                    timing_stats['optimizer'].append(opt_time)

                    # TIMING: EMA update
                    ema_start = time.perf_counter()
                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()
                    ema_time = time.perf_counter() - ema_start
                    timing_stats['ema'].append(ema_time)

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        # PROFILING: Report timing breakdown
                        if len(timing_stats['forward']) > 0:
                            avg_data_load = sum(timing_stats['data_loading']) / len(timing_stats['data_loading']) * 1000
                            avg_transfer = sum(timing_stats['data_transfer']) / len(timing_stats['data_transfer']) * 1000
                            avg_forward = sum(timing_stats['forward']) / len(timing_stats['forward']) * 1000
                            avg_backward = sum(timing_stats['backward']) / len(timing_stats['backward']) * 1000
                            avg_optimizer = sum(timing_stats['optimizer']) / len(timing_stats['optimizer']) * 1000
                            avg_ema = sum(timing_stats['ema']) / len(timing_stats['ema']) * 1000
                            total_time = avg_data_load + avg_transfer + avg_forward + avg_backward + avg_optimizer + avg_ema

                            tqdm.write(f'\n{"="*80}')
                            tqdm.write(f'⏱️  TIMING BREAKDOWN (Step {step}):')
                            tqdm.write(f'{"="*80}')
                            tqdm.write(f'  Data Loading:   {avg_data_load:>8.2f}ms ({avg_data_load/total_time*100:>5.1f}%)')
                            tqdm.write(f'  Data Transfer:  {avg_transfer:>8.2f}ms ({avg_transfer/total_time*100:>5.1f}%)')
                            tqdm.write(f'  Forward Pass:   {avg_forward:>8.2f}ms ({avg_forward/total_time*100:>5.1f}%)')
                            tqdm.write(f'  Backward Pass:  {avg_backward:>8.2f}ms ({avg_backward/total_time*100:>5.1f}%)')
                            tqdm.write(f'  Optimizer:      {avg_optimizer:>8.2f}ms ({avg_optimizer/total_time*100:>5.1f}%)')
                            tqdm.write(f'  EMA Update:     {avg_ema:>8.2f}ms ({avg_ema/total_time*100:>5.1f}%)')
                            tqdm.write(f'  {"─"*80}')
                            tqdm.write(f'  TOTAL:          {total_time:>8.2f}ms ({total_time/1000:.2f}s per iteration)')
                            tqdm.write(f'{"="*80}\n')

                            # Reset timing stats
                            for key in timing_stats:
                                timing_stats[key].clear()

                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')

                        # Log to wandb every 25 steps (removed GPU memory logging - it requires sync)
                        if use_wandb:
                            wandb.log({
                                'train/loss_step': loss_disp,
                                'train/avg_loss': avg_loss,
                            }, step=wandb_step)

                step += accelerator.num_processes
                wandb_step += accelerator.num_processes  # Count steps across all GPUs
                batch_start_time = time.perf_counter()  # Start timing for next batch

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    # return
            
            epoch_train_loss /= num_train_batches

            # Gather training loss across all ranks for accurate logging
            epoch_train_loss_tensor = torch.tensor(epoch_train_loss, device=device)
            gathered_train_loss = accelerator.gather(epoch_train_loss_tensor)
            if accelerator.is_main_process:
                epoch_train_loss = gathered_train_loss.mean().item()

            # **Validation Loop (After Training, Before wandb Logging)**
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dl, desc="Validation", disable=not accelerator.is_main_process):
                    inpt = batch[0].contiguous().float().to(device, non_blocking=True)
                    inpt = inpt.squeeze(2)  # shape: (batch_size, sequence_length, 24, 360)
                    cond_img = inpt[:, :args.conditioning_length, :, :]    # first conditioning_length time steps
                    target_img = inpt[:, args.conditioning_length:args.conditioning_length+args.predict_steps, :, :]  # next predict_steps time steps
                    cond_label = batch[1].to(device, non_blocking=True)

                    cond_label_inp = cond_label[:, :args.conditioning_length+args.predict_steps, :]  # :16


                    extra_args = {}
                    noise = torch.randn_like(target_img).to(device)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([target_img.shape[0]], device=device)

                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss(target_img, cond_img, noise, sigma, mapping_cond=cond_label_inp, **extra_args)

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

            # Sampling and Visualization
            
            if epoch % args.evaluate_every == 0 and accelerator.is_main_process:

                # Get spatial dimensions based on cartesian_transform flag
                if args.cartesian_transform:
                    spatial_shape = (64, 64)
                else:
                    spatial_shape = (24, 360)

                # Test sampling
                samples = generate_samples(model_ema, 1, device, cond_label=cond_label_inp[0, :args.conditioning_length+args.predict_steps, :].reshape(1, args.conditioning_length+args.predict_steps, 4), sampler="dpmpp_2m_sde", cond_img=cond_img[0].reshape(1, args.conditioning_length, *spatial_shape), num_pred_frames=args.predict_steps).cpu()

                import matplotlib.pyplot as plt
                import imageio
                import numpy as np
                import torch
                import io

                # Get the generated sample and target for comparison
                generated_sample = samples[0]  # shape: [predict_steps, H, W]
                target_sample = target_img[0].cpu().numpy()  # shape: [predict_steps, H, W]

                # For visualization, use the first prediction step (you can modify this to show all steps)
                generated_sample_np = generated_sample[0].cpu().numpy()  # shape: [H, W] - first predicted frame
                target_sample_first = target_sample[0]  # shape: [H, W] - first target frame

                # Revert transformation to original scale for better visualization
                generated_sample_orig = generated_sample_np * 80000.0
                target_sample_orig = target_sample_first * 80000.0

                # Create visualizations based on cartesian_transform and prediction steps
                if args.cartesian_transform:
                    # Use regular imshow for Cartesian data (224x224)
                    if args.predict_steps == 1:
                        # Single step prediction - show comparison
                        fig_comparison, axes = plt.subplots(1, 2, figsize=(16, 8))
                        im0 = axes[0].imshow(target_sample_orig, cmap='plasma', aspect='auto')
                        axes[0].set_title("Target (Ground Truth)")
                        axes[0].axis('off')
                        plt.colorbar(im0, ax=axes[0], shrink=0.8)

                        im1 = axes[1].imshow(generated_sample_orig, cmap='plasma', aspect='auto')
                        axes[1].set_title("Generated Prediction")
                        axes[1].axis('off')
                        plt.colorbar(im1, ax=axes[1], shrink=0.8)

                        fig_single, ax = plt.subplots(figsize=(10, 10))
                        im = ax.imshow(generated_sample_orig, cmap='plasma', aspect='auto')
                        ax.set_title(f"Generated Ionosphere Prediction - Epoch {epoch}")
                        ax.axis('off')
                        plt.colorbar(im, ax=ax, shrink=0.8)
                    else:
                        # Multi-step prediction
                        generated_all_orig = generated_sample.cpu().numpy() * 80000.0  # [predict_steps, 224, 224]
                        target_all_orig = target_sample * 80000.0  # [predict_steps, 224, 224]
                        generated_sample_orig = generated_all_orig  # For vmin/vmax in later plotting

                        # Handle case where dimensions might be squeezed
                        target_first = target_all_orig[0] if len(target_all_orig.shape) > 2 else target_all_orig
                        gen_first = generated_all_orig[0] if len(generated_all_orig.shape) > 2 else generated_all_orig

                        fig_comparison, axes = plt.subplots(1, 2, figsize=(16, 8))
                        im0 = axes[0].imshow(target_first, cmap='plasma', aspect='auto')
                        axes[0].set_title("Target Step 1")
                        axes[0].axis('off')
                        plt.colorbar(im0, ax=axes[0], shrink=0.8)

                        im1 = axes[1].imshow(gen_first, cmap='plasma', aspect='auto')
                        axes[1].set_title("Generated Step 1")
                        axes[1].axis('off')
                        plt.colorbar(im1, ax=axes[1], shrink=0.8)
                        fig_comparison.suptitle(f"Multi-Step Forecasting - Epoch {epoch}", fontsize=16)

                        # Single plot shows the last predicted frame
                        fig_single, ax = plt.subplots(figsize=(10, 10))
                        im = ax.imshow(generated_all_orig[-1], cmap='plasma', aspect='auto')
                        ax.set_title(f"Generated Final Frame ({args.predict_steps}/{args.predict_steps}) - Epoch {epoch}")
                        ax.axis('off')
                        plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    # Use polar plotting for polar data (24x360)
                    from util import plot_polar_ionosphere_single, plot_polar_ionosphere_comparison

                    if args.predict_steps == 1:
                        # Single step prediction - show comparison
                        fig_comparison, _ = plot_polar_ionosphere_comparison(
                            data_original=target_sample_orig,
                            data_pred=generated_sample_orig,
                            titles=["Target (Ground Truth)", "Generated Prediction"],
                            cmap='plasma',
                            figsize=(16, 8)
                        )

                        fig_single, _ = plot_polar_ionosphere_single(
                            data=generated_sample_orig,
                            title=f"Generated Ionosphere Prediction - Epoch {epoch}",
                            cmap='plasma',
                            figsize=(10, 10)
                        )
                    else:
                        # Multi-step prediction - show sequence evolution
                        generated_all_orig = generated_sample.cpu().numpy() * 80000.0  # [predict_steps, 24, 360]
                        target_all_orig = target_sample * 80000.0  # [predict_steps, 24, 360]
                        generated_sample_orig = generated_all_orig  # For vmin/vmax in later plotting

                        # Handle case where dimensions might be squeezed
                        target_first = target_all_orig[0] if len(target_all_orig.shape) > 2 else target_all_orig
                        gen_first = generated_all_orig[0] if len(generated_all_orig.shape) > 2 else generated_all_orig

                        fig_comparison, _ = plot_polar_ionosphere_comparison(
                            data_original=target_first,
                            data_pred=gen_first,
                            titles=["Target Step 1", "Generated Step 1"],
                            cmap='plasma',
                            figsize=(16, 8)
                        )
                        fig_comparison.suptitle(f"Multi-Step Forecasting - Epoch {epoch}\nTop: Targets, Bottom: Generated", fontsize=16)

                        # Single plot shows the last predicted frame
                        fig_single, _ = plot_polar_ionosphere_single(
                            data=generated_all_orig[-1],  # Last predicted frame
                            title=f"Generated Final Frame ({args.predict_steps}/{args.predict_steps}) - Epoch {epoch}",
                            cmap='plasma',
                            figsize=(10, 10)
                        )
                
                # Convert matplotlib figures to images for wandb
                buf = io.BytesIO()
                fig_comparison.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                buf_single = io.BytesIO()
                fig_single.savefig(buf_single, format='png', dpi=150, bbox_inches='tight')
                buf_single.seek(0)

                # Create traditional sequence animation for gif
                frames = []
                # For sequence, show the conditioning frames + all generated frames
                full_sequence = torch.cat([cond_img[0].cpu(), target_img[0].cpu()], dim=0)  # [sequence_length, H, W]

                # Also create sequence with generated samples
                generated_full_sequence = torch.cat([cond_img[0].cpu(), generated_sample.cpu()], dim=0)  # [conditioning_length + predict_steps, H, W]
                
                # Dynamically set figsize
                img_h, img_w = full_sequence.shape[1], full_sequence.shape[2]
                aspect = img_w / img_h
                base_height = 4
                figsize = (base_height * aspect, base_height)

                # Create frames for target sequence
                for t in range(full_sequence.shape[0]):
                    img = full_sequence[t].cpu().numpy() * 80000.0  # Original scale
                    fig, ax = plt.subplots(figsize=figsize)
                    im = ax.imshow(img, cmap='plasma', aspect='auto', vmin=generated_sample_orig.min(), vmax=generated_sample_orig.max())
                    
                    if t < args.conditioning_length:
                        ax.set_title(f"Conditioning Frame {t+1}/{args.conditioning_length}")
                    else:
                        pred_step = t - args.conditioning_length + 1
                        ax.set_title(f"Target Frame {pred_step}/{args.predict_steps}")
                    
                    ax.axis('off')
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    
                    # Convert plot to image array
                    fig.canvas.draw()
                    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(frame)
                    plt.close(fig)

                # Add separator frame
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, 'PREDICTED SEQUENCE', ha='center', va='center', fontsize=20, 
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue'))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
                plt.close(fig)

                # Create frames for generated sequence
                for t in range(generated_full_sequence.shape[0]):
                    img = generated_full_sequence[t].cpu().numpy() * 55000.0  # Original scale
                    fig, ax = plt.subplots(figsize=figsize)
                    im = ax.imshow(img, cmap='plasma', aspect='auto', vmin=generated_sample_orig.min(), vmax=generated_sample_orig.max())
                    
                    if t < args.conditioning_length:
                        ax.set_title(f"Conditioning Frame {t+1}/{args.conditioning_length}")
                    else:
                        pred_step = t - args.conditioning_length + 1
                        ax.set_title(f"Generated Frame {pred_step}/{args.predict_steps}")
                    
                    ax.axis('off')
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    
                    # Convert plot to image array
                    fig.canvas.draw()
                    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(frame)
                    plt.close(fig)

                # Save sequence as gif
                imageio.mimsave(f'{dir_path_res}/sequence_gen_{epoch}.gif', frames, duration=0.8)
                
                # Log to wandb
                if use_wandb:
                    import wandb
                    from PIL import Image
                    
                    # Calculate metrics for all prediction steps
                    generated_all_orig = generated_sample.cpu().numpy() * 80000.0  # [predict_steps, H, W]
                    target_all_orig = (torch.cat([cond_img, target_img], dim=1) * 80000.0).cpu().numpy()  # [predict_steps, H, W]

                    # Overall metrics across all prediction steps
                    mse_overall = np.mean((generated_all_orig - target_all_orig) ** 2)
                    mae_overall = np.mean(np.abs(generated_all_orig - target_all_orig))
                    
                    # Convert BytesIO buffers to PIL Images for wandb
                    buf.seek(0)
                    pil_comparison = Image.open(buf)
                    buf_single.seek(0)
                    pil_single = Image.open(buf_single)
                    
                    # Per-step metrics
                    eval_dict = {
                        'evaluation/polar_comparison': wandb.Image(pil_comparison, caption=f"Polar comparison at epoch {epoch} (step 1/{args.predict_steps})"),
                        'evaluation/polar_generated': wandb.Image(pil_single, caption=f"Generated polar plot at epoch {epoch} (step 1/{args.predict_steps})"),
                        'evaluation/sequence_animation': wandb.Video(f'{dir_path_res}/sequence_gen_{epoch}.gif', 
                                                                   fps=1.25, format="gif"),
                        'evaluation/mse_overall': mse_overall,
                        'evaluation/mae_overall': mae_overall,
                    }
                    
                    # Add per-step metrics if predicting multiple steps
                    if args.predict_steps > 1:
                        for step in range(args.predict_steps):
                            step_mse = np.mean((generated_all_orig[step] - target_all_orig[step]) ** 2)
                            step_mae = np.mean(np.abs(generated_all_orig[step] - target_all_orig[step]))
                            eval_dict[f'evaluation/mse_step_{step+1}'] = step_mse
                            eval_dict[f'evaluation/mae_step_{step+1}'] = step_mae
                    
                    wandb.log(eval_dict, step=wandb_step)
                
                # Clean up
                plt.close(fig_comparison)
                plt.close(fig_single)
                buf.close()
                buf_single.close()
                
            # **wandb Logging (Now Includes Validation Loss and Max-Min Difference)**
            if use_wandb and accelerator.is_main_process:
                # Get spatial dimensions based on cartesian_transform flag
                if args.cartesian_transform:
                    spatial_shape = (64, 64)
                else:
                    spatial_shape = (24, 360)

                # Calculate max-min difference after reverting transformation
                # Use current batch target_img for consistent max-min calculation
                # torch.cat([unet_cond, noised_input], dim=1)
                target_img_reverted = torch.cat([cond_img, target_img], dim=1) * 80000.0  # [batch_size, predict_steps, H, W]

                if args.predict_steps == 1:
                    # Single frame prediction - use the single frame
                    target_reverted_flat = target_img_reverted.flatten()
                    max_min_diff_gt = (target_reverted_flat.max() - target_reverted_flat.min()).item()

                    pred_reverted = generated_sample[0].cpu().numpy() * 80000.0  # [H, W]
                    pred_reverted_flat = pred_reverted.flatten()
                    max_min_diff_pred = (pred_reverted_flat.max() - pred_reverted_flat.min()).item()
                else:
                    # Multi-step prediction - compute max-min across entire predicted sequence
                    # Reshape to [batch_size * predict_steps, H, W] then flatten
                    target_reverted_seq = target_img_reverted.reshape(-1, *spatial_shape)  # [batch_size * predict_steps, H, W]
                    target_reverted_flat = target_reverted_seq.flatten()
                    max_min_diff_gt = (target_reverted_flat.max() - target_reverted_flat.min()).item()

                    pred_reverted_seq = generated_sample.cpu().numpy() * 80000.0  # [predict_steps, H, W]
                    pred_reverted_seq = pred_reverted_seq.reshape(-1, *spatial_shape)  # [predict_steps, H, W]
                    pred_reverted_flat = pred_reverted_seq.flatten()
                    max_min_diff_pred = (pred_reverted_flat.max() - pred_reverted_flat.min()).item()

                log_dict = {
                    'epoch': epoch,
                    'loss': epoch_train_loss,
                    'val_loss': val_loss,
                    'lr': sched.get_last_lr()[0],
                    'ema_decay': ema_decay,
                    'max_min_difference_sequence': max_min_diff_gt,
                    'max_min_difference_sequence_pred': max_min_diff_pred,
                }
                if args.gns:
                    log_dict['gradient_noise_scale'] = gns_stats.get_gns()

                wandb.log(log_dict, step=wandb_step)

            # Save every 5 epochs or at the end
            if epoch % 5 == 0 or (args.max_epochs is not None and epoch >= args.max_epochs - 1):
                save()
            epoch += 1  # Move to the next epoch
            
            # Check if we've reached max epochs
            if args.max_epochs is not None and epoch >= args.max_epochs:
                if accelerator.is_main_process:
                    tqdm.write(f'Reached maximum epochs ({args.max_epochs}). Training complete!')
                break

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()