import argparse
import os
import sys
from copy import deepcopy
import json
from pathlib import Path
import time

# Add attri_VAE directory to path (for training_testing_functions, model, etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add ionosphere directory to path (for src.data.dataset)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch._dynamo
from torch import device, optim
from tqdm.auto import tqdm
import accelerate
from IPython import embed
import wandb
import matplotlib.pyplot as plt

def count_parameters(model):
    """Count and print model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"\n{'='*50}")
    print(f"MODEL PARAMETERS")
    print(f"{'='*50}")
    print(f"  Total parameters:        {total_params:,}")
    print(f"  Trainable parameters:    {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"  Model size:              {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"{'='*50}\n")

    return total_params, trainable_params


def visualize_reconstruction(model, data, device, epoch, save_path=None, num_samples=4):
    """Visualize original vs reconstructed images."""
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        if len(data.shape) == 3:
            data = data.unsqueeze(1)  # Add channel dimension if needed

        data = data[:num_samples].to(device)
        recon, *_ = model(data)

        # Move to CPU for plotting
        data_cpu = data.cpu().numpy()
        recon_cpu = recon.cpu().numpy()

        # Create figure (squeeze=False ensures 2D array even with 1 column)
        fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6), squeeze=False)

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(data_cpu[i, 0], cmap='viridis')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # Reconstructed
            axes[1, i].imshow(recon_cpu[i, 0], cmap='viridis')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')

        plt.suptitle(f'Epoch {epoch}: Original vs Reconstructed')
        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'reconstruction_epoch_{epoch}.png'), dpi=150)

        return fig

# VAE-specific imports (from attri_VAE)
from training_testing_funcs import save_ckp, load_ckp
from model.model import ConvVAE, ConvVAE2D, initialize_weights, initialize_weights_2d, initialize_weights_unet, UNetVAE
from model.loss_functions import reconstruction_loss, mlp_loss_function, KL_loss, reg_loss, mean_accuracy, PerceptualLoss 

# Discriminator loss
from taming.modules.discriminator.model import NLayerDiscriminator
from taming.modules.losses.vqperceptual import hinge_d_loss

# Data loading from ionosphere (parent directory)
from src.data.dataset import get_sequence_data_objects_iterable

def main():
    '''
    Define the configurations
    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Configurations for the training/testing
    p.add_argument('--batch-size', type=int, default=16,
                help='the batch size')
    p.add_argument('--num-workers', type=int, default=1,
                help='the number of data loader workers')
    p.add_argument('--seed', type=int, default=3,
                help='the random seed')
    p.add_argument('--epochs', type=int, default=5000,
                help='number of training epochs')
    p.add_argument('--learning-rate', type=float, default=0.0001,
                help='the learning rate')

    # Configurations for VAE
    p.add_argument('--img-channels', type=int, default=1,
                help='1 channel input = IMG')
    p.add_argument('--num-class', type=int, default=2,
                help='number of classes in the dataset')
    p.add_argument('--win-size', type=int, nargs=3, default=[1, 64, 64],
                help='dimensions of the input to the net')
    p.add_argument('--hdim', type=int, default=96,
                help='dim of the FC layer before the latent space (mu and sigma)')
    p.add_argument('--latent-size', type=int, default=64,
                help='latent space dimension')
    p.add_argument('--unflatten-channel', type=int, default=2,
                help='number of channels before unflatten')
    p.add_argument('--dim-start-up-decoder', type=int, nargs=3, default=[5, 5, 5],
                help='dimensions in unflatten inside the decoder')
    
    # Discriminator loss
    p.add_argument('--disc-loss', action='store_true',
                help='enable discriminator loss (GAN-style)')
    p.add_argument('--lambda-gan', type=float, default=0.01,
                help='weight for the GAN loss component')

    # Hyperparameters
    p.add_argument('--recon-param', type=float, default=1.0,
                help='reconstruction parameter')
    p.add_argument('--beta', type=float, default=2.0,
                help='beta value of the beta-VAE')
    p.add_argument('--alpha', type=float, default=1.0,
                help='alpha multiplier for mlp_loss')
    p.add_argument('--gamma', type=float, default=10.0,
                help='gamma multiplier for AR-LOSS')
    p.add_argument('--factor', type=float, default=100.0,
                help='factor in AR-loss to scale the regularized latent dimension')
    p.add_argument('--perceptual-weight', type=float, default=0.0,
                help='weight for perceptual (LPIPS-style) loss. 0 = disabled, try 0.1-1.0')

    # Data loading and distributed training
    # Note: data path is hardcoded in src/data/dataset.py
    p.add_argument('--csv-path', type=str, required=True,
                help='path to the main CSV file')
    p.add_argument('--transform-cond-csv', type=str, required=True,
                help='path to the transform condition CSV file')
    p.add_argument('--saving-path', type=str, default="./results",
                help='the path where to save the model')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                help='the number of gradient accumulation steps')
    p.add_argument('--mixed-precision', type=str, default=None,
                help='the mixed precision type (fp16, bf16)')
    p.add_argument('--resume-training', action='store_true',
                help='resume training from checkpoint')
    p.add_argument('--use-iterable-dataset', action='store_true',
                help='use IterableDataset for proper multi-GPU/multi-worker sharding')
    p.add_argument('--use-wandb', action='store_true',
                help='enable wandb logging')
    p.add_argument('--wandb-project', type=str,
                help='wandb project name')
    p.add_argument('--wandb-runname', type=str,
                help='the run name for wandb')
    p.add_argument('--sequence-length', type=int, default=30,
                help='the total length of the sequence')
    p.add_argument('--normalization-type', type=str, default='absolute_max',
                choices=['absolute_max', 'mean_sigma_tanh', 'ionosphere_preprocess'],
                help='type of normalization to use')
    p.add_argument('--cartesian-transform', action='store_true',
                help='apply Cartesian transform to data')
    p.add_argument('--only-complete-sequences', action='store_true',
                help='only use sequences with no missing frames')
    p.add_argument('--skip-mode', type=str, default='full',
                choices=['full', 'none', 'dropout', 'weighted'],
                help='skip connection mode: full=use all, none=no skip connections, dropout=random dropout, weighted=scale by skip-scale')
    p.add_argument('--skip-dropout-prob', type=float, default=0.5,
                help='probability of dropping each skip connection (only used if skip-mode=dropout)')
    p.add_argument('--skip-scale', type=float, default=0.3,
                help='scale factor for skip connections (only used if skip-mode=weighted). 0=no skips, 1=full skips')

    args = p.parse_args()

    # n_filters need to stay as tuples (hard to pass via argparse)
    # n_filters_ENC = (8, 16, 32, 64, 16)
    n_filters_ENC = (16, 32, 64, 128, 256)
    n_filters_DEC = (64, 32, 16, 8, 4, 2)

    print(f"Parameter List: \n recon_param = {args.recon_param} ; beta = {args.beta} ; alpha = {args.alpha} ; gamma = {args.gamma} ; factor = {args.factor} \n epoch number is {args.epochs} and latent dimension is {args.latent_size} ")

    path_tosave_nets = args.saving_path + "/Experiment1_WHOLE_betaVAE_mlp_ar_woL1_VarianceMI_onlyVolume_woCapWoAnnealing" + "_epochs%d_batchs%d"%(args.epochs, args.batch_size) + "_beta%2f_alpha%2f_gamma%2f_factor%2f_reconparam%d_latentsize%d_batchsize%d"%(args.beta, args.alpha, args.gamma, args.factor, args.recon_param, args.latent_size, args.batch_size) + "/"
    # check_dir(path_tosave_nets)  # Uncomment when check_dir function is available

    # Initialize accelerator for distributed training
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    device = accelerator.device
    unwrap = accelerator.unwrap_model

    if accelerator.is_main_process:
        print(f'Process {accelerator.process_index} using device: {device}')
        print(f'World size: {accelerator.num_processes}')
        print(f'Batch size per GPU: {args.batch_size}')
        print(f'Global batch size: {args.batch_size * accelerator.num_processes}')
        print(f'Effective batch size: {args.batch_size * accelerator.num_processes * args.grad_accum_steps}')
        if args.mixed_precision:
            print(f'Mixed precision: {args.mixed_precision}')

    # Set random seed for reproducibility
    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes],
                             generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])


#########################################################  LOADING DATASET  ##########################################################

    # Set preprocess_config (None for absolute_max normalization)
    preprocess_config = None

    # Data loading with iterable dataset support
    if args.use_iterable_dataset:
        if accelerator.is_main_process:
            print("✅ Using IterableDataset for proper multi-GPU/multi-worker sharding!")

        train_ds, train_sampler, train_loader = get_sequence_data_objects_iterable(
            csv_path=args.csv_path,
            transform_cond_csv=args.transform_cond_csv,
            batch_size=args.batch_size,
            num_data_workers=args.num_workers,
            split='train',
            seed=args.seed,
            sequence_length=args.sequence_length,
            normalization_type=args.normalization_type,
            preprocess_config=preprocess_config,
            use_l1_conditions=True,
            min_center_distance=15,
            cartesian_transform=args.cartesian_transform,
            output_size=64,
            only_complete_sequences=args.only_complete_sequences,
            persistent_workers=True,
            prefetch_factor=4,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )

        val_ds, val_sampler, val_loader = get_sequence_data_objects_iterable(
            csv_path=args.csv_path,
            transform_cond_csv=args.transform_cond_csv,
            batch_size=args.batch_size,
            num_data_workers=args.num_workers,
            split='valid',
            seed=args.seed,
            sequence_length=args.sequence_length,
            normalization_type=args.normalization_type,
            preprocess_config=preprocess_config,
            use_l1_conditions=True,
            min_center_distance=30,
            cartesian_transform=args.cartesian_transform,
            output_size=64,
            only_complete_sequences=args.only_complete_sequences,
            persistent_workers=True,
            prefetch_factor=4,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )

        if accelerator.is_main_process:
            print(f'Train loader and Valid loader are up! Lengths: {len(train_loader)}, {len(val_loader)}')
            print("Note: Activity labels (0=low, 1=medium, 2=high) are computed by the dataset")
    else:
        # For now, raise an error if not using iterable dataset
        # You can add regular dataset loading here if needed
        raise NotImplementedError("Regular dataset loading not yet implemented. Please use --use-iterable-dataset flag.")

    # Call the training function
    model, train_losses, test_losses, acc_metrices, auc_metrices, train_loader, val_loader = main_train(
        args=args,
        accelerator=accelerator,
        n_filters_ENC=n_filters_ENC,
        n_filters_DEC=n_filters_DEC,
        train_loader=train_loader,
        val_loader=val_loader,
        path_tosave_nets=path_tosave_nets,
        device=device,
        unwrap=unwrap,
        is_L1=False,  # Set based on your needs
        use_AR_LOSS=True,  # Enable attribute regularization
        resume=args.resume_training
    )


def main_train(args, accelerator, n_filters_ENC, n_filters_DEC, train_loader, val_loader,
               path_tosave_nets, device, unwrap, is_L1=False, use_AR_LOSS=False, resume=False):
    """
    Main training function for VAE with ionosphere sequence data.

    The dataset returns 4 items per batch:
    - batch[0]: images (batch_size, sequence_length, channels, H, W)
    - batch[1]: L1 conditions (batch_size, sequence_length, 4)
    - batch[2]: epsilon values (batch_size, sequence_length, 1)
    - batch[3]: activity labels 0/1/2 (batch_size, sequence_length, 1)
    """

    train_losses =[]
    test_losses = []
    acc_metrices = []
    auc_metrices = []

    start_epoch = 0
    best_test_loss = np.finfo('f').max
    best_test_loss_epoch = -1
    best_metric = -1
    best_metric_epoch = -1
    best_auc = -1
    best_auc_epoch = -1

    # Initialize model
    # model = ConvVAE(image_channels=args.img_channels, h_dim=args.hdim,
    #                 latent_size=args.latent_size, n_filters_ENC=n_filters_ENC,
    #                 n_filters_DEC=n_filters_DEC)

    # model = UNetVAE(image_channels=args.img_channels, h_dim=args.hdim,
    #                 latent_size=args.latent_size, n_filters_ENC=n_filters_ENC,
    #                 n_filters_DEC=n_filters_DEC, img_size=64,
    #                 skip_mode=args.skip_mode, skip_dropout_prob=args.skip_dropout_prob,
    #                 skip_scale=args.skip_scale)

    model = ConvVAE2D(image_channels=args.img_channels, h_dim=args.hdim,                                                                                                                                                                                                                                                                                                                                                           
                    latent_size=args.latent_size, n_filters_ENC=n_filters_ENC,                                                                                                                                                                                                                                                                                                                                                   
                    n_filters_DEC=n_filters_DEC, img_size=64, num_classes=3)                                                                                                                                                                                                                                                                                                                                                     
    model.apply(initialize_weights_2d)  
    # model.apply(initialize_weights_unet)  # Initialize weights for UNetVAE

    # Initialize perceptual loss if enabled
    perceptual_loss_fn = None
    if args.perceptual_weight > 0:
        perceptual_loss_fn = PerceptualLoss(device=device)
        if accelerator.is_main_process:
            print(f"Perceptual loss enabled with weight: {args.perceptual_weight}")


    if accelerator.is_main_process:
        print(f"Skip connection mode: {args.skip_mode}")
        if args.skip_mode == 'dropout':
            print(f"Skip dropout probability: {args.skip_dropout_prob}")
        if args.skip_mode == 'weighted':
            print(f"Skip scale factor: {args.skip_scale}")

    # Print model parameter count
    if accelerator.is_main_process:
        count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # --- Discriminator (SD-style PatchGAN) ---
    if args.disc_loss:
        D = NLayerDiscriminator(input_nc=1).to(device)
        optimizer_D = optim.Adam(D.parameters(), lr=args.learning_rate)
        disc_start = 5000   # same idea as Stable Diffusion
        global_step = 0

    # Initialize wandb (only on main process)
    if accelerator.is_main_process and args.use_wandb:
        wandb.init(
            project=args.wandb_project if args.wandb_project else "attri-vae",
            entity="francescopio",
            name=args.wandb_runname,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "latent_size": args.latent_size,
                "h_dim": args.hdim,
                "beta": args.beta,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "factor": args.factor,
                "recon_param": args.recon_param,
                "perceptual_weight": args.perceptual_weight,
                "n_filters_ENC": n_filters_ENC,
                "n_filters_DEC": n_filters_DEC,
                "use_AR_LOSS": use_AR_LOSS,
            }
        )

    # Prepare model and optimizer with accelerator
    if args.use_iterable_dataset:
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Watch model with wandb (track gradients and parameters)
    if accelerator.is_main_process and args.use_wandb:
        wandb.watch(model, log="all", log_freq=1) 


    if resume:

        resume_path = path_tosave_nets + "checkpoint.pth"
        print('=> loading checkpoint %s' % resume)
        checkpoint = torch.load(resume_path, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        best_test_loss = checkpoint['best_test_loss']
        best_test_loss_epoch = checkpoint['best_test_loss_epoch']
        best_metric = checkpoint['best_metric']
        best_metric_epoch = checkpoint['best_metric_epoch']
        best_auc = checkpoint['best_auc']
        best_auc_epoch = checkpoint['best_auc_epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint %s' % resume)

    for epoch in range(start_epoch, args.epochs):
        # ============================================
        # TRAINING LOOP
        # ============================================
        model.train()
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_mlp_loss = 0
        epoch_kl_loss = 0
        epoch_attr_reg_loss = 0
        epoch_percep_loss = 0
        epoch_train_acc = 0
        epoch_train_auc = 0
        # epoch_disc_loss = 0
        num_train_batches = 0
        disc_train_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, smoothing=0.1,
                                                disable=not accelerator.is_main_process)):
            # Debug: Print first batch shape
            if accelerator.is_main_process and epoch == 0 and batch_idx == 0:
                tqdm.write(f"\nFIRST BATCH SHAPES:")
                tqdm.write(f"  batch[0].shape (images) = {batch[0].shape}")
                tqdm.write(f"  batch[1].shape (conditions) = {batch[1].shape}")
                tqdm.write(f"  batch[2].shape (epsilon) = {batch[2].shape}")
                tqdm.write(f"  batch[3].shape (labels) = {batch[3].shape}")
                tqdm.write(f"  Batch size in data: {batch[0].shape[0]}")
                tqdm.write(f"")

            with accelerator.accumulate(model):
                # Data transfer to GPU
                inpt = batch[0].contiguous().float().to(device, non_blocking=True)
                inpt = inpt.squeeze(2)  # shape: (batch_size, sequence_length, H, W)
                cond_label = batch[1].to(device, non_blocking=True)
                epsilon_seq = batch[2].to(device, non_blocking=True)  # (batch_size, sequence_length, 1)
                label_seq = batch[3].to(device, non_blocking=True)    # (batch_size, sequence_length, 1) - activity labels

                # For VAE: extract data from the center frame
                # center_idx = args.sequence_length // 2
                data = inpt # (batch_size, H, W)
                # data = data.unsqueeze(1)  # (batch_size, 1, H, W) - add channel dimension

                # Use center frame conditions and labels
                rad_ = cond_label.squeeze(1)  # (batch_size, num_conditions) - L1 conditions for attr reg
                label = label_seq.squeeze(1)   # (batch_size, 1) - activity label (0, 1, or 2)

                # Forward pass
                recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data)
                # Compute losses
                recon_loss = reconstruction_loss(recon_batch, data, args.recon_param, dist='gaussian')
                mlp_loss = mlp_loss_function(label, out_mlp, args.alpha)
                kl_loss1, kl_loss2 = KL_loss(mu, logvar, z_dist, prior_dist, args.beta, c=0.0)
                loss = recon_loss + mlp_loss + kl_loss2

                # Optional: Attribute regularization loss
                attr_reg_loss = torch.tensor(0.0).to(device)
                if use_AR_LOSS:
                    attr_reg_loss = reg_loss(z_tilde, rad_, len(data), gamma=args.gamma, factor=args.factor)
                    loss += attr_reg_loss

                # Optional: Perceptual loss for high-frequency details
                percep_loss = torch.tensor(0.0).to(device)
                if perceptual_loss_fn is not None:
                    percep_loss = args.perceptual_weight * perceptual_loss_fn(recon_batch, data)
                    loss += percep_loss

                # Optional: GAN loss
                g_adv_loss = torch.tensor(0.0).to(device)
                if args.disc_loss:
                    lambda_gan = args.lambda_gan  # keep small
                    g_adv_loss = -D(recon_batch).mean()
                    loss += lambda_gan * g_adv_loss


                # Optional: L1 weight regularization
                if is_L1:
                    l1_crit = torch.nn.L1Loss(reduction="sum")
                    weight_reg_loss = 0
                    for param in model.parameters():
                        weight_reg_loss += l1_crit(param, target=torch.zeros_like(param))
                    fctr = 0.00005
                    loss += fctr * weight_reg_loss

                # Backward pass
                accelerator.backward(loss)

                # Optimizer step (only when gradients are synced)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()       
                # --- Discriminator step ---
                if args.disc_loss:
                    if global_step > disc_start:
                        optimizer_D.zero_grad()

                        logits_real = D(data.detach())
                        logits_fake = D(recon_batch.detach())

                        d_loss = hinge_d_loss(logits_real, logits_fake)

                        accelerator.backward(d_loss)
                        optimizer_D.step()                                                                                                                                                                                                                                                                                                                                                                                                             

                # Compute metrics
                accuracy, roc = mean_accuracy(label, out_mlp)

                # Accumulate losses and metrics
                epoch_train_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_mlp_loss += mlp_loss.item()
                epoch_kl_loss += kl_loss2.item()
                epoch_attr_reg_loss += attr_reg_loss.item()
                epoch_percep_loss += percep_loss.item()
                # epoch_disc_loss += disc_train_loss.item()
                epoch_train_acc += accuracy
                epoch_train_auc += roc
                num_train_batches += 1

                if args.disc_loss:
                    global_step += 1

        # Average training metrics
        epoch_train_loss /= num_train_batches
        epoch_recon_loss /= num_train_batches
        epoch_mlp_loss /= num_train_batches
        epoch_kl_loss /= num_train_batches
        epoch_attr_reg_loss /= num_train_batches
        epoch_percep_loss /= num_train_batches
        # epoch_disc_loss /= num_train_batches
        epoch_train_acc /= num_train_batches
        epoch_train_auc /= num_train_batches

        # Gather training metrics across GPUs
        train_loss_tensor = torch.tensor(epoch_train_loss, device=device)
        gathered_train_loss = accelerator.gather(train_loss_tensor)
        if accelerator.is_main_process:
            train_loss = gathered_train_loss.mean().item()
        else:
            train_loss = epoch_train_loss

        # ============================================
        # VALIDATION LOOP
        # ============================================
        model.eval()
        epoch_val_loss = 0
        epoch_val_recon_loss = 0
        epoch_val_mlp_loss = 0
        epoch_val_kl_loss = 0
        epoch_val_attr_reg_loss = 0
        epoch_val_percep_loss = 0
        # epoch_val_disc_loss = 0
        epoch_val_acc = 0
        epoch_val_auc = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation",
                                                    disable=not accelerator.is_main_process)):
                # Data transfer to GPU
                inpt = batch[0].contiguous().float().to(device, non_blocking=True)
                inpt = inpt.squeeze(2)  # shape: (batch_size, sequence_length, H, W)
                cond_label = batch[1].to(device, non_blocking=True)
                epsilon_seq = batch[2].to(device, non_blocking=True)
                label_seq = batch[3].to(device, non_blocking=True)    # activity labels

                # For VAE: extract center frame
                center_idx = args.sequence_length // 2
                data_test = inpt[:, center_idx, :, :]  # (batch_size, H, W)
                data_test = data_test.unsqueeze(1)  # (batch_size, 1, H, W)

                # Use center frame conditions and labels
                rad_ = cond_label[:, center_idx, :]  # L1 conditions for attr reg
                label = label_seq[:, center_idx, :]  # activity label (0, 1, or 2)

                # Forward pass
                recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data_test)

                # Compute losses 
                recon_loss = reconstruction_loss(recon_batch, data_test, args.recon_param, dist='gaussian')
                kl_loss1, kl_loss2 = KL_loss(mu, logvar, z_dist, prior_dist, args.beta, c=0.0)
                mlp_loss = mlp_loss_function(label, out_mlp, args.alpha)
                loss_ = recon_loss + mlp_loss + kl_loss2

                # Optional: Attribute regularization loss
                attr_reg_loss = torch.tensor(0.0).to(device)
                if use_AR_LOSS:
                    attr_reg_loss = reg_loss(z_tilde, rad_, len(data_test), gamma=args.gamma, factor=args.factor)
                    loss_ += attr_reg_loss

                # Optional: Perceptual loss
                percep_loss = torch.tensor(0.0).to(device)
                if perceptual_loss_fn is not None:
                    percep_loss = args.perceptual_weight * perceptual_loss_fn(recon_batch, data_test)
                    loss_ += percep_loss

                # Optional: GAN loss
                g_adv_loss = torch.tensor(0.0).to(device)
                if args.disc_loss:
                    lambda_gan = args.lambda_gan  # keep small
                    g_adv_loss = -D(recon_batch).mean()
                    loss_ += lambda_gan * g_adv_loss

                # Compute metrics
                accuracy, roc = mean_accuracy(label, out_mlp)

                # Accumulate validation metrics
                epoch_val_loss += loss_.item()
                epoch_val_recon_loss += recon_loss.item()
                epoch_val_mlp_loss += mlp_loss.item()
                epoch_val_kl_loss += kl_loss2.item()
                epoch_val_attr_reg_loss += attr_reg_loss.item()
                epoch_val_percep_loss += percep_loss.item()
                # epoch_val_disc_loss += disc_train_loss.item()
                epoch_val_acc += accuracy
                epoch_val_auc += roc
                num_val_batches += 1

        # Average validation metrics
        epoch_val_loss /= num_val_batches
        epoch_val_recon_loss /= num_val_batches
        epoch_val_mlp_loss /= num_val_batches
        epoch_val_kl_loss /= num_val_batches
        epoch_val_attr_reg_loss /= num_val_batches
        epoch_val_percep_loss /= num_val_batches
        # epoch_val_disc_loss /= num_val_batches
        epoch_val_acc /= num_val_batches
        epoch_val_auc /= num_val_batches

        # Gather validation metrics across GPUs
        val_loss_tensor = torch.tensor(epoch_val_loss, device=device)
        gathered_val_loss = accelerator.gather(val_loss_tensor)
        if accelerator.is_main_process:
            test_loss = gathered_val_loss.mean().item()
            acc_metric = epoch_val_acc
            auc_metric = epoch_val_auc
        else:
            test_loss = epoch_val_loss
            acc_metric = epoch_val_acc
            auc_metric = epoch_val_auc

        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        acc_metrices.append(acc_metric)
        auc_metrices.append(auc_metric)

        # Print epoch summary (only on main process)
        if accelerator.is_main_process:
            print('='*80)
            print(f'Epoch [{epoch + 1}/{args.epochs}]')
            print(f'  Training   - Loss: {train_loss:.4f}, Acc: {epoch_train_acc:.4f}, AUC: {epoch_train_auc:.4f}')
            print(f'             - Recon: {epoch_recon_loss:.4f}, MLP: {epoch_mlp_loss:.4f}, KL: {epoch_kl_loss:.4f}, AR: {epoch_attr_reg_loss:.4f}, Percep: {epoch_percep_loss:.4f}')
            print(f'  Validation - Loss: {test_loss:.4f}, Acc: {acc_metric:.4f}, AUC: {auc_metric:.4f}')
            print(f'             - Recon: {epoch_val_recon_loss:.4f}, MLP: {epoch_val_mlp_loss:.4f}, KL: {epoch_val_kl_loss:.4f}, AR: {epoch_val_attr_reg_loss:.4f}, Percep: {epoch_val_percep_loss:.4f}')
            print('='*80)

            # Log to wandb
            if args.use_wandb:
                wandb_log = {
                    "epoch": epoch + 1,
                    # Training metrics
                    "train/loss": train_loss,
                    "train/recon_loss": epoch_recon_loss,
                    "train/mlp_loss": epoch_mlp_loss,
                    "train/kl_loss": epoch_kl_loss,
                    "train/attr_reg_loss": epoch_attr_reg_loss,
                    "train/percep_loss": epoch_percep_loss,
                    "train/accuracy": epoch_train_acc,
                    "train/auc": epoch_train_auc,
                    # "train/disc_loss": epoch_disc_loss,
                    # Validation metrics
                    "val/loss": test_loss,
                    "val/recon_loss": epoch_val_recon_loss,
                    "val/mlp_loss": epoch_val_mlp_loss,
                    "val/kl_loss": epoch_val_kl_loss,
                    "val/attr_reg_loss": epoch_val_attr_reg_loss,
                    "val/percep_loss": epoch_val_percep_loss,
                    "val/accuracy": acc_metric,
                    "val/auc": auc_metric,
                    # "val/disc_loss": epoch_val_disc_loss,
                }
                wandb.log(wandb_log)

                # Log reconstruction visualization every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    fig = visualize_reconstruction(
                        model=unwrap(model),
                        data=data_test,
                        device=device,
                        epoch=epoch + 1,
                        save_path=path_tosave_nets,
                        num_samples=1
                    )
                    wandb.log({"reconstructions": wandb.Image(fig)})
                    plt.close(fig)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'best_test_loss': best_test_loss,
                'best_test_loss_epoch': best_test_loss_epoch,
                'best_metric': best_metric,
                'best_metric_epoch': best_metric_epoch,
                'best_auc': best_auc,
                'best_auc_epoch': best_auc_epoch,
                'state_dict': accelerator.get_state_dict(model),
                'optimizer': optimizer.state_dict()
            }
            save_ckp(checkpoint, path_tosave_nets)

            # Save best models
            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                print(f"✅ Best accuracy achieved: {best_metric:.4f} in epoch {best_metric_epoch}")
                torch.save(accelerator.get_state_dict(model), path_tosave_nets + "best_metric_model.pth")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_loss_epoch = epoch + 1
                print(f"✅ Best loss achieved: {best_test_loss:.4f} in epoch {best_test_loss_epoch}")
                torch.save(accelerator.get_state_dict(model), path_tosave_nets + "best_test_loss_model.pth")

            if auc_metric > best_auc:
                best_auc = auc_metric
                best_auc_epoch = epoch + 1
                print(f"✅ Best AUC achieved: {best_auc:.4f} in epoch {best_auc_epoch}")
                torch.save(accelerator.get_state_dict(model), path_tosave_nets + "best_AUC_model.pth")

        # Wait for all processes to sync
        accelerator.wait_for_everyone()

    print("OUTCOME")
    print(f"Best accuracy of {best_metric} was achieved in epoch {best_metric_epoch}")
    print(f"Best loss of {best_test_loss} was achieved in epoch {best_test_loss_epoch}")
    print(f"Best AUC of {best_auc} was achieved in epoch {best_auc_epoch}")

    # Close wandb run
    if accelerator.is_main_process and args.use_wandb:
        wandb.finish()

    return model, train_losses, test_losses, acc_metrices, auc_metrices, train_loader, val_loader


if __name__ == "__main__":
    main()

