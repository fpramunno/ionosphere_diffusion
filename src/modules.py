# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:28:21 2023

@author: pio-r
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from IPython import embed
from torch import Tensor
import numpy as np


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
):
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        spatial: The number of spatial dimensions (1, 2, or 3).
        identity_init: Initialize the convolution as a (pseudo-)identity.
        kwargs: Keyword arguments passed to Conv layer.
    """
    CONVS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError(f"spatial={spatial} not supported")

    conv = Conv(in_channels, out_channels, **kwargs)

    if identity_init:
        kernel_size = conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]

        eye = torch.zeros_like(conv.weight.data)

        for i in range(out_channels):
            eye[(i, i % in_channels, *kernel_center)] = 1

        conv.weight.data.mul_(1e-2)
        conv.weight.data.add_(eye)

    return conv


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    """
    Pre Layer norm  -> multi-headed tension -> skip connections -> pass it to
    the feed forward layer (layer-norm -> 2 multiheadattention)

    Handles both 4D (B, C, H, W) and 5D (B, C, T, H, W) inputs.
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        from einops import rearrange

        # Flatten all spatial/temporal dimensions into sequence dimension
        # Works for both (B, C, H, W) and (B, C, T, H, W)
        original_shape = x.shape
        y = rearrange(x, "B C ... -> B (...) C")

        y_ln = self.ln(y)
        attention_value, _ = self.mha(y_ln, y_ln, y_ln)
        attention_value = attention_value + y
        attention_value = self.ff_self(attention_value) + attention_value

        # Reshape back to original shape
        attention_value = rearrange(attention_value, "B L C -> B C L").reshape(original_shape)
        return attention_value


class DoubleConv(nn.Module):
    """
    Normal convolution block, with Nd convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, spatial=2):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ConvNd(in_channels, mid_channels, spatial=spatial, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            ConvNd(mid_channels, out_channels, spatial=spatial, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            # return F.gelu(x + self.double_conv(x))
            return nn.GELU()(x + self.double_conv(x))
        else:
            return self.double_conv(x).requires_grad_(True)


class Down(nn.Module):
    """
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer

    """
    def __init__(self, in_channels, out_channels, emb_dim=256, spatial=2):
        super().__init__()
        self.spatial = spatial

        # Select appropriate MaxPool based on spatial dimensions
        if spatial == 2:
            maxpool = nn.MaxPool2d(2)
        elif spatial == 3:
            # For 3D, only pool spatial dimensions (H, W), not temporal (T)
            # kernel_size=(1, 2, 2) means: no pooling on T, 2x2 pooling on H and W
            maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        else:
            raise NotImplementedError(f"spatial={spatial} not supported")

        self.maxpool_conv = nn.Sequential(
            maxpool,
            DoubleConv(in_channels, in_channels, residual=True, spatial=spatial),
            DoubleConv(in_channels, out_channels, spatial=spatial),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear( # linear projection to bring the time embedding to the proper dimension
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # Dynamically handle the spatial dimensions
        if self.spatial == 2:
            emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        elif self.spatial == 3:
            emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    """
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256, spatial=2):
        super().__init__()
        self.spatial = spatial

        # Select appropriate upsampling mode based on spatial dimensions
        if spatial == 2:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif spatial == 3:
            # For 3D, only upsample spatial dimensions (H, W), not temporal (T)
            # scale_factor=(1, 2, 2) means: no upsampling on T, 2x upsampling on H and W
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        else:
            raise NotImplementedError(f"spatial={spatial} not supported")

        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, spatial=spatial),
            DoubleConv(in_channels, out_channels, in_channels // 2, spatial=spatial),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        # Dynamically handle the spatial dimensions
        if self.spatial == 2:
            emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        elif self.spatial == 3:
            emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb

class SineEncoding(nn.Module):
    r"""Creates a sinusoidal positional encoding.

    .. math::
        e_{2i} & = \sin \left( x \times \omega^\frac{-2i}{D} \right) \\
        e_{2i+1} & = \cos \left( x \times \omega^\frac{-2i}{D} \right)

    References:
        | Attention Is All You Need (Vaswani et al., 2017)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        features: The number of embedding features :math:`D`. Must be even.
        omega: The maximum frequency :math:`\omega`.
    """

    def __init__(self, features: int, omega: float = 1e3):
        super().__init__()

        assert features % 2 == 0

        freqs = np.linspace(0, 1, features // 2)
        freqs = omega ** (-freqs)

        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The position :math:`x`, with shape :math:`(*)`.

        Returns:
            The embedding vector :math:`e`, with shape :math:`(*, D)`.
        """

        x = x.unsqueeze(dim=-1)

        return torch.cat(
            (
                torch.sin(x * self.freqs),
                torch.cos(x * self.freqs),
            ),
            dim=-1,
        )

class PaletteModelV2(nn.Module):
    def __init__(
        self,
        c_in=1,
        c_out=1,
        image_size=64,
        time_dim=256,
        device='cuda',
        latent=False,
        true_img_size=64,
        num_classes=None,
        channel_mapping_cond=30,
        spatial=3,
        channel_mults=(1, 2, 4, 4),  # Channel multipliers at each depth
        num_res_blocks=2,  # Number of residual blocks per depth
        attention_resolutions=(16,),  # Resolutions where attention is applied
        num_bottleneck_blocks=3,  # Number of bottleneck blocks
    ):
        super(PaletteModelV2, self).__init__()

        # Store configuration
        self.true_img_size = true_img_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device
        self.channel_mapping_cond = channel_mapping_cond
        self.spatial = spatial
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks

        # Calculate channel dimensions at each depth
        channels = [image_size * mult for mult in channel_mults]

        # Initial convolution
        self.inc = DoubleConv(c_in, channels[0], spatial=spatial)

        # Encoder (downsampling path)
        self.downs = nn.ModuleList()
        self.down_attns = nn.ModuleList()

        for i in range(len(channels) - 1):
            # Add down block
            self.downs.append(Down(channels[i], channels[i+1], spatial=spatial))

            # Calculate current resolution after this downsampling
            current_res = true_img_size // (2 ** (i + 1))

            # Add attention if at specified resolution
            if current_res in attention_resolutions:
                self.down_attns.append(SelfAttention(channels[i+1], current_res))
            else:
                self.down_attns.append(nn.Identity())

        # Bottleneck
        bottleneck_ch = channels[-1]
        self.bottleneck = nn.ModuleList()
        for i in range(num_bottleneck_blocks):
            if i == 0:
                self.bottleneck.append(DoubleConv(bottleneck_ch, bottleneck_ch * 2, spatial=spatial))
            elif i == num_bottleneck_blocks - 1:
                self.bottleneck.append(DoubleConv(bottleneck_ch * 2, bottleneck_ch, spatial=spatial))
            else:
                self.bottleneck.append(DoubleConv(bottleneck_ch * 2, bottleneck_ch * 2, spatial=spatial))

        # Decoder (upsampling path)
        self.ups = nn.ModuleList()
        self.up_attns = nn.ModuleList()

        for i in reversed(range(len(channels) - 1)):
            # For Up blocks, input is concatenated features from:
            # - upsampled x: channels[i+1]
            # - skip connection: channels[i]
            # Total input: channels[i+1] + channels[i]
            self.ups.append(Up(channels[i+1] + channels[i], channels[i], spatial=spatial))

            # Calculate current resolution after this upsampling
            current_res = true_img_size // (2 ** i)

            # Add attention if at specified resolution
            if current_res in attention_resolutions:
                self.up_attns.append(SelfAttention(channels[i], current_res))
            else:
                self.up_attns.append(nn.Identity())

        # Output convolution
        self.outc = ConvNd(channels[0], c_out, spatial=spatial, kernel_size=1)

        self.mapping_cond = nn.Sequential(
            nn.Linear(self.channel_mapping_cond * 4, self.time_dim),  # Flatten spatial dimensions and project to mod_features BEFORE 16 NOW 30
            nn.SiLU(),
            nn.LayerNorm(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.timestep_embed = nn.Sequential(
            SineEncoding(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
  
  
    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 /  (
            10000 
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc 
    

    def forward(self, x, sigma, cond, mapping_cond):
        # Pass the source image through the encoder network
        c_noise = sigma.log() / 4
        timestep_embed = self.timestep_embed(c_noise.unsqueeze(-1))
        # Squeeze out the extra dimension from SineEncoding: (B, 1, D) -> (B, D)
        timestep_embed = timestep_embed.squeeze(1)
        
        # embed()

        if mapping_cond is not None:
            mapping_cond_embed = self.mapping_cond(mapping_cond.flatten(-2)) 
            timestep_embed += mapping_cond_embed

        

        batch_size = x.shape[0]
        num_pred_frames = x.shape[1]
        num_cond_frames = cond.shape[1] if cond is not None else 0

        # # Concatenate the source image and reference image
        # x = torch.cat([x, cond], dim=1)

        temporal_positions = torch.arange(-num_cond_frames, num_cond_frames,
                                         device=x.device, dtype=x.dtype)
        
        total_frames = x.shape[1]

        # Normalize by max horizon (max absolute temporal distance)
        # This gives range roughly [-1, 1] centered at present (t=0)
        max_horizon = num_cond_frames
        temporal_positions_normalized = temporal_positions / max_horizon

        # Add temporal coordinate as explicit input channel
        # Following solar project approach: use (B, C, T, H, W) format
        time_coord = temporal_positions_normalized.view(1, -1, 1, 1).expand(batch_size, total_frames, x.shape[-2], x.shape[-1])

        l1_cond = mapping_cond.unsqueeze(-1).unsqueeze(-1)  # (batch, total_frames, 4, 1, 1)
        l1_cond = l1_cond.expand(batch_size, total_frames, 4, x.shape[-2], x.shape[-1])

        # Rearrange to (B, C, T, H, W) format like solar project
        input = x.unsqueeze(1)  # (B, 1, T, H, W)
        time_coord = time_coord.unsqueeze(1)  # (B, 1, T, H, W)
        l1_cond = l1_cond.permute(0, 2, 1, 3, 4)  # (B, T, 4, H, W) -> (B, 4, T, H, W)

        # Concatenate along channel dimension (dim=1)
        x = torch.cat([input, time_coord, l1_cond], dim=1)  # (B, 6, T, H, W)

        # Initial convolution
        x = self.inc(x)

        # Encoder path with skip connections
        skip_connections = [x]
        for down, attn in zip(self.downs, self.down_attns):
            x = down(x, timestep_embed)
            x = attn(x)
            skip_connections.append(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)

        # Decoder path with skip connections
        skip_connections = skip_connections[:-1]  # Remove last (we're already there)
        for up, attn in zip(self.ups, self.up_attns):
            skip = skip_connections.pop()
            x = up(x, skip, timestep_embed)
            x = attn(x)

        # Output convolution
        output = self.outc(x)
        output = output.squeeze(1)  # Remove channel dim if needed: (B, 1, T, H, W) -> (B, T, H, W)

        return output.requires_grad_(True)

    def param_groups(self, base_lr=2e-4):
        """
        Separate parameters into groups for weight decay.
        Applies weight decay to weights but not to biases and normalization layers.
        """
        wd_params, no_wd_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".weight") and "norm" not in name.lower() and "bias" not in name.lower():
                wd_params.append(param)
            else:
                no_wd_params.append(param)
        return [
            {"params": wd_params, "lr": base_lr},
            {"params": no_wd_params, "lr": base_lr, "weight_decay": 0.0},
        ]
    