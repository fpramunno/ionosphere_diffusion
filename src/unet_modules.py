# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:28:21 2023

@author: pio-r
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

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
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Normal convolution block, with 2d convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer
    
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
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
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # projection
        return x + emb


class Up(nn.Module):
    """
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
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
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class PaletteModelV2(nn.Module):
    def __init__(self, c_in=1, c_out=1, image_size=64, time_dim=256, device='cuda', latent=False, true_img_size=64, num_classes=None):
        super(PaletteModelV2, self).__init__()

        # Encoder
        self.true_img_size = true_img_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device
        self.inc = DoubleConv(c_in, self.image_size) # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.image_size, self.image_size*2) # input and output channels
        # self.sa1 = SelfAttention(self.image_size*2,int( self.true_img_size/2)) # 1st is channel dim, 2nd current image resolution
        self.down2 = Down(self.image_size*2, self.image_size*4)
        # self.sa2 = SelfAttention(self.image_size*4, int(self.true_img_size/4))
        self.down3 = Down(self.image_size*4, self.image_size*4)
        # self.sa3 = SelfAttention(self.image_size*4, int(self.true_img_size/8))

        # Bootleneck
        self.bot1 = DoubleConv(self.image_size*4, self.image_size*8)
        self.bot2 = DoubleConv(self.image_size*8, self.image_size*8)
        self.bot3 = DoubleConv(self.image_size*8, self.image_size*4)

        # Decoder: reverse of encoder
        self.up1 = Up(self.image_size*8, self.image_size*2)
        # self.sa4 = SelfAttention(self.image_size*2, int(self.true_img_size/4))
        self.up2 = Up(self.image_size*4, self.image_size)
        # self.sa5 = SelfAttention(self.image_size, int(self.true_img_size/2))
        self.up3 = Up(self.image_size*2, self.image_size)
        # self.sa6 = SelfAttention(self.image_size, self.true_img_size)
        self.outc = nn.Conv2d(self.image_size, c_out, kernel_size=1) # projecting back to the output channel dimensions

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        if latent == True:
            self.latent = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256)).to(device)

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

    def forward(self, x, lab, t):
        # Pass the source image through the encoder network
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Encoding timesteps is HERE, we provide the dimension we want to encode


        if lab is not None:
            t += self.label_emb(lab)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t) # We note that upsampling box that in the skip connections from encoder
        # x = self.sa4(x)
        x = self.up2(x, x2, t)
        # x = self.sa5(x)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)

        return output


class DisentangledAutoEncoder(nn.Module):
    """
    Disentangled Autoencoder for unsupervised learning of interpretable factors.

    Encodes a single frame into disentangled factors:
    - content: What the structure looks like (time-invariant)
    - rotation: Rotation angle (smooth temporal changes)
    - intensity: Brightness/magnitude (smooth temporal changes)
    - sign: Polarity flip (sparse temporal changes)
    - slow: Unknown smooth processes
    - fast: Unknown fast/abrupt processes
    """
    def __init__(self,
                 c_in=1,
                 c_out=1,
                 image_size=64,
                 true_img_size=64,
                 z_content_dim=32,
                 z_slow_dim=8,
                 z_fast_dim=4,
                 device='cuda'):
        super(DisentangledAutoEncoder, self).__init__()

        self.image_size = image_size
        self.true_img_size = true_img_size
        self.device = device
        self.c_out = c_out

        # ============ ENCODER ============
        # Reuse the encoder structure from PaletteModelV2
        self.inc = DoubleConv(c_in, image_size)
        self.down1 = DoubleConv(image_size, image_size*2)
        self.sa1 = SelfAttention(image_size*2, true_img_size // 2)  # After down1
        self.down2 = DoubleConv(image_size*2, image_size*4)
        self.sa2 = SelfAttention(image_size*4, true_img_size // 4)  # After down2
        self.down3 = DoubleConv(image_size*4, image_size*4)
        self.sa3 = SelfAttention(image_size*4, true_img_size // 8)  # After down3

        # Bottleneck
        self.bot1 = DoubleConv(image_size*4, image_size*8)
        self.bot2 = DoubleConv(image_size*8, image_size*8)
        self.sa_bot = SelfAttention(image_size*8, true_img_size // 8)  # After bottleneck

        # Global pooling for bottleneck features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        bottleneck_dim = image_size * 8

        # ============ DISENTANGLEMENT HEADS ============
        # Content: what the structure looks like (time-invariant)
        self.head_content = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim // 2, z_content_dim)
        )

        # Rotation: (sin θ, cos θ) for cyclic representation
        self.head_rotation = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Will be normalized to unit circle
        )

        # Intensity: scalar brightness
        self.head_intensity = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Sign: polarity (+1 or -1)
        self.head_sign = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Unknown slow factors (smooth temporal changes)
        self.head_slow = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_slow_dim)
        )

        # Unknown fast factors (abrupt temporal changes)
        self.head_fast = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_fast_dim)
        )

        # ============ DECODER ============
        # Content decoder: generates canonical frame from z_content only
        self.content_decoder_fc = nn.Sequential(
            nn.Linear(z_content_dim, 256),
            nn.ReLU(),
            nn.Linear(256, image_size * 4 * (true_img_size // 8) * (true_img_size // 8))
        )

        # Upsampling path
        self.up1 = nn.Sequential(
            nn.Conv2d(image_size*4, image_size*2, kernel_size=3, padding=1),
            nn.GroupNorm(1, image_size*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.sa_up1 = SelfAttention(image_size*2, true_img_size // 4)  # After up1

        self.up2 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size, kernel_size=3, padding=1),
            nn.GroupNorm(1, image_size),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.sa_up2 = SelfAttention(image_size, true_img_size // 2)  # After up2

        self.up3 = nn.Sequential(
            nn.Conv2d(image_size, image_size, kernel_size=3, padding=1),
            nn.GroupNorm(1, image_size),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.sa_up3 = SelfAttention(image_size, true_img_size)  # After up3

        self.outc = nn.Conv2d(image_size, c_out, kernel_size=1)

        # Modulation network for unknown factors
        self.modulation_net = nn.Sequential(
            nn.Conv2d(c_out + z_slow_dim + z_fast_dim, image_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(image_size, c_out, kernel_size=3, padding=1)
        )

    def encode(self, x):
        """
        Encode a single frame into disentangled factors.

        Args:
            x: [B, C, H, W] - single frame

        Returns:
            dict with keys: 'content', 'rotation', 'intensity', 'sign', 'slow', 'fast'
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = F.max_pool2d(x1, 2)
        x2 = self.down1(x2)
        x2 = self.sa1(x2)  # Self-attention after down1

        x3 = F.max_pool2d(x2, 2)
        x3 = self.down2(x3)
        x3 = self.sa2(x3)  # Self-attention after down2

        x4 = F.max_pool2d(x3, 2)
        x4 = self.down3(x4)
        x4 = self.sa3(x4)  # Self-attention after down3

        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.sa_bot(x4)  # Self-attention after bottleneck

        # Global pooling
        features = self.global_pool(x4).squeeze(-1).squeeze(-1)  # [B, bottleneck_dim]

        # Disentanglement heads
        z_content = self.head_content(features)
        z_rotation_raw = self.head_rotation(features)
        z_rotation = F.normalize(z_rotation_raw, dim=-1)  # Normalize to unit circle
        z_intensity = self.head_intensity(features)
        z_sign = torch.tanh(self.head_sign(features))  # Range [-1, 1]
        z_slow = self.head_slow(features)
        z_fast = self.head_fast(features)

        return {
            'content': z_content,
            'rotation': z_rotation,
            'intensity': z_intensity,
            'sign': z_sign,
            'slow': z_slow,
            'fast': z_fast
        }

    def apply_rotation(self, image, rotation):
        """
        Apply rotation using affine transformation.

        Args:
            image: [B, C, H, W]
            rotation: [B, 2] - (sin θ, cos θ)
        """
        B = image.shape[0]

        sin_theta = rotation[:, 0]
        cos_theta = rotation[:, 1]

        # Create rotation matrix [B, 2, 3]
        theta = torch.zeros(B, 2, 3, device=image.device)
        theta[:, 0, 0] = cos_theta
        theta[:, 0, 1] = -sin_theta
        theta[:, 1, 0] = sin_theta
        theta[:, 1, 1] = cos_theta

        # Apply spatial transformation
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        rotated = F.grid_sample(image, grid, align_corners=False)

        return rotated

    def decode(self, z_dict):
        """
        Decode disentangled factors back to frame.

        Args:
            z_dict: dict with keys 'content', 'rotation', 'intensity', 'sign', 'slow', 'fast'

        Returns:
            [B, C, H, W] - reconstructed frame
        """
        B = z_dict['content'].shape[0]

        # 1. Decode content to canonical frame
        h = self.content_decoder_fc(z_dict['content'])
        spatial_size = self.true_img_size // 8
        h = h.view(B, self.image_size * 4, spatial_size, spatial_size)

        canonical = self.up1(h)
        canonical = self.sa_up1(canonical)  # Self-attention after up1
        canonical = self.up2(canonical)
        canonical = self.sa_up2(canonical)  # Self-attention after up2
        canonical = self.up3(canonical)
        canonical = self.sa_up3(canonical)  # Self-attention after up3
        canonical = self.outc(canonical)

        # 2. Apply geometric rotation
        rotated = self.apply_rotation(canonical, z_dict['rotation'])

        # 3. Apply intensity modulation
        intensity = z_dict['intensity'].view(B, 1, 1, 1)
        scaled = rotated * F.softplus(intensity)  # Ensure positive

        # 4. Apply sign flip
        sign = z_dict['sign'].view(B, 1, 1, 1)
        flipped = scaled * sign

        # 5. Apply unknown factors modulation
        # Broadcast slow and fast factors spatially
        H, W = flipped.shape[2], flipped.shape[3]
        z_slow_spatial = z_dict['slow'].view(B, -1, 1, 1).expand(-1, -1, H, W)
        z_fast_spatial = z_dict['fast'].view(B, -1, 1, 1).expand(-1, -1, H, W)

        # Concatenate with image and apply modulation
        modulation_input = torch.cat([flipped, z_slow_spatial, z_fast_spatial], dim=1)
        modulated = self.modulation_net(modulation_input)

        final = flipped + modulated  # Residual connection

        return final

    def forward(self, x):
        """
        Full forward pass: encode then decode.

        Args:
            x: [B, C, H, W] - single frame

        Returns:
            recon: [B, C, H, W] - reconstructed frame
            z_dict: dict of disentangled factors
        """
        z_dict = self.encode(x)
        recon = self.decode(z_dict)
        return recon, z_dict