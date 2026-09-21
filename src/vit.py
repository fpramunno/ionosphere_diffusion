r"""Vision Transformer (ViT) building blocks.

References:
    | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2021)
    | https://arxiv.org/abs/2010.11929

    | Scalable Diffusion Models with Transformers (Peebles et al., 2022)
    | https://arxiv.org/abs/2212.09748
"""

__all__ = [
    "ViTBlock",
    "ViT",
    "PhysicsInformedConditioningModule",
    "MultiScaleTemporalEncoder",
]

import functools
import math
import torch
import torch.nn as nn
import numpy as np
import warnings
from einops import rearrange
from torch.nn import functional as F


# import xformers.components.attention.core as xfa
# import xformers.sparse as xfs
    
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Hashable, Optional, Sequence, Tuple, Union
from torch.utils.checkpoint import checkpoint

# debug
from IPython import embed


class PhysicsInformedConditioningModule(nn.Module):
    """
    Physics-aware embedding for solar wind parameters with:
    - Factorized representation (energy, geometry, dynamics)
    - Spherical harmonic encoding for directional components (By, Bz)
    - Derivative-aware temporal encoding

    Input order: [bx, by, bz, vwind]
    """

    def __init__(
        self,
        mod_features: int,
        context_channels: Optional[int] = None,
        spherical_order: int = 4,
    ):
        super().__init__()

        self.mod_features = mod_features
        self.context_channels = context_channels
        self.spherical_order = spherical_order

        # Number of spherical harmonic coefficients: 1 + 2*order (DC + cos/sin pairs)
        num_sh_coeffs = 1 + 2 * spherical_order

        # === Spherical/Angular Embedding for By, Bz ===
        self.angular_embed = nn.Sequential(
            nn.Linear(num_sh_coeffs, mod_features // 2),
            nn.SiLU(),
            nn.LayerNorm(mod_features // 2),
            nn.Linear(mod_features // 2, mod_features // 2),
        )

        # === Energy Input Factor (Bz intensity + Vwind) ===
        self.energy_embed = nn.Sequential(
            nn.Linear(2, mod_features),
            nn.SiLU(),
            nn.LayerNorm(mod_features),
            nn.Linear(mod_features, mod_features),
        )

        # === Bx embedding (least important, simple) ===
        self.bx_embed = nn.Sequential(
            nn.Linear(1, mod_features // 4),
            nn.SiLU(),
            nn.Linear(mod_features // 4, mod_features // 4),
        )

        # === Temporal Dynamics (derivatives) ===
        # 4 params + 4 first derivatives + 4 second derivatives = 12
        self.dynamics_embed = nn.Sequential(
            nn.Linear(12, mod_features // 2),
            nn.SiLU(),
            nn.LayerNorm(mod_features // 2),
            nn.Linear(mod_features // 2, mod_features // 2),
        )

        # === Combine all factors ===
        total_features = (
            mod_features // 2 +  # angular
            mod_features +       # energy
            mod_features // 4 +  # bx
            mod_features // 2    # dynamics
        )
        self.factor_combine = nn.Sequential(
            nn.Linear(total_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

        # === Cross-attention context encoder ===
        if context_channels is not None:
            self.context_proj = nn.Sequential(
                nn.Linear(4 + num_sh_coeffs + 8, context_channels),  # raw + SH + derivatives
                nn.SiLU(),
                nn.LayerNorm(context_channels),
                nn.Linear(context_channels, context_channels),
            )
        else:
            self.context_proj = None

    def compute_spherical_harmonics(self, by: Tensor, bz: Tensor) -> Tensor:
        """
        Compute circular harmonic features from IMF clock angle.

        theta = atan2(By, Bz)
        Features: [1, cos(θ), sin(θ), cos(2θ), sin(2θ), ...]
        """
        theta = torch.atan2(by, bz)

        features = [torch.ones_like(theta)]

        for n in range(1, self.spherical_order + 1):
            features.append(torch.cos(n * theta))
            features.append(torch.sin(n * theta))

        return torch.stack(features, dim=-1)

    def compute_derivatives(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute first and second derivatives via finite differences."""
        # First derivative
        dx = torch.zeros_like(x)
        dx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
        dx[:, 0] = x[:, 1] - x[:, 0]
        dx[:, -1] = x[:, -1] - x[:, -2]

        # Second derivative
        d2x = torch.zeros_like(x)
        d2x[:, 1:-1] = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        d2x[:, 0] = d2x[:, 1]
        d2x[:, -1] = d2x[:, -2]

        return dx, d2x

    def forward(self, mapping_cond: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            mapping_cond: (batch, time, 4) = [bx, by, bz, vwind]

        Returns:
            global_embed: (batch, mod_features) for ada-zero modulation
            context: (batch, time, context_channels) for cross-attention
        """
        # Correct order: [bx, by, bz, vwind]
        bx = mapping_cond[..., 0]
        by = mapping_cond[..., 1]
        bz = mapping_cond[..., 2]
        vwind = mapping_cond[..., 3]

        dx, d2x = self.compute_derivatives(mapping_cond)

        # Angular features via spherical harmonics
        sh_features = self.compute_spherical_harmonics(by, bz)
        angular_feat = self.angular_embed(sh_features).mean(dim=1)

        # Energy input (Bz signed + Vwind)
        energy_input = torch.stack([bz, vwind], dim=-1)
        energy_feat = self.energy_embed(energy_input).mean(dim=1)

        # Bx embedding
        bx_feat = self.bx_embed(bx.unsqueeze(-1)).mean(dim=1)

        # Dynamics
        dynamics_input = torch.cat([mapping_cond, dx, d2x], dim=-1)
        dynamics_feat = self.dynamics_embed(dynamics_input).mean(dim=1)

        # Combine
        global_embed = self.factor_combine(
            torch.cat([angular_feat, energy_feat, bx_feat, dynamics_feat], dim=-1)
        )

        # Cross-attention context
        if self.context_proj is not None:
            context_input = torch.cat([mapping_cond, sh_features, dx, d2x], dim=-1)
            context = self.context_proj(context_input)
        else:
            context = None

        return global_embed, context

class AttnPool1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        w = self.score(x)          # (B, 1, T)
        w = torch.softmax(w, dim=-1)
        return (x * w).sum(dim=-1)  # (B, C)
    
def block(dilation):
    return nn.Sequential(
        nn.Conv1d(
            in_features,
            64,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        ),
        nn.SiLU(),
        nn.Conv1d(
            64,
            64,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        ),
        nn.SiLU(),
    )

class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal convolution encoder for time series conditioning.

    Captures short-term, medium-term, and long-term patterns using different
    kernel sizes optimized for 2-minute temporal resolution (30 frames = 1 hour).

    Short-term (2 timesteps = 4 min): Immediate solar wind impact, rapid fluctuations
    Medium-term (10 timesteps = 20 min): Sustained driving, cumulative effects
    Long-term (20 timesteps = 40 min): Background state, overall trends
    """
    def __init__(self, in_features=4, mod_features=256):
        super().__init__()

        # Short-term: 2 timesteps = 4 minutes
        # self.short_term = nn.Sequential(
        #     nn.Conv1d(in_features, 64, kernel_size=2, padding='same'),
        #     nn.SiLU(),
        #     nn.BatchNorm1d(64),
        #     nn.Conv1d(64, 64, kernel_size=2, padding='same'),
        #     nn.SiLU(),
        #     nn.Conv1d(64, 64, kernel_size=2, padding='same'),
        # )

        # # Medium-term: 15 timesteps = 30 minutes
        # self.medium_term = nn.Sequential(
        #     nn.Conv1d(in_features, 64, kernel_size=15, padding='same'),
        #     nn.SiLU(),
        #     nn.BatchNorm1d(64),
        #     nn.Conv1d(64, 64, kernel_size=15, padding='same'),
        #     nn.SiLU(),
        #     nn.Conv1d(64, 64, kernel_size=15, padding='same'),
        # )

        # # Long-term: 30 timesteps = 60 minutes
        # self.long_term = nn.Sequential(
        #     nn.Conv1d(in_features, 64, kernel_size=30, padding='same'),
        #     nn.SiLU(),
        #     nn.BatchNorm1d(64),
        #     nn.Conv1d(64, 64, kernel_size=30, padding='same'),
        #     nn.SiLU(),
        #     nn.Conv1d(64, 64, kernel_size=30, padding='same'),
        # )

        # Multi-scale temporal encoders
        self.short_term  = nn.Sequential(
                            nn.Conv1d(
                                in_features,
                                64,
                                kernel_size=3,
                                dilation=1,
                                padding=1,
                            ),
                            nn.SiLU(),
                            nn.Conv1d(
                                64,
                                64,
                                kernel_size=3,
                                dilation=1,
                                padding=1,
                            ),
                            nn.SiLU(),
                        )   # local
        self.medium_term = nn.Sequential(
                            nn.Conv1d(
                                in_features,
                                64,
                                kernel_size=3,
                                dilation=5,
                                padding=5,
                            ),
                            nn.SiLU(),
                            nn.Conv1d(
                                64,
                                64,
                                kernel_size=3,
                                dilation=5,
                                padding=5,
                            ),
                            nn.SiLU(),
                        )    # medium-range
        self.long_term = nn.Sequential(
                            nn.Conv1d(
                                in_features,
                                64,
                                kernel_size=3,
                                dilation=10,
                                padding=10,
                            ),
                            nn.SiLU(),
                            nn.Conv1d(
                                64,
                                64,
                                kernel_size=3,
                                dilation=10,
                                padding=10,
                            ),
                            nn.SiLU(),
                        )    # long-range

        # self.mix = nn.Conv1d(64, 192, kernel_size=1)

        # For version 2.0
        self.mix = nn.Conv1d(192, 192, kernel_size=1)
        
        self.pool = AttnPool1D(192)

        # Combine all scales (64*3 = 192 features)
        self.combine = nn.Sequential(
            nn.Linear(192, mod_features),
            nn.SiLU(),
            nn.LayerNorm(mod_features),
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, time, 4) - [bx, by, bz, vwind] time series

        Returns:
            (batch, mod_features) - global temporal embedding
        """

        # embed()
        # x: (batch, time, 4) → (batch, 4, time) for conv1d
        x = x.transpose(1, 2).contiguous()  # CRITICAL: Make contiguous to avoid cuDNN workspace issues with gradient accumulation

        # CRITICAL FIX: Disable cuDNN entirely for dilated convolutions to avoid workspace corruption
        # This is a known bug with dilated Conv1d + cuDNN + mixed precision + gradient accumulation
        # Fallback to PyTorch's native implementation which is slower but stable
        cudnn_enabled = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = False
            with torch.amp.autocast('cuda', enabled=False):
                x = x.float()  # Ensure float32
                # Each scale: (batch, 4, 30) → (batch, 64, 30)
                short = self.short_term(x) #.mead(dim=-1) # (batch, 64)
                medium = self.medium_term(x) #.mead(dim=-1) # (batch, 64)
                long = self.long_term(x) #.mead(dim=-1)     # (batch, 64)

                # Concatenate: (batch, 192)
                # features = torch.cat([short, medium, long], dim=-1)

                # For version 2.0
                features = torch.cat([short, medium, long], dim=1)

                h = self.mix(features)      # (B, 192, 90)
                global_feat = self.pool(h) # (B, 192)
        finally:
            torch.backends.cudnn.enabled = cudnn_enabled

        # short = self.short_term(x).mean(dim=-1) # (batch, 64)
        # medium = self.medium_term(x).mean(dim=-1) # (batch, 64)
        # long = self.long_term(x).mean(dim=-1)     # (batch, 64)

        # features = torch.cat([short, medium, long], dim=-1)

        out = self.combine(global_feat)

        return out


class AdaptiveMultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoder with learnable attention pooling.

    Alternative to MultiScaleTemporalEncoder that uses attention to learn which
    timesteps are important within each scale, rather than simple mean pooling.

    This allows the model to adaptively focus on relevant temporal positions
    for each trajectory (e.g., recent timesteps vs distant past).
    """
    def __init__(self, in_features=4, mod_features=256):
        super().__init__()

        # Short-term: 2 timesteps = 4 minutes
        self.short_term = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=2, padding='same'),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=2, padding='same'),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding='same'),
        )

        # Medium-term: 15 timesteps = 30 minutes
        self.medium_term = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=15, padding='same'),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=15, padding='same'),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=15, padding='same'),
        )

        # Long-term: 30 timesteps = 60 minutes
        self.long_term = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=30, padding='same'),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=30, padding='same'),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=30, padding='same'),
        )

        # Learnable attention pooling for each scale
        self.short_attn = nn.Linear(64, 1)
        self.medium_attn = nn.Linear(64, 1)
        self.long_attn = nn.Linear(64, 1)

        # Combine all scales (64*3 = 192 features)
        self.combine = nn.Sequential(
            nn.Linear(192, mod_features),
            nn.SiLU(),
            nn.LayerNorm(mod_features),
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

    def attention_pool(self, x, attn_layer):
        """Attention-based pooling over temporal dimension.

        Args:
            x: (batch, channels, time) - convolutional features
            attn_layer: attention layer that scores each timestep

        Returns:
            (batch, channels) - attention-weighted features
        """
        x_t = x.transpose(1, 2)  # (batch, time, channels)
        scores = attn_layer(x_t)  # (batch, time, 1)
        weights = F.softmax(scores, dim=1)  # (batch, time, 1)
        return (x_t * weights).sum(dim=1)  # (batch, channels)

    def forward(self, x):
        """
        Args:
            x: (batch, time, 4) - [bx, by, bz, vwind] time series

        Returns:
            (batch, mod_features) - global temporal embedding
        """

        # embed()
        # x: (batch, time, 4) → (batch, 4, time) for conv1d
        x = x.transpose(1, 2)

        # Each scale: (batch, 4, time) → (batch, 64, time)
        short_conv = self.short_term(x)
        medium_conv = self.medium_term(x)
        long_conv = self.long_term(x)

        # Adaptive attention pooling instead of mean
        short = self.attention_pool(short_conv, self.short_attn)    # (batch, 64)
        medium = self.attention_pool(medium_conv, self.medium_attn)  # (batch, 64)
        long = self.attention_pool(long_conv, self.long_attn)        # (batch, 64)

        # Concatenate: (batch, 192)
        features = torch.cat([short, medium, long], dim=-1)

        # Final projection: (batch, 192) → (batch, mod_features)
        return self.combine(features)


class MultiheadSelfAttention(nn.Module):
    r"""Creates a multi-head self-attention layer.

    Arguments:
        channels: The number of channels :math:`H \times C`.
        attention_heads: The number of attention heads :math:`H`.
        qk_norm: Whether to use query-key RMS-normalization or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        attention_heads: int = 1,
        qk_norm: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ):
        super().__init__()

        assert channels % attention_heads == 0

        self.qkv_proj = nn.Linear(channels, 3 * channels, bias=False)
        self.y_proj = nn.Linear(channels, channels)

        if qk_norm:
            self.qk_norm = nn.RMSNorm(
                channels // attention_heads,
                elementwise_affine=False,
                eps=1e-5,
            )
        else:
            self.qk_norm = nn.Identity()

        self.heads = attention_heads
        self.dropout = nn.Dropout(0.0 if dropout is None else dropout)
        self.checkpointing = checkpointing

    def _forward(
        self,
        x: Tensor,
        theta: Optional[Tensor] = None,
        mask: Optional[Union[Tensor]] = None,# xfs.SparseCSRTensor]] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, H \times C)`.
            theta: Optional rotary positional embedding :math:`\theta`,
                with shape :math:`(*, L, H \times C / 2)`.
            mask: Optional attention mask, with shape :math:`(L, L)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, H \times C)`.
        """

        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "... L (n H C) -> n ... H L C", n=3, H=self.heads)
        q, k = self.qk_norm(q), self.qk_norm(k)

        if theta is not None:
            theta = rearrange(theta, "... L (H C) -> ... H L C", H=self.heads)
            q, k = apply_rope(q, k, theta)

        # if isinstance(mask, xfs.SparseCSRTensor):
        #     y = xfa.scaled_dot_product_attention(
        #         q=rearrange(q, "... L C -> (...) L C"),
        #         k=rearrange(k, "... L C -> (...) L C"),
        #         v=rearrange(v, "... L C -> (...) L C"),
        #         att_mask=xfa.SparseCS._wrap(mask),
        #         dropout=self.dropout if self.training else None,
        #     )
        #     y = y.reshape(q.shape[:-2] + y.shape[-2:])
        # else: !TODO remove later
        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0,
        )

        y = rearrange(y, "... H L C -> ... L (H C)")
        y = self.y_proj(y)

        return y

    def forward(
        self,
        x: Tensor,
        theta: Optional[Tensor] = None,
        mask: Optional[Union[Tensor]] = None, # xfs.SparseCSRTensor]] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, theta, mask, use_reentrant=False)
        else:
            return self._forward(x, theta, mask)


class MultiheadCrossAttention(nn.Module):
    r"""Creates a multi-head cross-attention layer.

    Arguments:
        channels: The number of channels :math:`H \times C`.
        context_channels: The number of context channels. If None, defaults to channels.
        attention_heads: The number of attention heads :math:`H`.
        qk_norm: Whether to use query-key RMS-normalization or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        context_channels: Optional[int] = None,
        attention_heads: int = 1,
        qk_norm: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ):
        super().__init__()

        assert channels % attention_heads == 0

        if context_channels is None:
            context_channels = channels

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.kv_proj = nn.Linear(context_channels, 2 * channels, bias=False)
        self.y_proj = nn.Linear(channels, channels)

        if qk_norm:
            self.qk_norm = nn.RMSNorm(
                channels // attention_heads,
                elementwise_affine=False,
                eps=1e-5,
            )
        else:
            self.qk_norm = nn.Identity()

        self.heads = attention_heads
        self.dropout = nn.Dropout(0.0 if dropout is None else dropout)
        self.checkpointing = checkpointing

    def _forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, H \times C)`.
            context: The context tokens, with shape :math:`(*, L_c, C_ctx)`.
            context_mask: Optional attention mask, with shape :math:`(L, L_c)`.

        Returns:
            The output tokens :math:`y`, with shape :math:`(*, L, H \times C)`.
        """

        q = self.q_proj(x)
        kv = self.kv_proj(context)

        q = rearrange(q, "... L (H C) -> ... H L C", H=self.heads)
        k, v = rearrange(kv, "... L (n H C) -> n ... H L C", n=2, H=self.heads)

        q, k = self.qk_norm(q), self.qk_norm(k)

        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=context_mask,
            dropout_p=self.dropout.p if self.training else 0,
        )

        y = rearrange(y, "... H L C -> ... L (H C)")
        y = self.y_proj(y)

        return y

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, context, context_mask, use_reentrant=False)
        else:
            return self._forward(x, context, context_mask)


def apply_rope(q: Tensor, k: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    r"""
    References:
        | RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
        | https://arxiv.org/abs/2104.09864

        | Rotary Position Embedding for Vision Transformer (Heo et al., 2024)
        | https://arxiv.org/abs/2403.13298

    Arguments:
        q: The query tokens :math:`q`, with shape :math:`(*, C)`.
        k: The key tokens :math:`k`, with shape :math:`(*, C)`.
        theta: Rotary angles, with shape :math:`(*, C / 2)`.

    Returns:
        The rotated query and key tokens, with shape :math:`(*, C)`.
    """

    # torch.polar and torch.view_as_complex don't support bfloat16
    # Cast to float32 for the entire operation, then cast back
    orig_dtype = q.dtype
    use_fp32 = orig_dtype == torch.bfloat16

    if use_fp32:
        q = q.float()
        k = k.float()
        theta = theta.float()

    rotation = torch.polar(torch.ones_like(theta), theta)

    q = torch.view_as_complex(torch.unflatten(q, -1, (-1, 2)))
    k = torch.view_as_complex(torch.unflatten(k, -1, (-1, 2)))

    q = torch.flatten(torch.view_as_real(rotation * q), -2)
    k = torch.flatten(torch.view_as_real(rotation * k), -2)

    # Cast back to original dtype if needed
    if use_fp32:
        q = q.to(orig_dtype)
        k = k.to(orig_dtype)

    return q, k



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
def Patchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... C (L l) -> ... L (C l)", l=l)
        else:
            return Rearrange("... C (L l) -> ... (C l) L", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... C (H h) (W w) -> ... H W (C h w)", h=h, w=w)
        else:
            return Rearrange("... C (H h) (W w) -> ... (C h w) H W", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... C (L l) (H h) (W w) -> ... L H W (C l h w)", l=l, h=h, w=w)
        else:
            return Rearrange("... C (L l) (H h) (W w) -> ... (C l h w) L H W", l=l, h=h, w=w)
    else:
        raise NotImplementedError()


def Unpatchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... L (C l) -> ... C (L l)", l=l)
        else:
            return Rearrange("... (C l) L -> ... C (L l)", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... H W (C h w) -> ... C (H h) (W w)", h=h, w=w)
        else:
            return Rearrange("... (C h w) H W -> ... C (H h) (W w)", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... L H W (C l h w) -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
        else:
            return Rearrange("... (C l h w) L H W -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
    else:
        raise NotImplementedError()

# def Unpatchify(patch_size: Sequence[int], t_out: int = 1, out_channels: int = 1, channel_last: bool = False) -> Rearrange:
#     if len(patch_size) == 3:
#         l, h, w = patch_size
#         if channel_last:
#             return Rearrange(
#                 "... L H W (C l h w t_out) -> ... (C t_out) (L l) (H h) (W w)",
#                 l=l, h=h, w=w, t_out=t_out
#             )
#         else:
#             return Rearrange(
#                 "... (C l h w t_out) L H W -> ... (C t_out) (L l) (H h) (W w)",
#                 l=l, h=h, w=w, t_out=t_out
#             )
#     else:
#         raise NotImplementedError("Unpatchify with t_out > 1 only implemented for 3D patches")




class ViTBlock(nn.Module):
    r"""Creates a ViT block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        context_channels: The number of context channels for cross-attention. If None, no cross-attention.
        ffn_factor: The channel factor in the FFN.
        spatial: The number of spatial dimensinons :math:`N`.
        rope: Whether to use rotary positional embedding (RoPE) or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to :class:`MultiheadSelfAttention`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        context_channels: Optional[int] = None,
        ffn_factor: int = 4,
        spatial: int = 2,
        rope: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        per_frame_mod: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing
        self.has_cross_attn = context_channels is not None

        # Ada-LN Zero
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

        # Determine number of modulation parameters based on whether we have cross-attention
        # Standard: 4 params (self-attn scale/shift, ffn gate, skip gate)
        # With cross-attn: 6 params (add cross-attn scale/shift)
        num_params = 6 if self.has_cross_attn else 4

        # per_frame_mod: when True, mod is (B, L, D) per-token instead of (B, D) global
        # "... (n C) -> n ... C"  keeps the token dim (no broadcast)
        # "... (n C) -> n ... 1 C" adds a 1 dim that broadcasts across all tokens
        rearrange_pattern = "... (n C) -> n ... C" if per_frame_mod else "... (n C) -> n ... 1 C"

        if mod_features > 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, num_params * channels),
                Rearrange(rearrange_pattern, n=num_params),
            )

            self.ada_zero[-2].weight.data.mul_(1e-2)
        else:
            self.ada_zero = nn.Parameter(torch.randn(num_params, channels))
            self.ada_zero.data.mul_(1e-2)

        # MSA
        self.msa = MultiheadSelfAttention(channels, **kwargs)

        # Cross-attention (optional)
        if self.has_cross_attn:
            self.cross_attn = MultiheadCrossAttention(
                channels=channels,
                context_channels=context_channels,
                **kwargs,
            )
            self.norm_cross = nn.LayerNorm(channels, elementwise_affine=False)

        ## Rotary PE
        if rope:
            amplitude = 1e2 ** -torch.rand(channels // 2)
            direction = torch.nn.functional.normalize(torch.randn(spatial, channels // 2), dim=0)

            self.theta = nn.Parameter(amplitude * direction)
        else:
            self.theta = None

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_factor * channels),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(ffn_factor * channels, channels),
        )

    def _forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        coo: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        skip: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, C)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(*, D)`.
            coo: The postition coordinates, with shape :math:`(*, L, N)`.
            mask: The attention mask, with shape :math:`(*, L, L)`.
            skip: A skip connection, with shape :math:`(*, L, C)`.
            context: Optional context for cross-attention, with shape :math:`(*, L_c, C_ctx)`.
            context_mask: Optional cross-attention mask, with shape :math:`(*, L, L_c)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, C)`.
        """

        if self.theta is None:
            theta = None
        else:
            theta = torch.einsum("...ij,jk", coo, self.theta)

        # embed()

        if torch.is_tensor(self.ada_zero):
            if self.has_cross_attn:
                a, b, c_ca, d_ca, c, d = self.ada_zero
            else:
                a, b, c, d = self.ada_zero
        else:
            params = self.ada_zero(mod)
            if self.has_cross_attn:
                a, b, c_ca, d_ca, c, d = params
            else:
                a, b, c, d = params

        # Self-attention (original behavior preserved)
        y = (a + 1) * self.norm(x) + b
        y = y + self.msa(y, theta, mask)

        # Cross-attention (if enabled and context provided)
        if self.has_cross_attn and context is not None:
            y_cross = (c_ca + 1) * self.norm_cross(y) + d_ca
            y = y + self.cross_attn(y_cross, context, context_mask)

        # FFN (original behavior preserved)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        # Skip connection (original behavior preserved)
        if skip is not None:
            y = (y + d * skip) * torch.rsqrt(1 + d * d)

        return y

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        coo: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        skip: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.checkpointing:
            # print(f"Input to ViTBlock: {x.shape}")
            res = checkpoint(self._forward, x, mod, coo, mask, skip, context, context_mask, use_reentrant=False)
            # print(f"Output from ViTBlock: {res.shape}")
            return res
        else:
            return self._forward(x, mod, coo, mask, skip, context, context_mask)


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_frames: int,
        cond_frames: int = 15,
        cond_channels: int = 0,
        mod_features: int = 256,
        hid_channels: int = 512,
        hid_blocks: int = 16,
        spatial: int = 3,
        patch_size: Union[int, Sequence[int]] = 1,
        unpatch_size: Union[int, Sequence[int], None] = None,
        window_size: Union[int, Sequence[int], None] = None,
        channel_mapping_cond: int = 15,
        context_channels: Optional[int] = None,
        use_physics_embedding: bool = False,
        use_multiscale_temporal: bool = False,
        use_adaptive_temporal: bool = False,
        use_per_frame_mod: bool = False,
        spherical_order: int = 4,
        use_cond_img_in_adaln: bool = False,
        **kwargs,
    ):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial
        if unpatch_size is None:
            unpatch_size = patch_size
        elif isinstance(unpatch_size, int):
            unpatch_size = [unpatch_size] * spatial



        self.patch_size = patch_size[-1] if isinstance(patch_size, Sequence) else patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cond_channels = cond_channels
        self.channel_mapping_cond = channel_mapping_cond
        self.pred_frames = pred_frames
        self.has_variance = out_channels > in_channels
        self.context_channels = context_channels
        self.use_physics_embedding = use_physics_embedding
        self.use_multiscale_temporal = use_multiscale_temporal
        self.use_adaptive_temporal = use_adaptive_temporal
        self.use_per_frame_mod = use_per_frame_mod
        self.use_cond_img_in_adaln = use_cond_img_in_adaln

        self.patch = Patchify(patch_size, channel_last=True)
        self.unpatch = Unpatchify(unpatch_size, channel_last=True)

        # +1 for temporal coordinate channel
        self.in_proj = nn.Linear(math.prod(patch_size) * (in_channels + cond_channels + 1 + 4), hid_channels) # 1 is for temporal index channel, 4 is for L1 cond channels
        self.out_proj = nn.Linear(hid_channels, math.prod(patch_size) * out_channels)
        # Learned temporal projection: maps all (cond+pred) frame outputs → pred_frames outputs
        # More flexible than hard-slicing: learns a weighted combination over all time positions
        self.time_proj = nn.Linear(cond_frames + pred_frames, pred_frames)

        self.positional_embedding = nn.Sequential(
            SineEncoding(hid_channels),
            Rearrange("... N C -> ... (N C)"),
            nn.Linear(spatial * hid_channels, hid_channels),
        )

        self.blocks = nn.ModuleList([
            ViTBlock(
                channels=hid_channels,
                mod_features=mod_features,
                context_channels=context_channels,
                spatial=spatial,
                checkpointing=True,
                per_frame_mod=use_per_frame_mod,
                **kwargs,
            ) for _ in range(hid_blocks)
        ])

        # Per-frame modulation: project each frame's L1 features to mod_features
        # so that each frame gets its own ada-zero parameters
        if use_per_frame_mod:
            self.per_frame_proj = nn.Sequential(
                nn.Linear(4, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, mod_features),
            )

        self.timestep_embed = nn.Sequential(
            SineEncoding(mod_features),
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )
        self.mapping = nn.Sequential(
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )
        self.mapping_cond = nn.Sequential(
            nn.Linear(self.channel_mapping_cond * 4, mod_features),  
            nn.SiLU(),
            nn.LayerNorm(mod_features),
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

        # Cross-attention context projection for mapping_cond time series
        if context_channels is not None:
            self.context_proj = nn.Sequential(
                nn.Linear(4, context_channels),  # Project each frame's 4 features to context_channels
                nn.SiLU(),
                nn.Linear(context_channels, context_channels),
            )
        else:
            self.context_proj = None

        # Physics-informed conditioning module (optional)
        if use_physics_embedding:
            self.physics_cond_module = PhysicsInformedConditioningModule(
                mod_features=mod_features,
                context_channels=context_channels,
                spherical_order=spherical_order,
            )
        else:
            self.physics_cond_module = None

        # Multi-scale temporal encoder (optional)
        if use_multiscale_temporal:
            self.multiscale_temporal = MultiScaleTemporalEncoder(
                in_features=4,  # [bx, by, bz, vwind]
                mod_features=mod_features,
            )
        else:
            self.multiscale_temporal = None

        # Adaptive multi-scale temporal encoder (optional)
        if use_adaptive_temporal:
            self.adaptive_temporal = AdaptiveMultiScaleTemporalEncoder(
                in_features=4,  # [bx, by, bz, vwind]
                mod_features=mod_features,
            )
        else:
            self.adaptive_temporal = None

        # Conditioning image encoder for adaLN (optional)
        # Small CNN applied independently to each past frame (shared weights),
        # then global-avg-pooled and mean-reduced over the temporal axis → mod_features
        if use_cond_img_in_adaln:
            # Shared CNN: extracts a 64-dim feature vector from each frame independently
            self.cond_img_conv_encoder = nn.Sequential(
                # (B*T, 1, 64, 64)
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # → (B*T, 32, 32, 32)
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # → (B*T, 64, 16, 16)
                nn.SiLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # → (B*T, 64,  8,  8)
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),  # → (B*T, 64, 1, 1)
                nn.Flatten(),             # → (B*T, 64)
            )
            # Attention pooling over the T frame features → (B, 64)
            self.cond_img_temporal_pool = AttnPool1D(64)
            # Final projection to mod_features
            self.cond_img_adaln_proj = nn.Sequential(
                nn.Linear(64, mod_features),
                nn.SiLU(),
                nn.LayerNorm(mod_features),
                nn.Linear(mod_features, mod_features),
            )
        else:
            self.cond_img_conv_encoder = None
            self.cond_img_temporal_pool = None
            self.cond_img_adaln_proj = None

        self.spatial = spatial
        self.window_size = tuple(window_size) if isinstance(window_size, Sequence) else ((window_size,) * spatial if window_size else None)

    @staticmethod
    def coo_and_mask(shape: Sequence[int], time_position: Optional[Tensor], spatial: int,
                     window_size: Sequence[int], dtype: torch.dtype, device: torch.device) -> Tuple[Tensor, Optional[Tensor]]:
        # Create coordinate grid for spatial dimensions
        # shape: (num_frames, H, W) -> coo: (num_tokens, spatial) where spatial=3
        coo = torch.cartesian_prod(*[torch.arange(s, device=device) for s in shape])
        coo = coo.view(-1, spatial)

        # If temporal positions are provided, replace the first dimension (time index) with actual temporal positions
        if time_position is not None:
            # time_position: (batch, num_frames) containing temporal positions like [-1.0, ..., 0.93]
            # coo: (num_tokens, spatial) where coo[:, 0] contains frame indices [0,0,0,...,1,1,1,...,2,2,2,...]

            batch_size = time_position.shape[0]
            num_tokens = coo.shape[0]

            # Create batch-aware coordinates: (batch, num_tokens, spatial)
            position = torch.zeros(batch_size, num_tokens, spatial, device=device, dtype=dtype)

            # Copy spatial dimensions (lat, lon) for all batch elements
            position[:, :, 1:] = coo[:, 1:].to(dtype=dtype)

            # For each batch element, map frame indices to actual temporal positions
            for b in range(batch_size):
                # coo[:, 0] contains frame indices [0,0,...,1,1,...,2,2,...]
                # time_position[b] contains actual temporal coords for each frame
                position[b, :, 0] = time_position[b, coo[:, 0].long()]

            coo = position
        else:
            # Add batch dimension for consistency
            coo = coo.unsqueeze(0).to(dtype=dtype)

        if window_size is None:
            return coo, None

        delta = torch.abs(coo[:, None] - coo[None, :])
        delta = torch.minimum(delta, delta.new_tensor(shape) - delta)
        mask = torch.all(delta <= coo.new_tensor(window_size) // 2, dim=-1)

        # if xfa._has_cpp_library:
        #     mask = xfa.SparseCS(mask, device=mask.device)._mat !TODO remove later

        return coo.to(dtype=dtype), mask

    def forward(
        self,
        input: Tensor,
        sigma: Tensor,
        mapping_cond: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        num_pred_frames = input.shape[1]
        num_cond_frames = cond.shape[1] if cond is not None else 0

        # --- Timestep embedding ---
        c_noise = sigma.log() / 4
        timestep_embed = self.timestep_embed(c_noise.unsqueeze(-1)).squeeze(1)  # (B, D)

        # --- Solar wind conditioning ---
        if mapping_cond is None:
            mapping_cond_embed = torch.zeros_like(timestep_embed)
            context = None
        elif self.use_per_frame_mod:
            per_frame_embed = self.per_frame_proj(mapping_cond)                        # (B, T, D)
            global_embed = self.mapping_cond(mapping_cond.flatten(-2))                 # (B, D)
            mapping_cond_embed = per_frame_embed + global_embed.unsqueeze(1)           # (B, T, D)
            context = None
        else:
            if self.use_physics_embedding and self.physics_cond_module is not None:
                mapping_cond_embed, context = self.physics_cond_module(mapping_cond)
            elif self.use_multiscale_temporal and self.multiscale_temporal is not None:
                mapping_cond_embed = self.multiscale_temporal(mapping_cond).unsqueeze(1)
                context = self.context_proj(mapping_cond) if self.context_proj is not None else None
            elif self.use_adaptive_temporal and self.adaptive_temporal is not None:
                mapping_cond_embed = self.adaptive_temporal(mapping_cond).unsqueeze(1)
                context = self.context_proj(mapping_cond) if self.context_proj is not None else None
            else:
                # Classic: flatten full solar wind window → MLP
                mapping_cond_embed = self.mapping_cond(mapping_cond.flatten(-2))       # (B, D)
                context = self.context_proj(mapping_cond) if self.context_proj is not None else None

        if self.use_per_frame_mod and mapping_cond_embed.dim() == 3:
            mapping_out = self.mapping(timestep_embed.unsqueeze(1) + mapping_cond_embed)  # (B, T, D)
        else:
            mapping_out = self.mapping(timestep_embed + mapping_cond_embed.squeeze(1))    # (B, D)

        context_mask = None

        # --- Past frame conditioning on adaLN ---
        if self.use_cond_img_in_adaln and cond is not None and self.cond_img_conv_encoder is not None:
            B_c, T_c, H_c, W_c = cond.shape
            frame_feats = self.cond_img_conv_encoder(cond.reshape(B_c * T_c, 1, H_c, W_c))  # (B*T, 64)
            frame_feats = frame_feats.reshape(B_c, 64, T_c)                                  # (B, 64, T)
            pooled = self.cond_img_temporal_pool(frame_feats)                                # (B, 64)
            mapping_out = mapping_out + self.cond_img_adaln_proj(pooled)                     # (B, D)

        # Per-frame mod: broadcast (B, T, D) to (B, T*Hp*Wp, D)
        if self.use_per_frame_mod and mapping_out.dim() == 3:
            tokens_per_frame = shape[1] * shape[2]
            mapping_out = mapping_out.repeat_interleave(tokens_per_frame, dim=1)

        # --- Build input tensor (B, 6, T, H, W): map + time coord + L1 per-pixel ---
        if cond is not None:
            input = torch.cat([cond, input], dim=1)  # (B, T_total, H, W)

        batch_size = input.shape[0]
        total_frames = input.shape[1]

        # Temporal coordinates in [-1, 1], past negative, future positive
        temporal_positions = torch.arange(-num_cond_frames, num_pred_frames,
                                          device=input.device, dtype=input.dtype)
        max_horizon = max(num_cond_frames, num_pred_frames)
        temporal_positions_normalized = temporal_positions / max_horizon
        time_position = temporal_positions_normalized.unsqueeze(0).expand(batch_size, -1)

        time_coord = temporal_positions_normalized.view(1, -1, 1, 1).expand(
            batch_size, total_frames, input.shape[-2], input.shape[-1])

        if mapping_cond is None:
            l1_cond = torch.zeros(batch_size, total_frames, 4, input.shape[-2], input.shape[-1],
                                  device=input.device, dtype=input.dtype)
        else:
            l1_cond = mapping_cond.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, total_frames, 4, input.shape[-2], input.shape[-1])

        input = torch.cat([
            input.unsqueeze(1),                        # (B, 1, T, H, W)
            time_coord.unsqueeze(1),                   # (B, 1, T, H, W)
            l1_cond.permute(0, 2, 1, 3, 4),            # (B, 4, T, H, W)
        ], dim=1)                                      # (B, 6, T, H, W)

        # --- Transformer ---
        input = self.in_proj(self.patch(input))
        shape = input.shape[-self.spatial - 1: -1]

        coo, mask = self.coo_and_mask(shape, time_position=time_position, spatial=self.spatial,
                                      window_size=self.window_size, dtype=input.dtype, device=input.device)

        x = skip = torch.flatten(input, -self.spatial - 1, -2)
        x = x + self.positional_embedding(coo)

        for block in self.blocks:
            x = block(x, mapping_out.squeeze(1), coo=coo, mask=mask, skip=skip,
                      context=context, context_mask=context_mask)

        # --- Output: learned temporal projection T_total → pred_frames ---
        x = self.unpatch(self.out_proj(torch.unflatten(x, sizes=shape, dim=-2)))
        x = x.permute(0, 1, 3, 4, 2)   # (B, C, H, W, T_total)
        x = self.time_proj(x)            # (B, C, H, W, pred_frames)
        x = x.permute(0, 1, 4, 2, 3)   # (B, C, pred_frames, H, W)

        return x.squeeze(1)
        
    def param_groups(self, base_lr=2e-4):
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