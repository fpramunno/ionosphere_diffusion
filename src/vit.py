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
]

import functools
import math
import torch
import torch.nn as nn
import numpy as np
import warnings
from einops import rearrange
from torch.nn import functional as F


import xformers.components.attention.core as xfa
import xformers.sparse as xfs
    
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Hashable, Optional, Sequence, Tuple, Union
from torch.utils.checkpoint import checkpoint


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
        mask: Optional[Union[Tensor, xfs.SparseCSRTensor]] = None,
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

        if isinstance(mask, xfs.SparseCSRTensor):
            y = xfa.scaled_dot_product_attention(
                q=rearrange(q, "... L C -> (...) L C"),
                k=rearrange(k, "... L C -> (...) L C"),
                v=rearrange(v, "... L C -> (...) L C"),
                att_mask=xfa.SparseCS._wrap(mask),
                dropout=self.dropout if self.training else None,
            )
            y = y.reshape(q.shape[:-2] + y.shape[-2:])
        else:
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
        mask: Optional[Union[Tensor, xfs.SparseCSRTensor]] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, theta, mask, use_reentrant=False)
        else:
            return self._forward(x, theta, mask)


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

    rotation = torch.polar(torch.ones_like(theta), theta)

    q = torch.view_as_complex(torch.unflatten(q, -1, (-1, 2)))
    k = torch.view_as_complex(torch.unflatten(k, -1, (-1, 2)))

    q = torch.flatten(torch.view_as_real(rotation * q), -2)
    k = torch.flatten(torch.view_as_real(rotation * k), -2)

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
        ffn_factor: int = 4,
        spatial: int = 2,
        rope: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Ada-LN Zero
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

        if mod_features > 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, 4 * channels),
                Rearrange("... (n C) -> n ... 1 C", n=4),
                # Rearrange("... (n C) -> ... n C", n=4)
            )

            self.ada_zero[-2].weight.data.mul_(1e-2)
        else:
            self.ada_zero = nn.Parameter(torch.randn(4, channels))
            self.ada_zero.data.mul_(1e-2)

        # MSA
        self.msa = MultiheadSelfAttention(channels, **kwargs)

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
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, C)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(*, D)`.
            coo: The postition coordinates, with shape :math:`(*, L, N)`.
            mask: The attention mask, with shape :math:`(*, L, L)`.
            skip: A skip connection, with shape :math:`(*, L, C)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, C)`.
        """

        if self.theta is None:
            theta = None
        else:
            theta = torch.einsum("...ij,jk", coo, self.theta)

        if torch.is_tensor(self.ada_zero):
            a, b, c, d = self.ada_zero
        else:
            a, b, c, d = self.ada_zero(mod) 

        y = (a + 1) * self.norm(x) + b
        y = y + self.msa(y, theta, mask)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

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
    ) -> Tensor:
        if self.checkpointing:
            # print(f"Input to ViTBlock: {x.shape}")
            res = checkpoint(self._forward, x, mod, coo, mask, skip, use_reentrant=False)
            # print(f"Output from ViTBlock: {res.shape}")
            return res
        else:
            return self._forward(x, mod, coo, mask, skip)


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_out: int,
        cond_channels: int = 0,
        mod_features: int = 256,
        hid_channels: int = 512,
        hid_blocks: int = 16,
        spatial: int = 3,
        patch_size: Union[int, Sequence[int]] = 1,
        unpatch_size: Union[int, Sequence[int], None] = None,
        window_size: Union[int, Sequence[int], None] = None,
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
        self.t_out = t_out
        self.has_variance = out_channels > in_channels

        self.patch = Patchify(patch_size, channel_last=True)
        self.unpatch = Unpatchify(unpatch_size, channel_last=True)

        self.in_proj = nn.Linear(math.prod(patch_size) * (in_channels + cond_channels), hid_channels)
        self.out_proj = nn.Linear(hid_channels, math.prod(patch_size) * out_channels)
        self.time_compressor = nn.Linear(10, t_out)
        # self.time_compressor = nn.Conv1d(in_channels=1, out_channels=t_out, kernel_size=1)

        self.positional_embedding = nn.Sequential(
            SineEncoding(hid_channels),
            Rearrange("... N C -> ... (N C)"),
            nn.Linear(spatial * hid_channels, hid_channels),
        )

        self.blocks = nn.ModuleList([
            ViTBlock(
                channels=hid_channels,
                mod_features=mod_features,
                spatial=spatial,
                checkpointing=True,
                **kwargs,
            ) for _ in range(hid_blocks)
        ])

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
            SineEncoding(mod_features),
            nn.Linear(mod_features, mod_features, bias=False),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

        self.spatial = spatial
        self.window_size = tuple(window_size) if isinstance(window_size, Sequence) else ((window_size,) * spatial if window_size else None)

    @staticmethod
    @functools.cache
    def coo_and_mask(shape: Sequence[int], spatial: int, window_size: Sequence[int], dtype: torch.dtype, device: torch.device) -> Tuple[Tensor, Optional[Tensor]]:
        coo = torch.cartesian_prod(*[torch.arange(s, device=device) for s in shape])
        coo = coo.view(-1, spatial)

        if window_size is None:
            return coo.to(dtype=dtype), None

        delta = torch.abs(coo[:, None] - coo[None, :])
        delta = torch.minimum(delta, delta.new_tensor(shape) - delta)
        mask = torch.all(delta <= coo.new_tensor(window_size) // 2, dim=-1)

        if xfa._has_cpp_library:
            mask = xfa.SparseCS(mask, device=mask.device)._mat

        return coo.to(dtype=dtype), mask

    def forward(
        self,
        input: Tensor,
        sigma: Tensor,
        mapping_cond: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        return_variance: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        c_noise = sigma.log() / 4
        timestep_embed = self.timestep_embed(c_noise.unsqueeze(-1))
        mapping_cond_embed = torch.zeros_like(timestep_embed) if mapping_cond is None else self.mapping_cond(mapping_cond)
        mapping_out = self.mapping(timestep_embed + mapping_cond_embed.mean(dim=1).unsqueeze(1))

        if cond is not None:
            input = torch.cat([input, cond], dim=1)
        
        input = self.patch(input)
        input = self.in_proj(input)
        shape = input.shape[-self.spatial - 1: -1]

        coo, mask = self.coo_and_mask(shape, spatial=self.spatial, window_size=self.window_size, dtype=input.dtype, device=input.device)

        x = skip = torch.flatten(input, -self.spatial - 1, -2)
        x = x + self.positional_embedding(coo)

        for block in self.blocks:
            x = block(x, mapping_out.squeeze(1), coo=coo, mask=mask, skip=skip)
        
        x = torch.unflatten(x, sizes=shape, dim=-2)
        x = self.out_proj(x)
        x = self.unpatch(x)

        return x
        
        
        # return x

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