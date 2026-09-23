"""Simple deterministic convolutional UNet baseline.

Takes past conditioning frames and directly predicts future frames (MSE loss).
No diffusion, no sigma, no noise — pure regression baseline.

    forward(cond) -> pred
where:
    cond : (B, cond_frames, H, W)  — past frames
    pred : (B, pred_frames, H, W)  — predicted future frames
"""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Downsample2d, Upsample2d


def _num_groups(channels):
    for g in [32, 16, 8, 4, 1]:
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Conv residual block with AdaGN conditioning.

    emb_proj is per-block because each level has different c_out —
    the projection emb_dim → c_out*2 must match the block's output channels.
    emb itself (the L1 embedding) is computed once outside and passed in.
    """

    def __init__(self, c_in, c_out, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1    = nn.GroupNorm(_num_groups(c_in), c_in)
        self.conv1    = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, c_out * 2)   # scale + shift
        self.norm2    = nn.GroupNorm(_num_groups(c_out), c_out)
        self.conv2    = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.drop     = nn.Dropout(dropout)
        self.skip     = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class UNetSimple(nn.Module):
    """Deterministic conv UNet: cond_frames -> pred_frames (MSE regression).

    Args:
        in_channels    : channels per frame (1 for ionosphere maps)
        out_channels   : channels per output frame (1)
        cond_frames    : number of conditioning (past) frames
        pred_frames    : number of frames to predict
        base_channels  : channel width at first encoder level
        channel_mults  : multipliers per level, e.g. (1, 2, 4, 4)
        num_res_blocks : residual blocks per encoder/decoder level
        dropout        : dropout rate inside ResBlocks
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        cond_frames=15,
        pred_frames=7,
        base_channels=128,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0,
        l1_features=4,       # number of L1 measurements per timestep
        total_frames=22,     # cond_frames + pred_frames, for L1 input size
    ):
        super().__init__()
        self.pred_frames  = pred_frames
        self.out_channels = out_channels
        n_levels  = len(channel_mults)
        total_in  = in_channels * cond_frames
        total_out = out_channels * pred_frames
        emb_dim   = base_channels * 4

        # L1 conditions: (B, total_frames, l1_features) → flatten → MLP → emb
        self.l1_embed = nn.Sequential(
            nn.Linear(total_frames * l1_features, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # input projection
        c0 = base_channels * channel_mults[0]
        self.in_proj = nn.Conv2d(total_in, c0, 3, padding=1)

        # encoder
        self.enc_blocks = nn.ModuleList()
        self.downs       = nn.ModuleList()
        enc_out_ch = []
        c = c0
        for i, mult in enumerate(channel_mults):
            c_out = base_channels * mult
            level = nn.ModuleList()
            for _ in range(num_res_blocks):
                level.append(ResBlock(c, c_out, emb_dim, dropout))
                c = c_out
            self.enc_blocks.append(level)
            enc_out_ch.append(c)
            if i < n_levels - 1:
                self.downs.append(Downsample2d())

        # middle
        self.mid_blocks = nn.ModuleList([
            ResBlock(c, c, emb_dim, dropout),
            ResBlock(c, c, emb_dim, dropout),
        ])

        # decoder
        self.ups        = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            c_skip = enc_out_ch[n_levels - 1 - i]
            c_out  = base_channels * mult
            if i > 0:
                self.ups.append(Upsample2d())
            c_in_first = c + c_skip
            level = nn.ModuleList()
            for j in range(num_res_blocks):
                level.append(ResBlock(c_in_first if j == 0 else c_out, c_out, emb_dim, dropout))
            self.dec_blocks.append(level)
            c = c_out

        # output
        self.out_norm = nn.GroupNorm(_num_groups(c), c)
        self.out_proj = nn.Conv2d(c, total_out, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.n_levels = n_levels

    def param_groups(self, lr):
        return [{'params': self.parameters(), 'lr': lr}]

    def forward(self, cond, l1_cond=None):
        """
        cond    : (B, cond_frames, H, W)
        l1_cond : (B, total_frames, 4) — L1 solar wind conditions, or None
        returns   (B, pred_frames, H, W)
        """
        # L1 embedding: computed once, passed to every ResBlock
        if l1_cond is not None:
            emb = self.l1_embed(l1_cond.flatten(1))   # (B, emb_dim)
        else:
            emb = torch.zeros(cond.shape[0], next(self.l1_embed.parameters()).shape[0],
                              device=cond.device, dtype=cond.dtype)

        h = self.in_proj(cond)

        skips = []
        for i, level in enumerate(self.enc_blocks):
            for block in level:
                h = block(h, emb)
            skips.append(h)
            if i < self.n_levels - 1:
                h = self.downs[i](h)

        for block in self.mid_blocks:
            h = block(h, emb)

        up_idx = 0
        for i, level in enumerate(self.dec_blocks):
            if i > 0:
                h = self.ups[up_idx](h)
                up_idx += 1
            h = torch.cat([h, skips[self.n_levels - 1 - i]], dim=1)
            for block in level:
                h = block(h, emb)

        return self.out_proj(F.silu(self.out_norm(h)))   # (B, pred_frames, H, W)
