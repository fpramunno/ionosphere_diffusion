"""Quick script to explore UNetSimple parameter counts across configs."""

import sys
sys.path.insert(0, '.')

from src.unet_simple import UNetSimple

TARGET = 866_249_953

def count(base_channels, channel_mults, num_res_blocks):
    m = UNetSimple(
        in_channels=1, out_channels=1,
        cond_frames=15, pred_frames=7,
        base_channels=base_channels,
        channel_mults=tuple(channel_mults),
        num_res_blocks=num_res_blocks,
        total_frames=22,
    )
    n = sum(p.numel() for p in m.parameters())
    ratio = n / TARGET
    return n, ratio

print(f"{'base_ch':>8} {'mults':<20} {'res_blocks':>10} {'params':>15} {'ratio vs ViT':>13}")
print("-" * 72)

configs = [
    # --- max 2048 channels (known to cause cuDNN OOM) ---
    (256,  [1,2,4,8],    2),   # 646M  -- FAILS (2048ch)
    (256,  [1,2,4,8],    3),   # 862M  -- FAILS (2048ch)
    # --- max 1024 channels: [1,2,4,4] with base=256 ---
    (256,  [1,2,4,4],    4),
    (256,  [1,2,4,4],    5),
    (256,  [1,2,4,4],    6),
    (256,  [1,2,4,4],    7),
    (256,  [1,2,4,4],    8),
    # --- max 1024 channels: [1,2,4,4] with base=320 ---
    (320,  [1,2,4,4],    3),
    (320,  [1,2,4,4],    4),
    (320,  [1,2,4,4],    5),
    (320,  [1,2,4,4],    6),
    # --- max 1024 channels: [1,2,4,4] with base=384 ---
    (384,  [1,2,4,4],    2),
    (384,  [1,2,4,4],    3),
    (384,  [1,2,4,4],    4),
    (384,  [1,2,4,4],    5),
    # --- max 1024 channels: deeper mults, lower base ---
    (192,  [1,2,4,8],    3),   # max=1536 -- borderline
    (160,  [1,2,4,8],    4),   # max=1280 -- borderline
    (128,  [1,2,4,8],    6),   # max=1024
    (128,  [1,2,4,8],    8),   # max=1024
]

for base_ch, mults, res_blocks in configs:
    n, ratio = count(base_ch, mults, res_blocks)
    marker = " <<<" if abs(ratio - 1.0) < 0.15 else ""
    print(f"{base_ch:>8} {str(mults):<20} {res_blocks:>10} {n:>15,} {ratio:>12.2f}x{marker}")

print(f"\nTarget: {TARGET:,} (ViT)")
