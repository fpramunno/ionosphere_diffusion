import torch
import os
import shutil

# Test checkpoint saving on your network filesystem
save_dir = "/capstor/store/cscs/ska/sk035/test_checkpoint_save"
os.makedirs(save_dir, exist_ok=True)

print("Testing checkpoint save on network filesystem...")

# Create a dummy checkpoint (small version)
dummy_ckpt = {
    'model': {f'layer_{i}': torch.randn(1000, 1000) for i in range(10)},  # ~40MB
    'opt': {f'param_{i}': torch.randn(1000, 1000) for i in range(10)},
    'epoch': 0,
    'step': 100,
}

# Method 1: Direct save (current method - can fail)
print("\n1. Testing direct save...")
try:
    filename1 = os.path.join(save_dir, "test_direct.pth")
    torch.save(dummy_ckpt, filename1)
    # Try to load it
    loaded = torch.load(filename1, map_location='cpu')
    print(f"   ✅ Direct save works! File size: {os.path.getsize(filename1)/(1024**2):.1f} MB")
    os.remove(filename1)
except Exception as e:
    print(f"   ❌ Direct save failed: {e}")

# Method 2: Atomic save with temp file (safer)
print("\n2. Testing atomic save (temp + rename)...")
try:
    filename2 = os.path.join(save_dir, "test_atomic.pth")
    temp_file = filename2 + ".tmp"

    torch.save(dummy_ckpt, temp_file)
    shutil.move(temp_file, filename2)

    # Try to load it
    loaded = torch.load(filename2, map_location='cpu')
    print(f"   ✅ Atomic save works! File size: {os.path.getsize(filename2)/(1024**2):.1f} MB")
    os.remove(filename2)
except Exception as e:
    print(f"   ❌ Atomic save failed: {e}")

# Method 3: Save to local disk first, then copy (most reliable for network FS)
print("\n3. Testing local save + copy to network...")
try:
    local_temp = "/tmp/test_local.pth"
    final_file = os.path.join(save_dir, "test_local_copy.pth")

    torch.save(dummy_ckpt, local_temp)
    shutil.copy2(local_temp, final_file)

    # Try to load from network location
    loaded = torch.load(final_file, map_location='cpu')
    print(f"   ✅ Local+copy works! File size: {os.path.getsize(final_file)/(1024**2):.1f} MB")

    os.remove(local_temp)
    os.remove(final_file)
except Exception as e:
    print(f"   ❌ Local+copy failed: {e}")

# Cleanup
shutil.rmtree(save_dir)
print("\n✅ Test complete!")