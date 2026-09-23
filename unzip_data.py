"""
Fast parallel unzip of ionosphere map archives.
Uses multiprocessing so each task runs in its own process with a live progress bar.
"""

import subprocess
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

ZIPS_DIR = Path("./data_root/ionosphere_data")
OUTPUT_DIR = Path("./data_root/ionosphere_data/all_maps")


def count_zip_files(zip_path: Path) -> int:
    result = subprocess.run(["unzip", "-l", str(zip_path)], capture_output=True, text=True)
    return sum(1 for line in result.stdout.splitlines() if ".npy" in line)


def unzip_file(args):
    zip_path, output_dir, position = args
    zip_path, output_dir = Path(zip_path), Path(output_dir)
    total = count_zip_files(zip_path)

    with tqdm(total=total, desc=zip_path.stem, position=position, leave=True, unit="file") as pbar:
        proc = subprocess.Popen(
            ["unzip", "-j", "-o", str(zip_path), "-d", str(output_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            if "inflating:" in line or "extracting:" in line:
                pbar.update(1)
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"unzip failed for {zip_path}")
    return zip_path.stem, total


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIPS_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files")
    print(f"Output: {OUTPUT_DIR}\n")

    unzip_args = [(str(z), str(OUTPUT_DIR), i) for i, z in enumerate(zip_files)]

    with Pool(processes=max(len(zip_files), 1)) as pool:
        unzip_results = [pool.apply_async(unzip_file, (a,)) for a in unzip_args]
        for r in unzip_results:
            r.get()

    print(f"\nDone. Total .npy files: {len(list(OUTPUT_DIR.glob('*.npy')))}")
