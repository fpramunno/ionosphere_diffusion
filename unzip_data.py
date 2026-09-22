"""
Fast parallel unzip of ionosphere data + copy of 2024 files.
Uses multiprocessing so each task runs in its own process with a live progress bar.
"""

import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

ZIPS_DIR = Path("./data_root/ionosphere_data")
SOURCE_2024 = Path("./data/ionosphere/ionosphere_data/pickled_maps")
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


def copy_2024(args):
    source, output_dir, position = args
    source, output_dir = Path(source), Path(output_dir)
    files = sorted(source.glob("map_2024_*"))

    with tqdm(total=len(files), desc="2024 copy", position=position, leave=True, unit="file") as pbar:
        for f in files:
            shutil.copy2(f, output_dir / f.name)
            pbar.update(1)
    return len(files)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIPS_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files + 2024 directory to copy")
    print(f"Output: {OUTPUT_DIR}\n")

    unzip_args = [(str(z), str(OUTPUT_DIR), i) for i, z in enumerate(zip_files)]
    copy_args = (str(SOURCE_2024), str(OUTPUT_DIR), len(zip_files))

    with Pool(processes=len(zip_files) + 1) as pool:
        unzip_results = [pool.apply_async(unzip_file, (a,)) for a in unzip_args]
        copy_result = pool.apply_async(copy_2024, (copy_args,))

        for r in unzip_results:
            r.get()
        copy_result.get()

    print(f"\nDone. Total .npy files: {len(list(OUTPUT_DIR.glob('*.npy')))}")
