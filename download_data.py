"""
Download ionosphere data files from Google Drive.

Map/event archives go to MAPS_OUTPUT_DIR (input for unzip_data.py).
L1 solar wind / IMF / orbit CSVs go to L1_OUTPUT_DIR (the location
merge_l1_to_maps_even_minutes.py / merge_l1_to_maps_2015_event.py expect
via their SOLAR_WIND_FILE/DSCOVR_FILE/L1_FILE constants).

gdown preserves each file's original Google Drive filename on download —
after downloading, check that filenames match what those scripts' hardcoded
constants expect, and rename/edit as needed.
"""

import gdown
from pathlib import Path
from tqdm import tqdm

MAPS_OUTPUT_DIR = Path("./data_root/ionosphere_data")
L1_OUTPUT_DIR = Path("./data/ionosphere")

MAP_ARCHIVE_LINKS = [
    "https://drive.google.com/file/d/1ZxWdT0s2xEb2vI6pQV7H6vcTPpNrcZ7b/view?usp=drive_link",
    "https://drive.google.com/file/d/1n467WrWE91RwJ_LnoNiK5cayZpmgsRj1/view?usp=drive_link",
    "https://drive.google.com/file/d/1WpGvwG6nKfhjili2rNZKVys4lTDxafd4/view?usp=drive_link",
    "https://drive.google.com/file/d/1MU9pCiBjb61nIazAztX8smrzTBrya88P/view?usp=drive_link",
    "https://drive.google.com/file/d/1gp7AvFv_-gpZLHKgeaxlFjmIUcE8zKAu/view?usp=drive_link",
    "https://drive.google.com/file/d/1_eqG-HOY8FBz-cOh3m8qyNc28-eVRXWY/view?usp=drive_link",

    # 2015 EVENT
    "https://drive.google.com/file/d/1YQrC-OwYvF3JArtsCJ2qf6562nhZuBxs/view?usp=drive_link",
]

L1_DATA_LINKS = [
    # DSCOVR orbit (position), 2020-2025 combined -- DSCOVR_ORBIT_PRE_2020_2025.csv
    "https://drive.google.com/file/d/1A5VwgKGFq9fU2X1dpXZ2z4nDTQeYNhgo/view?usp=sharing",

    # L1-to-map matched pairs (post-pairing, ready for --csv-path)
    "https://drive.google.com/file/d/17C-ljKrgdwhx8H2fN8rxRq5qewhlBRBn/view?usp=sharing",

    # ACE 2015 (VWind/Bx/By/Bz + position, for the March 2015 case study)
    "https://drive.google.com/file/d/1kZ4UEjwdEfqhG1NC8mOTkqV7cERWrPpv/view?usp=sharing",

    # DSCOVR solar wind / IMF (VWind, Bx, By, Bz), 2020-2025 combined + interpolated
    "https://drive.google.com/file/d/1UZaP14rHRZVFVPF7bDCMRyeknzjyMHEr/view?usp=sharing",
]


def download_all(links, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for url in tqdm(links):
        print(f"\nDownloading: {url}")
        gdown.download(url, output=str(output_dir) + "/", quiet=False, fuzzy=True)
    print(f"\nDone. Files saved to {output_dir}")


if __name__ == "__main__":
    download_all(MAP_ARCHIVE_LINKS, MAPS_OUTPUT_DIR)
    download_all(L1_DATA_LINKS, L1_OUTPUT_DIR)
