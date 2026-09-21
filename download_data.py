"""
Download ionosphere data files from Google Drive.
Files are saved to scratch storage to avoid filling up /vast quota.
"""

import gdown
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("/capstor/scratch/cscs/framunno/ionosphere_data")

GDRIVE_LINKS = [
    "https://drive.google.com/file/d/1ZxWdT0s2xEb2vI6pQV7H6vcTPpNrcZ7b/view?usp=drive_link",
    "https://drive.google.com/file/d/1WpGvwG6nKfhjili2rNZKVys4lTDxafd4/view?usp=drive_link",
    "https://drive.google.com/file/d/1MU9pCiBjb61nIazAztX8smrzTBrya88P/view?usp=drive_link",
    "https://drive.google.com/file/d/1gp7AvFv_-gpZLHKgeaxlFjmIUcE8zKAu/view?usp=drive_link",
    "https://drive.google.com/file/d/1_eqG-HOY8FBz-cOh3m8qyNc28-eVRXWY/view?usp=drive_link",
]


def download_all(links, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for url in tqdm(links):
        print(f"\nDownloading: {url}")
        gdown.download(url, output=str(output_dir) + "/", quiet=False, fuzzy=True)
    print(f"\nDone. Files saved to {output_dir}")


if __name__ == "__main__":
    download_all(GDRIVE_LINKS, OUTPUT_DIR)
