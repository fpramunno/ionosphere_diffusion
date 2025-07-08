from pathlib import Path

BASE_DATA_PATH = Path("/mnt/home/rmorel/ceph/data/solar")
BASE_PM_DATA_PATH = Path("/mnt/home/polymathic/ceph/solar")  # for storing data on polymathic ceph


####### PATHS USED FOR STORING TEMPORARY FILES #######
FULL_DISK_DATA_PATH = BASE_PM_DATA_PATH / "full_disk"
# HMI data paths
HMI_DATA_PATH = FULL_DISK_DATA_PATH / "hmi"
HMI_DIFFROT_DATA_PATH = FULL_DISK_DATA_PATH / "hmi_diff_rot"
HMI_META_DATA = FULL_DISK_DATA_PATH / "hmi_meta"

# AIA data paths
AIA_DATA_PATH = FULL_DISK_DATA_PATH / "aia"

####### FINAL DATASET PATHS #######
HMI_ZOOMED_DATA_PATH = BASE_PM_DATA_PATH / "hmi_zoomed"
SUN_REGION_DATA_PATH = BASE_PM_DATA_PATH / "sun_regions"

####### EVALUATION #######
# models prediction data
EVAL_PATH = BASE_PM_DATA_PATH / "evaluation"
# visualization
WEBSITE_PATH = Path("/mnt/home/rmorel/public_www")
VISU_PRIVATE_PATH = WEBSITE_PATH / "eval_jGmebwyas9y7wYtp"
VISU_PUBLIC_PATH = WEBSITE_PATH / "external_bYHQ9y9yhw42ZgJD"
# deprecated visualization
SOLAR_VIS_PATH = Path("/mnt/home/rmorel/ceph/data/solar/visualization")