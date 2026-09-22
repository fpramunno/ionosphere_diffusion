# Ionosphere Convection Forecasting

Code for the paper *"Polar Cap Potential Patterns Forecasting in the Southern
Hemisphere with Deep Learning Techniques"*: a ViT-based diffusion model
(CLASSIC), an unconditioned diffusion ablation (NOCOND), and a deterministic
UNet baseline, forecasting high-latitude ionospheric convection from L1 solar
wind / IMF measurements, trained on SuperDARN-derived electric potential maps.

## Repository layout

```
src/                    Model/data library (ViT, diffusion, UNet, dataset)
training_pred.py         Train the diffusion model (CLASSIC or NOCOND)
training_unet.py         Train the deterministic UNet baseline
generate_data_v4.py       Run inference: single-pass, autoregressive rollout, or storm case study
compute_metrics_v2.py     Per-frame pixel metrics (used by v3)
compute_metrics_v3.py     Metrics on saved inference output (singlepass / AR / storm)
evaluation/eval_model.py  Standalone evaluation metrics (PSNR/SSIM/NRMSE/CPCP/...)
configs/                  Model + accelerate configs (JSON/YAML)
scripts/                  sbatch launchers for all of the above
notebooks/                Notebooks that reproduce the paper's figures
Data preparation (see below): download_data.py, unzip_data.py,
  interpolate_l1_gaps.py, merge_l1_to_maps_even_minutes.py,
  merge_l1_to_maps_2015_event.py, merge_data.py, precompute_dynamics_scores.py
```

## Environment setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` assumes `torch`/`torchvision` are already provided (this
was developed against CSCS's `pytorch/v2.6.0` uenv) — install them yourself
first if your system doesn't already have them.

### Configuring paths

Scripts under `scripts/` reference these environment variables instead of a
hardcoded location — set the ones you need before submitting a job:

| Variable | Used for |
|---|---|
| `IONO_DATA_ROOT` | Large data / results storage (checkpoints, generated samples, logs) |
| `IONO_FAST_ROOT` | Faster/smaller scratch tier, if your cluster has one (falls back to `IONO_DATA_ROOT`-like usage otherwise) |
| `IONO_VENV` | Path to the Python venv created above |
| `IONO_REPO` | Path to this repo checkout |
| `IONO_HOME_ROOT` | Home-directory-scale storage (small CSVs, logs) |
| `IONO_MAPS_DIR` | Directory containing the `.npy` ionosphere map files (read by `src/data/dataset.py`) |

Python scripts (`training_pred.py`, `generate_data_v4.py`, etc.) take the
equivalent paths as CLI flags (`--data-path`, `--saving-path`, `--csv-path`,
...) with relative placeholder defaults — pass your own paths explicitly
rather than relying on the defaults.

**Note on `#SBATCH` directives:** `--output`/`--error` in the `scripts/*.sh`
launchers use paths relative to the submission directory (`./logs/...`) —
`#SBATCH` comment directives don't reliably expand shell variables, so
`mkdir -p logs/out logs/err` before submitting. `#SBATCH -A <account>` and
`--partition=<name>` are specific to the original cluster (CSCS Alps) and
must be edited to your own account/partition.

## Pipeline

### 1. Data preparation and L1-to-map pairing

Raw source data: 1-minute DSCOVR solar wind/IMF measurements for 2020-2025
and DSCOVR orbit (position) data (both from NOAA/NASA SPDF), and 1-minute
ACE solar wind/IMF measurements for the March 2015 case study (NASA), as
described in the paper's Data section. Both DSCOVR and ACE sit at the L1
Lagrangian point.

`download_data.py` fetches everything from Google Drive in two groups:
- `MAP_ARCHIVE_LINKS` → `IONO_DATA_ROOT/ionosphere_data` (map `.zip` archives)
- `L1_DATA_LINKS` → `./data/ionosphere` (L1/orbit CSVs, including the
  already-paired, ready-to-train `l1_to_map_matched_*.csv`)

gdown preserves each file's original Drive filename — check the downloaded
filenames against what `merge_l1_to_maps_even_minutes.py` and
`merge_l1_to_maps_2015_event.py` expect (their `SOLAR_WIND_FILE`/
`DSCOVR_FILE`/`L1_FILE`/`OUTPUT_FILE` constants) and rename/edit as needed.

**Quick path** (just want to train/reproduce): `download_data.py` already
includes a ready-to-use, already-paired-and-deduplicated matched CSV — unzip
the maps (step 2 below) and point `--csv-path` at that file. No merging
needed.

**From-scratch path** (reproducing the pairing itself, or extending to new
years): steps 2-8 below.

1. `unzip_data.py` — parallel unzip of the map archives into `IONO_MAPS_DIR`.
2. `interpolate_l1_gaps.py` — linearly interpolate short gaps in each raw
   per-year L1 CSV (`--help` for max-gap-duration options). Run once per year.
3. `combine_l1_years.py` — concatenate the per-year interpolated CSVs into a
   single multi-year file (`--inputs year1.csv year2.csv ... --output
   combined.csv`).
4. `merge_l1_to_maps_even_minutes.py` — the actual pairing step: for each L1
   measurement (~1 min cadence), computes when it physically arrives at Earth
   (propagation delay, using the real DSCOVR/ACE position — not a fixed L1
   distance) and matches it to the nearest ionosphere map (~2 min cadence) in
   time. Edit the `SOLAR_WIND_FILE`/`DSCOVR_FILE`/`OUTPUT_FILE` constants at
   the top of the script to point at your paths. Because maps are coarser
   than L1, several L1 rows legitimately match the same map, so the raw
   output has duplicate map filenames.
5. `merge_l1_to_maps_2015_event.py` — same pairing logic, restricted to the
   March 2015 St. Patrick's Day storm window used for the out-of-distribution
   case study (edit `L1_FILE`/`OUTPUT_FILE` at the top). Uses ACE, whose
   position is already a column in the raw file (no separate orbit file
   needed for this one).
6. `deduplicate_matched_pairs.py` — collapses step 4/5's output to one row
   per unique map (keeps the closest-matching L1 measurement per map). This
   deduplicated CSV is what every training/generation script consumes via
   `--csv-path` — this is the file the quick path above downloads directly.
7. `precompute_dynamics_scores.py` — optional but recommended before
   generation: caches a per-sequence "dynamics score" (mean frame-to-frame
   change) so `--dynamics-filter`/`--activity-filter` don't recompute it from
   scratch at dataset init.

### 2. Training

```bash
# Diffusion model
sbatch scripts/training_script_ddp.sh
# Deterministic UNet baseline
sbatch scripts/training_script_ddp_unet.sh
```

Both call `training_pred.py` / `training_unet.py` respectively via
`accelerate launch`, using `configs/accelerate_config_ddp.yaml` and the model
config `configs/forecast_iono_15_big_cosine_solar_classic.json`.

**CLASSIC vs. NOCOND is a CLI flag, not a config file**: `training_pred.py`
takes `--no-mapping-cond` — include it to train the unconditioned ablation
(NOCOND), omit it to train the L1-conditioned model (CLASSIC). As committed,
`scripts/training_script_ddp.sh` passes `--no-mapping-cond` (trains NOCOND)
— remove that flag from the `accelerate launch` invocation to train CLASSIC
instead. The same flag/logic applies to `generate_data_v4.py` at inference
time (`NO_MAPPING_COND` variable in `scripts/generate_data_v4.sh`).

### 3. Generation / inference

`generate_data_v4.py` handles all three evaluation regimes from the paper;
see the docstring at the top of the file for full flag documentation.

```bash
# Single-pass, matching the training setup
sbatch scripts/generate_data_v4.sh            # diffusion
sbatch scripts/generate_data_v4_unet.sh       # UNet

# Extended autoregressive rollout (--autoregressive, longer --sequence-length)
# — same scripts, edit MODE_TAG/SEQUENCE_LENGTH inside

# Out-of-distribution 2015 storm case study
sbatch scripts/generate_2015_diffusion.sh     # CLASSIC + NOCOND
sbatch scripts/generate_2015_unet.sh
# or launch all three at once:
sbatch scripts/launch_2015_all.sh
```

### 4. Evaluation

```bash
sbatch scripts/compute_metrics_v3.sh          # singlepass results
sbatch scripts/compute_metrics_v3_ar3.sh      # short AR rollout
sbatch scripts/compute_metrics_v3_ark50.sh    # extended AR rollout (K=50)
```

`compute_metrics_v3.py` reuses `compute_metrics_v2.py`'s per-frame pixel
metrics (PSNR/SSIM/LPIPS/power-spectrum) and adds NRMSE of scalar summary
statistics (mean/std/CPCP), reported per-sample for multi-sample (diffusion)
results. See the docstring at the top of the file for the exact CSV outputs.
`evaluation/eval_model.py` provides a standalone/alternative metrics pass
(PSNR/SSIM/relative-L2/NRMSE/CPCP).

Paper figures (2015 storm timeseries, AR schematic, regime examples,
training-strategy diagram) are reproduced in `notebooks/`.

## Verifying your setup

```bash
python3 -c "import src.vit, src.diffusion, src.unet_simple, src.data.dataset"
```
should import without error once `requirements.txt` is installed.
