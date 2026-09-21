# Evaluation Strategy — 2015 St. Patrick's Day Storm

## Event Overview

The March 17, 2015 geomagnetic storm (dubbed "St. Patrick's Day Storm") is the most intense storm
of Solar Cycle 24 (Dst_min ≈ −223 nT). It is an ideal benchmark for ionospheric forecasting models
because:
- It is well-documented in the literature with clear phase transitions
- ACE solar wind data is available with real orbit positions (X ≈ 1.54 × 10⁶ km)
- It falls outside the 2020–2025 training window → genuine out-of-sample evaluation

### Key Phase Timestamps (UT, March 17 2015)

| Phase | Time | Indicator |
|---|---|---|
| SSC onset | ~04:45 | Sudden jump in solar wind dynamic pressure (interplanetary shock arrival); reflected as sudden H-component increase in ground magnetograms |
| Main phase onset | ~06:00 | Bz_GSM turns strongly and persistently southward in ACE data |
| Main phase duration | 06:00 → 22:00 | ~16 h of sustained southward Bz |
| Storm peak (Dst min) | ~22:00 | Dst ≈ −223 nT (external index, not in solar wind data) |
| Recovery phase | Mar 17 22:00 → Mar 18+ | Bz returns northward |

> **Source of phase timestamps**: Published literature on this well-studied event (e.g.,
> Cherniak et al. 2015, Liu et al. 2015). SSC onset is an official ISGI bulletin entry.
> The Bz southward transition and SSC are independently verifiable from the ACE CSV
> (`bz_gsm` and `speed_kms` columns — see §Detection from solar wind below).

### Detection from solar wind data (bx, by, bz, v_wind)

You **can** approximate all key transitions from the ACE CSV alone:

| Phase | Signal in ACE data | Limitation |
|---|---|---|
| SSC / shock arrival | Sudden step-up in `speed_kms` (typically +100–200 km/s jump) **and** a spike in dynamic pressure ∝ n·v² (density not in our CSV, but speed jump alone is visible) | ±15 min uncertainty due to propagation delay estimation |
| Main phase onset | `bz_gsm` crosses below ~ −10 nT and stays negative | Direct, reliable |
| Recovery onset | `bz_gsm` returns toward 0 / positive | Direct |
| Dst minimum | **Not derivable from solar wind alone** — requires ground magnetometer Dst index | External dataset needed |

---

## Model Constraints

| Parameter | Value |
|---|---|
| `sequence_length` | 50 frames |
| Frame cadence | 2 min |
| Total window | 100 min |
| Conditioning frames | 15 frames = 30 min |
| Predicted frames | 35 frames (5 AR steps × 7 frames/step) = 70 min |
| First predicted frame offset from sequence center | +20 min |
| Autoregressive steps | 5 |

Each AR step extends prediction by 7 × 2 min = **14 min**.
After 5 steps the model has predicted **70 min ahead** of the last conditioning frame.

---

## Generation Strategy

### Why multiple PRED_START instead of one long sequence?

The storm main phase spans ~16 hours. A single AR rollout of 70 min covers only ~7% of the storm.
Options considered:

1. **One very long AR rollout** (e.g., sequence_length=550 → ~17 h): covers the full storm but
   error compounds over 77 AR steps — spatial coherence degrades rapidly and the result is
   scientifically uninterpretable.

2. **Overlapping short windows (chosen approach)**: Run generation at multiple `PRED_START` points,
   each producing a clean 70-min window. Together they tile the storm timeline.
   - Error is bounded per window (max 5 AR steps)
   - The overlap between windows allows consistency checks
   - For the diffusion models (probabilistic), 10 samples per window gives uncertainty quantification

### Chosen PRED_START Points

Selected to capture each distinct storm phase with at least one window:

| PRED_START | Phase captured | Scientific motivation |
|---|---|---|
| `2015-03-17 02:00` | Pre-storm quiet | Baseline for model drift / false-alarm rate |
| `2015-03-17 04:45` | SSC onset | Can the model respond to the interplanetary shock? |
| `2015-03-17 06:00` | Main phase onset | Response to sustained southward Bz |
| `2015-03-17 10:00` | Mid main phase | Sustained storm-time convection |
| `2015-03-17 16:00` | Late main phase | Approaching peak |
| `2015-03-17 20:00` | Near Dst minimum | Maximum storm intensity |

Each window: conditioning on frames t−30min…t, prediction for t…t+70min.

### Why keep sequence_length=50?

- The model was trained with `sequence_length=50`; changing this at inference risks
  distributional shift in positional encodings.
- 70 min of prediction is physically meaningful: it covers the initial ionospheric
  response to a solar wind impulse (typical convection pattern adjustment: 20–40 min).
- Error accumulation with 5 AR steps is already non-trivial; going to 7–10 steps
  would likely wash out storm-phase structure.

---

## Scripts

- `scripts/generate_2015_diffusion.sh` — CLASSIC + NOCOND diffusion models, 10 samples each
- `scripts/generate_2015_unet.sh` — deterministic UNet baseline
- Change `PRED_START` variable in each script to run different windows.

## Data Files

- Solar wind (L1): `/users/framunno/data/ionosphere/combined_ACE_1min_gsm_2015.csv`
  - Interpolated up to MAX_GAP=34 min, 0 NaN in Mar 10–20 window
- Matched pairs: `/users/framunno/data/ionosphere/l1_to_map_matched_2015_march_10_20_deduplicated.csv`
  - 7,493 rows, one per even-minute map, `proton_vx_gsm = -speed_kms` added
- Maps directory: `/capstor/scratch/cscs/framunno/ionosphere_data/event_maps/2015_march_10_20/`
