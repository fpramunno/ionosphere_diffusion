#!/bin/bash
# Reads storm_phase_timestamps_2015.json and submits one SLURM job per
# PRED_START window for both diffusion (CLASSIC+NOCOND) and UNet.
# Usage: bash launch_2015_all.sh [diffusion|unet|all]  (default: all)

TIMESTAMPS_JSON="/users/framunno/data/ionosphere/storm_phase_timestamps_2015.json"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-all}"

mapfile -t PRED_STARTS < <(python3 -c "
import json
with open('$TIMESTAMPS_JSON') as f:
    d = json.load(f)
for ts in d['pred_starts'].values():
    print(ts[:16].replace('T', ' '))
")

echo "=== 2015 storm evaluation launcher ==="
echo "Model : $MODEL"
echo "Windows (${#PRED_STARTS[@]}):"
for ps in "${PRED_STARTS[@]}"; do echo "  $ps"; done
echo ""

for PRED_START in "${PRED_STARTS[@]}"; do
    TAG=$(echo "$PRED_START" | tr ' ' 'T' | tr ':' 'h')

    if [[ "$MODEL" == "diffusion" || "$MODEL" == "all" ]]; then
        JOB=$(sbatch --job-name="diff_${TAG}" \
                     "$SCRIPTS_DIR/generate_2015_diffusion.sh" "$PRED_START" | awk '{print $4}')
        echo "Submitted DIFFUSION  $PRED_START → job $JOB"
    fi

    if [[ "$MODEL" == "unet" || "$MODEL" == "all" ]]; then
        JOB=$(sbatch --job-name="unet_${TAG}" \
                     "$SCRIPTS_DIR/generate_2015_unet.sh" "$PRED_START" | awk '{print $4}')
        echo "Submitted UNET       $PRED_START → job $JOB"
    fi
done

echo ""
echo "Check with: squeue -u $USER"
