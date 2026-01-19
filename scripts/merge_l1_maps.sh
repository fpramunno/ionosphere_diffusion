#!/bin/bash
#SBATCH --job-name=l1_map_merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -A sk035
#SBATCH --output=/users/framunno/logs/out/out_l1_map_merge.log
#SBATCH --error=/users/framunno/logs/err/err_l1_map_merge.log

# =============================================================================
# L1-to-Map Matching Script
# =============================================================================
# This script matches L1 solar wind measurements with ionosphere maps,
# accounting for propagation delay from L1 to Earth.
#
# Resources:
#   - 16 CPUs (for parallel processing with 8 workers + overhead)
#   - 32 GB RAM (for loading large CSV files)
#   - 4 hours (should be enough for full dataset)
# =============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "=========================================="
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 32G"
echo "=========================================="

# =============================================================================
# Environment setup
# =============================================================================
echo "Activating Python environment..."
source /users/framunno/envs/ionosphere/bin/activate

# Set Python to unbuffered mode for real-time logging
export PYTHONUNBUFFERED=1

# Create log directories if they don't exist
mkdir -p /users/framunno/logs/out
mkdir -p /users/framunno/logs/err

# =============================================================================
# Run the matching script
# =============================================================================
echo "=========================================="
echo "Starting L1-to-Map matching..."
echo "=========================================="

python /users/framunno/projects/ionosphere_diffusion/merge_l1_to_maps_even_minutes.py

# Capture exit code
EXIT_CODE=$?

# =============================================================================
# Job completion
# =============================================================================
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Matching completed successfully!"
    echo "Output file: /users/framunno/data/ionosphere/l1_to_map_matched_even_minutes.csv"
else
    echo "✗ ERROR: Matching failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
