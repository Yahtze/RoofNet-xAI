#!/bin/bash
# ============================================================
# Slurm job for NYU Torch HPC
# Generate attribution heatmap overlays for curated subsets
#
# Submit:  sbatch hpc/run_attribution_overlays.sbatch
# Monitor: squeue --me
# Cancel:  scancel -u yd3288
# ============================================================
#SBATCH --job-name=attr_overlay
#SBATCH --account=<YOUR_SLURM_ACCOUNT>   # run: my_slurm_accounts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00                   # 30 min (166 images @ ~2s each)
#SBATCH --output=/scratch/yd3288/RoofNet-xAI/xAI_outputs/segmentation/logs/attr_overlay_%j_out.txt
#SBATCH --error=/scratch/yd3288/RoofNet-xAI/xAI_outputs/segmentation/logs/attr_overlay_%j_err.txt

set -euo pipefail

REPO_DIR="/scratch/yd3288/RoofNet-xAI"
CONDA_ENV_PATH="/scratch/yd3288/conda-envs/roofnet"
SCRIPT="$REPO_DIR/scripts/generate_attribution_overlays.py"

# --- Load environment ---
export HOME="/scratch/yd3288/home"
export TMPDIR="/scratch/yd3288/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR="/scratch/yd3288/pip-cache"
export CONDA_PKGS_DIRS="/scratch/yd3288/conda-pkgs"
export CONDA_ENVS_PATH="/scratch/yd3288/conda-envs"
export CONDARC="/scratch/yd3288/condarc"
export HF_HOME="/scratch/yd3288/hf-cache"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"
mkdir -p "$HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$HF_HOME"

set +u
module purge
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PATH"
set -u

# --- Verify GPU ---
echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
echo ""

# --- Pre-flight ---
if [ ! -f "$REPO_DIR/best_clip_model_balanced.pth" ]; then
    echo "ERROR: Missing model weights"
    exit 1
fi

# --- Run ---
python "$SCRIPT" \
    --subsets curated_top100 curated_top66 \
    --device cuda \
    --model-weights "$REPO_DIR/best_clip_model_balanced.pth"

echo ""
echo "=== Job finished: $(date) ==="

# --- Report outputs ---
for subset in curated_top100 curated_top66; do
    dir="$REPO_DIR/xAI_outputs/segmentation/$subset/attribution_overlays"
    if [ -d "$dir" ]; then
        count=$(ls "$dir" | wc -l)
        echo "$subset: $count attribution overlays generated"
    fi
done
