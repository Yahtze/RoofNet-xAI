#!/bin/bash

# ==============================================================================
# RoofNet Automated Execution Pipeline
# ==============================================================================
# This script automates the downloading, cropping, metadata generation, and 
# classification of roofing images.
#
# Usage: ./roofnet_execution.sh <path_to_coordinates.csv> <path_to_venv>
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# --- 0. ARGUMENT VALIDATION ---
# Check if exactly one argument was passed
if [ "$#" -ne 2 ]; then
    echo "❌ Error: Incorrect number of arguments."
    echo "Usage: ./roofnet_execution.sh <path_to_coordinates.csv> <path_to_venv>"
    exit 1
fi

INPUT_COORDS_CSV="$1"
VENV_PATH="$2"

# Check if the provided file actually exists on the filesystem
if [ ! -f "$INPUT_COORDS_CSV" ]; then
    echo "❌ Error: The coordinates CSV file '$INPUT_COORDS_CSV' does not exist."
    exit 1
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Error: The virtual environment directory '$VENV_PATH' does not exist."
    exit 1
fi

echo "========================================"
echo "🚀 Starting RoofNet Pipeline"
echo "📂 Input CSV: $INPUT_COORDS_CSV"
echo "========================================"

# --- 1. VIRTUAL ENVIRONMENT ACTIVATION ---

echo "-> Targeting environment at ${VENV_PATH}..."

# Instead of relying on fragile activation scripts in non-interactive bash,
# we force the environment's binary folder to the absolute front of the PATH.
if [ -d "${VENV_PATH}/bin" ]; then
    export PATH="${VENV_PATH}/bin:$PATH"
    
    # Optional: Set CONDA_PREFIX so conda-aware libraries know where they are
    export CONDA_PREFIX="${VENV_PATH}"
    
    echo "Environment path injected successfully."
else
    echo "❌ Error: Could not find 'bin' directory inside ${VENV_PATH}"
    exit 1
fi

# Verification
ACTIVE_PY=$(which python3)
echo "Active Python: $ACTIVE_PY"

# Safety check to ensure we aren't using the Homebrew or System Python
if [[ "$ACTIVE_PY" == *"/opt/homebrew/"* ]] || [[ "$ACTIVE_PY" == "/usr/bin/"* ]]; then
    echo "❌ Error: Still pointing to system Python. Path injection failed."
    exit 1
fi

# --- 2. PIPELINE CONFIGURATION ---
INPUT_COORDS_DIR=$(dirname "$INPUT_COORDS_CSV")
MODEL_WEIGHTS="resources/best_clip_model_balanced.pth"
GSAT_DIR="roofnet_gsat_imagery"
GSAT_CROPPED_DIR="roofnet_gsat_imagery_cropped"
METADATA_AUGMENTED_CSV="roof_materials_augmented_osm.csv"
CLASSIFIED_DIR="classified_roofs"


# --- 3. DOWNLOAD IMAGES ---
echo ""
echo "-> STEP 1: Downloading satellite images via Google Static Maps API..."
# Pass the required CSV argument to the download script
# (Make sure download_from_csv.py is updated to accept this via argparse or sys.argv!)
python3 download_preprocess/download_from_csv.py \
    --csv_file "$INPUT_COORDS_CSV"

# --- 4. CROP IMAGES ---
echo ""
echo "-> STEP 2: Cropping images using Grounding DINO (RoofView filter)..."
python3 download_preprocess/roof_view.py \
    --input_dir "$GSAT_DIR"

# --- 5. ADD METADATA ---
echo ""
echo "-> STEP 3: Generating geographical and structural metadata..."
python3 download_preprocess/generate_roof_material_metadata.py \
    --dataset_dir "$GSAT_CROPPED_DIR" \
    --building_csv "$INPUT_COORDS_CSV" \
    --out_dir "${INPUT_COORDS_DIR}"


# --- 6. CLASSIFY IMAGES ---
echo ""
echo "-> STEP 4: Classifying roof materials using fine-tuned RemoteCLIP..."
python3 training_evaluation/remoteclip_classify.py \
    --classification_dir "$GSAT_CROPPED_DIR" \
    --model_weights "$MODEL_WEIGHTS" \
    --metadata_csv "${INPUT_COORDS_DIR}/${METADATA_AUGMENTED_CSV}"

# --- 7. CLASSIFY IMAGES ---
echo ""
echo "-> STEP 5: Verifying RemoteCLIP classifications..."
python3 training_evaluation/verify_material_classes.py \
    --dataset_dir "$CLASSIFIED_DIR" \
    --building_csv "${INPUT_COORDS_DIR}/${METADATA_AUGMENTED_CSV}"

echo ""
echo "========================================"
echo "✅ Pipeline completed successfully!"
echo "========================================"

# Optional: Deactivate the virtual environment
deactivate