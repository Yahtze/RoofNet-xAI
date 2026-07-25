"""
infer_chipped_dataset.py
========================
Run frozen RemoteCLIP inference on the chipped_roof_material_classification
dataset and evaluate against ground-truth folder labels.

Uses the MATERIAL_DESCRIPTIONS prompts from training (not city-conditioned
prompts) to match what the model was fine-tuned on.

Progress-aware: writes a JSON manifest so the script can be restarted after
failures without re-processing completed images.

Usage:
    python infer_chipped_dataset.py [--reset]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "chipped_roof_material_classification"
MODEL_WEIGHTS = REPO_ROOT / "best_clip_model_balanced.pth"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

IMAGE_EXTS = {".tif", ".tiff"}

# 15-class material descriptions matching training prompts
MATERIAL_DESCRIPTIONS = {
    "Thatch": "thatched roof (dried grasses / straw or palm)",
    "GreenVegetative": "roof with vegetation on it",
    "StoneSlates": "dark stone slate roof",
    "ClayTiles": "tiled clay / tiled ceramic roof",
    "AsphaltTiles": "angled asphalt shingle roof",
    "ConcreteTiles": "tiled concrete / tiled cement roof",
    "WoodTiles": "wood shingle roof",
    "MetalSheetMaterials": "corrugated or tiled metal roof (silver / dark / painted)",
    "PolycarbonateSheetMaterials": "polycarbonate roof",
    "GlassSheetMaterials": "glass roof (clear or mirrored)",
    "AmorphousConcrete": "flat concrete roof",
    "AmorphousAsphalt": "asphalt-coated roof (bitumen layer or rolled roofing)",
    "AmorphousMembrane": "membrane roof (bright EPDM/TPO)",
    "AmorphousFabric": "tensile fabric roof (PVC / PTFE / canvas)",
    "Unknown": "unknown material, image may be too low resolution or obstructed",
}

MATERIAL_CLASSES = list(MATERIAL_DESCRIPTIONS.keys())
MATERIAL_PROMPTS = list(MATERIAL_DESCRIPTIONS.values())

# Ground-truth folder name -> training class mapping
GT_TO_TRAINING = {
    "metal_sheet": "MetalSheetMaterials",
    "thatch": "Thatch",
    "other": "Unknown",
    "plastic": "PolycarbonateSheetMaterials",
}

# Inverse for reporting: training class -> new class (None if no mapping)
TRAINING_TO_GT = {v: k for k, v in GT_TO_TRAINING.items()}

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

MANIFEST_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("manifest_version", MANIFEST_VERSION)
        data.setdefault("created_at", utc_now_iso())
        data.setdefault("updated_at", data["created_at"])
        data.setdefault("jobs", {})
        return data
    now = utc_now_iso()
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": now,
        "updated_at": now,
        "jobs": {},
    }


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = utc_now_iso()
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(manifest, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------


def discover_images(image_dir: Path) -> list[dict]:
    """Walk class folders and return list of {path, gt_class, gt_training_class}."""
    images = []
    for class_folder in sorted(image_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        gt_class = class_folder.name
        gt_training = GT_TO_TRAINING.get(gt_class)
        if gt_training is None:
            print(f"WARNING: Unknown class folder '{gt_class}', skipping.")
            continue
        for img_path in sorted(class_folder.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                images.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "gt_class": gt_class,
                    "gt_training_class": gt_training,
                })
    return images


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    images: list[dict],
    model,
    tokenizer,
    preprocess,
    device: str,
    manifest_path: Path,
    results_csv_path: Path,
):
    """Run inference on all images, writing results incrementally to CSV."""
    manifest = load_manifest(manifest_path)

    # Determine which images still need processing
    pending = []
    for img_info in images:
        job_id = img_info["filename"]
        job = manifest["jobs"].get(job_id)
        if job and job.get("status") == "done":
            continue
        pending.append(img_info)

    already_done = len(images) - len(pending)
    print(f"Total images: {len(images)}, already done: {already_done}, remaining: {len(pending)}")

    if not pending:
        print("All images already processed.")
        return

    # Write header if CSV doesn't exist yet
    csv_header_written = results_csv_path.exists() and results_csv_path.stat().st_size > 0

    with tqdm(total=len(pending), desc="Inference", unit="img") as pbar:
        for img_info in pending:
            job_id = img_info["filename"]
            img_path = img_info["path"]

            # Mark running
            manifest["jobs"][job_id] = {
                "job_id": job_id,
                "status": "running",
                "started_at": utc_now_iso(),
            }
            save_manifest(manifest_path, manifest)

            try:
                # Load and preprocess image
                pil_img = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(pil_img).unsqueeze(0).to(device)

                # Build prompts
                tokenized = tokenizer(MATERIAL_PROMPTS).to(device)

                # Forward pass
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    text_features = model.encode_text(tokenized)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logits = 100.0 * image_features @ text_features.T
                    probs = logits.softmax(dim=-1).squeeze(0)

                top_idx = probs.argmax().item()
                top_class = MATERIAL_CLASSES[top_idx]
                top_confidence = probs[top_idx].item()

                # Write result row
                row = {
                    "filename": img_info["filename"],
                    "gt_class": img_info["gt_class"],
                    "gt_training_class": img_info["gt_training_class"],
                    "predicted_class": top_class,
                    "confidence": round(top_confidence, 6),
                }
                # Add all 15 class probabilities
                for i, cls in enumerate(MATERIAL_CLASSES):
                    row[f"prob_{cls}"] = round(probs[i].item(), 6)

                # Append to CSV
                write_header = not csv_header_written
                pd.DataFrame([row]).to_csv(
                    results_csv_path,
                    mode="a",
                    header=write_header,
                    index=False,
                )
                csv_header_written = True

                # Mark done
                manifest["jobs"][job_id] = {
                    "job_id": job_id,
                    "status": "done",
                    "finished_at": utc_now_iso(),
                }
                save_manifest(manifest_path, manifest)

            except Exception as exc:
                manifest["jobs"][job_id] = {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(exc),
                    "finished_at": utc_now_iso(),
                }
                save_manifest(manifest_path, manifest)
                tqdm.write(f"FAILED: {img_info['filename']}: {exc}")

            pbar.update(1)

    # Summary
    done = sum(1 for j in manifest["jobs"].values() if j.get("status") == "done")
    failed = sum(1 for j in manifest["jobs"].values() if j.get("status") == "failed")
    print(f"\nInference complete. Done: {done}, Failed: {failed}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(results_csv_path: Path, output_dir: Path):
    """Compute metrics from the results CSV."""
    if not results_csv_path.exists():
        print("No results CSV found. Skipping evaluation.")
        return

    df = pd.read_csv(results_csv_path)
    if len(df) == 0:
        print("Results CSV is empty. Skipping evaluation.")
        return

    print(f"\n{'='*60}")
    print(f"EVALUATION: {len(df)} images")
    print(f"{'='*60}\n")

    # --- Overall accuracy (mapped classes) ---
    df["correct_mapped"] = df.apply(
        lambda r: r["predicted_class"] == r["gt_training_class"], axis=1
    )
    acc_mapped = df["correct_mapped"].mean()
    print(f"Overall accuracy (mapped 15-class → 4-class): {acc_mapped:.4f} ({df['correct_mapped'].sum()}/{len(df)})")

    # --- Per-class accuracy on mapped classes ---
    print(f"\nPer-class accuracy (correct if predicted_class == gt_training_class):")
    for gt_cls in sorted(GT_TO_TRAINING.values()):
        subset = df[df["gt_training_class"] == gt_cls]
        if len(subset) == 0:
            continue
        cls_acc = subset["correct_mapped"].mean()
        new_cls = TRAINING_TO_GT.get(gt_cls, "?")
        print(f"  {new_cls:>12s} → {gt_cls:>30s}: {cls_acc:.4f} ({subset['correct_mapped'].sum()}/{len(subset)})")

    # --- Confusion matrix (mapped classes) ---
    from sklearn.metrics import classification_report, confusion_matrix

    # Map predicted_class to new class label if it has one, else "other_predicted"
    def map_to_new(cls):
        return TRAINING_TO_GT.get(cls, f"other:{cls}")

    df["pred_mapped"] = df["predicted_class"].apply(map_to_new)

    gt_labels = sorted(GT_TO_TRAINING.keys())
    pred_labels = gt_labels + sorted(
        set(df["pred_mapped"].unique()) - set(gt_labels)
    )

    print(f"\nConfusion matrix (rows=ground truth, cols=predicted):")
    cm = confusion_matrix(
        df["gt_class"], df["pred_mapped"], labels=pred_labels
    )
    cm_df = pd.DataFrame(cm, index=pred_labels, columns=pred_labels)
    print(cm_df.to_string())

    # --- Classification report ---
    print(f"\nClassification report:")
    print(
        classification_report(
            df["gt_class"],
            df["pred_mapped"],
            labels=gt_labels,
            zero_division=0,
        )
    )

    # --- Confidence stats ---
    print(f"Mean confidence: {df['confidence'].mean():.4f}")
    print(f"Median confidence: {df['confidence'].median():.4f}")
    correct_conf = df[df["correct_mapped"]]["confidence"]
    wrong_conf = df[~df["correct_mapped"]]["confidence"]
    if len(correct_conf) > 0:
        print(f"Mean confidence (correct): {correct_conf.mean():.4f}")
    if len(wrong_conf) > 0:
        print(f"Mean confidence (wrong):   {wrong_conf.mean():.4f}")

    # --- Save evaluation report ---
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Chipped Dataset Inference Evaluation Report\n")
        f.write(f"Generated: {utc_now_iso()}\n")
        f.write(f"Total images: {len(df)}\n\n")
        f.write(f"Overall accuracy (mapped): {acc_mapped:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nClassification report:\n")
        f.write(classification_report(df["gt_class"], df["pred_mapped"], labels=gt_labels, zero_division=0))
    print(f"\nReport saved to: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run RemoteCLIP inference on chipped dataset.")
    parser.add_argument("--reset", action="store_true", help="Clear manifest and results, start fresh")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "inference_manifest.json"
    results_csv_path = output_dir / "inference_results.csv"

    if args.reset:
        print("Resetting: clearing manifest and results CSV.")
        for p in [manifest_path, results_csv_path]:
            if p.exists():
                p.unlink()

    # --- Discover images ---
    print(f"Scanning {IMAGE_DIR}...")
    images = discover_images(IMAGE_DIR)
    print(f"Found {len(images)} images across {len(set(i['gt_class'] for i in images))} classes.")

    if not images:
        print("No images found. Exiting.")
        sys.exit(1)

    # --- Load model ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Import open_clip here to avoid import errors during --help
    import open_clip

    print(f"Loading RemoteCLIP ViT-L/14...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    print(f"Loading fine-tuned weights from {MODEL_WEIGHTS}...")
    state = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print("Model loaded.")

    # --- Run inference ---
    run_inference(images, model, tokenizer, preprocess, device, manifest_path, results_csv_path)

    # --- Evaluate ---
    evaluate(results_csv_path, output_dir)


if __name__ == "__main__":
    main()
