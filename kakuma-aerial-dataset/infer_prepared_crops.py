#!/usr/bin/env python3
"""
infer_prepared_crops.py
=======================
Run RemoteCLIP inference on prepared alpha-cropped JPEGs.

Reads only prepared JPEGs plus the preparation manifest. Uses a CSV result
table as resume state and validates immutable experiment fingerprints before
continuing.

Usage:
    python infer_prepared_crops.py --prepared-dir <path> --weights <path> --output-dir <path> [--reset] [--allow-failed]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from crop_experiment import MANIFEST_COLUMNS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

GT_TO_TRAINING = {
    "metal_sheet": "MetalSheetMaterials",
    "thatch": "Thatch",
    "plastic": "PolycarbonateSheetMaterials",
    "other": "Unknown",
}

RESULTS_COLUMNS = [
    "sample_id",
    "prepared_filename",
    "gt_class",
    "predicted_class",
    "confidence",
] + [f"prob_{cls}" for cls in MATERIAL_CLASSES] + [
    "status",
    "error",
    "started_at",
    "finished_at",
]

RESULTS_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_input_manifest(rows: list[dict]) -> None:
    """Validate the preparation manifest for duplicate IDs and other issues."""
    seen = set()
    for row in rows:
        sid = row.get("sample_id", "")
        if sid in seen:
            raise ValueError(f"duplicate sample_id: {sid}")
        seen.add(sid)


def load_prepared_manifest(manifest_path: Path) -> list[dict]:
    """Load and validate the preparation manifest CSV."""
    df = pd.read_csv(manifest_path)
    rows = df.to_dict("records")
    validate_input_manifest(rows)
    return rows


def load_preparation_metadata(metadata_path: Path) -> dict:
    """Load the preparation metadata JSON."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Result store (CSV-only resume)
# ---------------------------------------------------------------------------


class ResultStore:
    """Atomic CSV result store with sidecar metadata for resume validation."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.metadata_path = self.csv_path.parent / "results_metadata.json"
        self._rows: list[dict] = []
        self._metadata: dict = {}
        self._load()

    def _load(self) -> None:
        """Load existing results and metadata from disk."""
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            self._rows = df.to_dict("records")
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def done_ids(self) -> set[str]:
        """Return set of sample_ids that have status 'done'."""
        return {r["sample_id"] for r in self._rows if r.get("status") == "done"}

    def upsert(self, row: dict) -> None:
        """Insert or update a result row by sample_id."""
        sid = row["sample_id"]
        for i, existing in enumerate(self._rows):
            if existing.get("sample_id") == sid:
                self._rows[i] = row
                self._save()
                return
        self._rows.append(row)
        self._save()

    def _save(self) -> None:
        """Atomically write results CSV."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows, columns=RESULTS_COLUMNS)

        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.csv_path.parent, delete=False, suffix=".csv"
        ) as tmp:
            df.to_csv(tmp, index=False)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.csv_path)

    def initialize_metadata(self, metadata: dict) -> None:
        """Write metadata sidecar for the first time."""
        self._metadata = metadata
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Atomically write metadata sidecar."""
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.csv_path.parent, delete=False, suffix=".json"
        ) as tmp:
            json.dump(self._metadata, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.metadata_path)

    def validate_metadata(self, expected: dict) -> None:
        """Validate that stored metadata matches expected values.

        Raises ValueError with remediation hint on mismatch.
        """
        if not self._metadata:
            return

        for key, expected_val in expected.items():
            actual_val = self._metadata.get(key)
            if actual_val != expected_val:
                raise ValueError(
                    f"{key} mismatch: stored={actual_val!r}, expected={expected_val!r}. "
                    f"Use --reset to clear stale results."
                )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def ensure_complete_for_evaluation(
    input_rows: list[dict], result_rows: list[dict]
) -> None:
    """Verify that every prepared input has exactly one successful result.

    Raises ValueError with count of missing/failed rows.
    """
    input_ids = {r["sample_id"] for r in input_rows}
    result_by_id: dict[str, dict] = {}
    for r in result_rows:
        result_by_id[r["sample_id"]] = r

    missing = input_ids - set(result_by_id.keys())
    failed = [
        sid for sid, r in result_by_id.items() if r.get("status") == "failed"
    ]

    if missing or failed:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing")
        if failed:
            parts.append(f"{len(failed)} failed")
        raise ValueError(
            f"Incomplete results: {', '.join(parts)}. "
            f"Re-run inference or use --allow-failed for partial evaluation."
        )


def compute_evaluation(
    input_rows: list[dict],
    result_rows: list[dict],
    output_dir: Path,
) -> dict:
    """Compute strict mapped accuracy and confusion matrix.

    Returns evaluation summary dict.
    """
    result_by_id = {r["sample_id"]: r for r in result_rows}

    # Build comparison table
    comparisons = []
    for inp in input_rows:
        sid = inp["sample_id"]
        gt_class = inp.get("gt_class", "")
        res = result_by_id.get(sid, {})
        predicted = res.get("predicted_class", "")
        mapped_target = GT_TO_TRAINING.get(gt_class, gt_class)
        is_correct = predicted == mapped_target
        comparisons.append(
            {
                "sample_id": sid,
                "gt_class": gt_class,
                "mapped_target": mapped_target,
                "predicted_class": predicted,
                "is_correct": is_correct,
                "confidence": float(res.get("confidence", 0)),
            }
        )

    df = pd.DataFrame(comparisons)

    # Strict accuracy
    total = len(df)
    correct = df["is_correct"].sum()
    accuracy = correct / total if total > 0 else 0.0

    # Per-class accuracy
    per_class = {}
    for gt_cls in sorted(GT_TO_TRAINING.keys()):
        subset = df[df["gt_class"] == gt_cls]
        if len(subset) > 0:
            cls_correct = subset["is_correct"].sum()
            per_class[gt_cls] = {
                "total": len(subset),
                "correct": int(cls_correct),
                "accuracy": float(cls_correct / len(subset)),
            }

    # Confusion matrix
    gt_labels = sorted(GT_TO_TRAINING.keys())
    pred_labels = gt_labels + sorted(
        set(df["predicted_class"].unique()) - set(gt_labels)
    )

    from sklearn.metrics import classification_report, confusion_matrix

    def map_pred(cls):
        return {v: k for k, v in GT_TO_TRAINING.items()}.get(cls, f"other:{cls}")

    df["pred_mapped"] = df["predicted_class"].apply(map_pred)
    pred_labels_mapped = gt_labels + sorted(
        set(df["pred_mapped"].unique()) - set(gt_labels)
    )

    cm = confusion_matrix(
        df["gt_class"], df["pred_mapped"], labels=pred_labels_mapped
    )
    cm_df = pd.DataFrame(cm, index=pred_labels_mapped, columns=pred_labels_mapped)

    report = classification_report(
        df["gt_class"],
        df["pred_mapped"],
        labels=gt_labels,
        zero_division=0,
    )

    # Count reconciliation
    done_count = sum(1 for r in result_rows if r.get("status") == "done")
    failed_count = sum(1 for r in result_rows if r.get("status") == "failed")

    summary = {
        "strict_mapped_target_accuracy": float(accuracy),
        "total_images": total,
        "correct_predictions": int(correct),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm_df.to_dict(),
        "prepared_count": len(input_rows),
        "done_count": done_count,
        "failed_count": failed_count,
        "missing_count": len(input_rows) - len(result_rows),
    }

    # Write evaluation report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Alpha-Derived Crop Inference Evaluation Report\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Total images: {total}\n\n")
        f.write(f"Strict mapped target accuracy: {accuracy:.4f} ({int(correct)}/{total})\n\n")
        f.write("Per-class accuracy:\n")
        for cls, stats in sorted(per_class.items()):
            f.write(f"  {cls}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})\n")
        f.write("\nConfusion matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nClassification report:\n")
        f.write(report)
        f.write(f"\n\nCount reconciliation:\n")
        f.write(f"  prepared: {summary['prepared_count']}\n")
        f.write(f"  done: {summary['done_count']}\n")
        f.write(f"  failed: {summary['failed_count']}\n")
        f.write(f"  missing: {summary['missing_count']}\n")

    # Write evaluation summary JSON
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")

    print(f"\nEvaluation report: {report_path}")
    print(f"Evaluation summary: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Main inference logic
# ---------------------------------------------------------------------------


def run_inference(
    prepared_dir: Path,
    weights_path: Path,
    output_dir: Path,
    *,
    reset: bool = False,
    allow_failed: bool = False,
    device: str = "auto",
) -> None:
    """Run RemoteCLIP inference on prepared JPEGs with CSV-only resume."""
    from remoteclip_runtime import (
        PREPROCESSING_ID,
        checkpoint_sha256,
        json_fingerprint,
        load_finetuned_remoteclip,
        preprocessing_spec,
    )

    prepared_dir = Path(prepared_dir)
    weights_path = Path(weights_path)
    output_dir = Path(output_dir)

    # Load preparation artifacts
    manifest_path = prepared_dir / "manifest.csv"
    metadata_path = prepared_dir / "preparation_metadata.json"

    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"Error: preparation metadata not found at {metadata_path}")
        sys.exit(1)

    input_rows = load_prepared_manifest(manifest_path)
    prep_metadata = load_preparation_metadata(metadata_path)

    print(f"Loaded {len(input_rows)} entries from preparation manifest")

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")

    # Compute fingerprints
    ckpt_sha = checkpoint_sha256(weights_path)
    prompt_fingerprint = json_fingerprint(MATERIAL_PROMPTS)
    prep_manifest_fingerprint = json_fingerprint(
        sorted([r["sample_id"] for r in input_rows])
    )

    # Setup result store
    results_csv = output_dir / "results.csv"
    if reset:
        import shutil

        if output_dir.exists():
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = ResultStore(results_csv)

    # Initialize or validate metadata
    expected_metadata = {
        "preparation_manifest_fingerprint": prep_manifest_fingerprint,
        "checkpoint_sha256": ckpt_sha,
        "prompt_set_sha256": prompt_fingerprint,
        "preprocessing_id": PREPROCESSING_ID,
        "model_name": "ViT-L-14",
        "device": device,
        "schema_version": RESULTS_SCHEMA_VERSION,
    }

    if store.metadata_path.exists():
        store.validate_metadata(expected_metadata)
        print("Result metadata validated (fingerprint match)")
    else:
        store.initialize_metadata(expected_metadata)
        print("Initialized result metadata")

    # Determine pending images
    done_ids = store.done_ids()
    pending = [r for r in input_rows if r["sample_id"] not in done_ids]

    # Filter failed rows unless --allow-failed
    if not allow_failed:
        failed_rows = [r for r in input_rows if r.get("status") == "failed"]
        if failed_rows:
            print(
                f"Warning: {len(failed_rows)} preparation failures will be skipped. "
                f"Use --allow-failed to include them."
            )

    already_done = len(input_rows) - len(pending)
    print(f"Total: {len(input_rows)}, already done: {already_done}, pending: {len(pending)}")

    if not pending:
        print("All rows already completed. Running evaluation...")
        result_rows = pd.read_csv(results_csv).to_dict("records")
        ensure_complete_for_evaluation(input_rows, result_rows)
        compute_evaluation(input_rows, result_rows, output_dir)
        return

    # Load model
    print("Loading RemoteCLIP model...")
    model, tokenizer, preprocess_val = load_finetuned_remoteclip(weights_path, device)
    print("Model loaded.")

    # Tokenize prompts once
    tokenized_prompts = tokenizer(MATERIAL_PROMPTS).to(device)

    # Process pending
    for i, row in enumerate(pending):
        sid = row["sample_id"]
        fname = row.get("prepared_filename", "")
        gt_class = row.get("gt_class", "")

        # Skip failed preparation rows unless --allow-failed
        if row.get("status") == "failed" and not allow_failed:
            continue

        now_start = datetime.now(timezone.utc).isoformat()

        try:
            if row.get("status") == "failed":
                # Record failure from preparation
                store.upsert(
                    {
                        "sample_id": sid,
                        "prepared_filename": "",
                        "gt_class": gt_class,
                        "predicted_class": "",
                        "confidence": 0.0,
                        **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES},
                        "status": "failed",
                        "error": f"Preparation failed: {row.get('error', '')}",
                        "started_at": now_start,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue

            # Validate path safety
            img_path = prepared_dir / "images" / fname
            if not img_path.resolve().is_relative_to((prepared_dir / "images").resolve()):
                raise ValueError(f"Path traversal detected in filename: {fname}")

            if not img_path.exists():
                raise FileNotFoundError(f"Prepared image not found: {img_path}")

            # Load and preprocess
            pil_img = Image.open(img_path).convert("RGB")
            image_tensor = preprocess_val(pil_img).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(tokenized_prompts)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

                # Use learned logit scale
                logit_scale = model.logit_scale.exp()
                similarities = logit_scale * image_features @ text_features.T
                probs = similarities.softmax(dim=-1).squeeze(0)

            top_idx = probs.argmax().item()
            predicted_class = MATERIAL_CLASSES[top_idx]
            confidence = float(probs[top_idx].item())

            # Build result row
            result_row = {
                "sample_id": sid,
                "prepared_filename": fname,
                "gt_class": gt_class,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 6),
                **{
                    f"prob_{cls}": round(float(probs[j].item()), 6)
                    for j, cls in enumerate(MATERIAL_CLASSES)
                },
                "status": "done",
                "error": "",
                "started_at": now_start,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            store.upsert(result_row)

        except Exception as exc:
            now_end = datetime.now(timezone.utc).isoformat()
            store.upsert(
                {
                    "sample_id": sid,
                    "prepared_filename": fname,
                    "gt_class": gt_class,
                    "predicted_class": "",
                    "confidence": 0.0,
                    **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES},
                    "status": "failed",
                    "error": str(exc),
                    "started_at": now_start,
                    "finished_at": now_end,
                }
            )

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pending)}")

    # Final evaluation
    print("\nInference complete. Running evaluation...")
    result_rows = pd.read_csv(results_csv).to_dict("records")

    done_count = sum(1 for r in result_rows if r.get("status") == "done")
    failed_count = sum(1 for r in result_rows if r.get("status") == "failed")
    print(f"Results: done={done_count}, failed={failed_count}")

    try:
        ensure_complete_for_evaluation(input_rows, result_rows)
        compute_evaluation(input_rows, result_rows, output_dir)
    except ValueError as e:
        print(f"Skipping evaluation: {e}")
        if allow_failed:
            print("Partial results saved. Re-run without --allow-failed for full evaluation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RemoteCLIP inference on prepared alpha-cropped JPEGs."
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        required=True,
        help="Root directory of prepared JPEGs (output of prepare_alpha_crops.py)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to fine-tuned RemoteCLIP weights (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results CSV, metadata, and evaluation",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing results and start fresh",
    )
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Include failed preparation rows in inference (preserves failures in report)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference (default: auto)",
    )
    args = parser.parse_args()

    run_inference(
        prepared_dir=Path(args.prepared_dir),
        weights_path=Path(args.weights),
        output_dir=Path(args.output_dir),
        reset=args.reset,
        allow_failed=args.allow_failed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
