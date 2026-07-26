#!/usr/bin/env python3
"""
infer_prepared_roofsense.py
===========================
RoofSense-specific RemoteCLIP inference workflow for prepared roof crops.

Evaluates all prepared crops as one combined cohort. Dataset splits remain
available as provenance but do not partition the reported metrics. Only
manifest rows whose status is ``prepared`` enter inference or the evaluation
denominator.

Usage::

    .venv/bin/python RoofSense/infer_prepared_roofsense.py \\
      --prepared-dir RoofSense-dataset/prepared_roofsense_coco_pad8 \\
      --weights /path/to/fine_tuned_remoteclip.pth \\
      --output-dir RoofSense-dataset/inference_roofsense_coco_pad8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_EVALUATION_DIR = REPO_ROOT / "training_evaluation"
if str(TRAINING_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_EVALUATION_DIR))

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

# ---------------------------------------------------------------------------
# Evaluation target mapping
# ---------------------------------------------------------------------------
# Versioned mapping from each native RoofSense class to one or more accepted
# RemoteCLIP targets. Ceramic Tile is deliberately multi-target because the
# source class description includes both clay/ceramic tiles and asphalt tiles.

EVALUATION_TARGET_MAP: dict[str, list[str]] = {
    "Ceramic Tile": ["ClayTiles", "AsphaltTiles"],
    "Dark-coloured Membrane": ["MetalSheetMaterials"],
    "Gravel": ["AmorphousConcrete"],
    "Light-coloured Membrane": ["AmorphousMembrane"],
    "Light-permitting Surface": ["GlassSheetMaterials"],
    "Metal": ["MetalSheetMaterials"],
    "Vegetation": ["GreenVegetative"],
}

ACCEPTED_TARGETS = EVALUATION_TARGET_MAP  # alias for clarity

EVALUATION_TARGET_MAP_VERSION = "1.1"

# Primary manifest mapping (RoofSense class -> primary RemoteCLIP target)
PRIMARY_MANIFEST_MAP: dict[str, str] = {
    "Ceramic Tile": "ClayTiles",
    "Dark-coloured Membrane": "AmorphousAsphalt",
    "Gravel": "Unknown",
    "Light-coloured Membrane": "AmorphousMembrane",
    "Light-permitting Surface": "GlassSheetMaterials",
    "Metal": "MetalSheetMaterials",
    "Vegetation": "GreenVegetative",
}

# Required manifest columns for RoofSense inference
REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "prepared_filename",
    "roofsense_class",
    "mapped_remoteclip_class",
    "split",
    "status",
}

# Diagnostic flags to copy from manifest
DIAGNOSTIC_FLAGS = [
    "low_target_occupancy",
    "other_class_in_context",
    "same_class_neighbor",
    "multi_label_overlap",
    "coco_mask_disagreement",
    "contains_nodata",
    "potential_building_merge",
    "probable_building_group_ambiguous",
]

# Result schema
RESULTS_COLUMNS = [
    "sample_id",
    "prepared_filename",
    "roofsense_class",
    "mapped_remoteclip_class",
    "accepted_remoteclip_targets",
    "split",
    "predicted_class",
    "confidence",
] + [f"prob_{cls}" for cls in MATERIAL_CLASSES] + [
    "status",
    "error",
    "started_at",
    "finished_at",
] + DIAGNOSTIC_FLAGS

RESULTS_SCHEMA_VERSION = "1.0"


class IncompleteInferenceError(RuntimeError):
    """Raised after partial results have been persisted but are incomplete."""


# ---------------------------------------------------------------------------
# Cohort filtering
# ---------------------------------------------------------------------------


def filter_prepared_rows(rows: list[dict]) -> list[dict]:
    """Filter manifest to rows with status == 'prepared'."""
    return [r for r in rows if r.get("status") == "prepared"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _is_safe_class_relative_path(filename: str) -> bool:
    """Check that filename is a safe class-relative path beneath images/."""
    if not filename:
        return False
    p = Path(filename)
    if p.is_absolute():
        return False
    parts = p.parts
    # Must be exactly (class_dir, filename.ext)
    if len(parts) != 2:
        return False
    if parts[0] in ("", ".", "..") or parts[1] in ("", ".", ".."):
        return False
    # No traversal
    if ".." in parts:
        return False
    return True


def validate_prepared_manifest(
    rows: list[dict],
    images_dir: Path | None = None,
) -> None:
    """Validate the preparation manifest for inference readiness.

    Only prepared rows are validated for filename and image existence.
    Raises ValueError on first invalid row.
    """
    prepared = filter_prepared_rows(rows)

    # Check at least one prepared row
    if not prepared:
        raise ValueError("manifest contains no prepared rows")

    seen_ids: set[str] = set()

    for row in prepared:
        sid = row.get("sample_id", "")
        if not isinstance(sid, str) or not sid:
            raise ValueError("missing or empty sample_id")
        if sid in seen_ids:
            raise ValueError(f"duplicate sample_id: {sid}")
        seen_ids.add(sid)

        # Validate native RoofSense class
        rs_class = row.get("roofsense_class", "")
        if not rs_class:
            raise ValueError(f"missing roofsense_class for {sid}")
        if rs_class not in EVALUATION_TARGET_MAP:
            raise ValueError(f"unsupported roofsense_class: {rs_class}")

        # Validate mapped RemoteCLIP class
        mapped = row.get("mapped_remoteclip_class", "")
        if mapped not in MATERIAL_CLASSES:
            raise ValueError(
                f"unsupported mapped RemoteCLIP class '{mapped}' for {sid}"
            )

        # Validate filename
        fname = row.get("prepared_filename", "")
        if not isinstance(fname, str) or not fname:
            raise ValueError(f"missing prepared_filename for {sid}")
        if not _is_safe_class_relative_path(fname):
            raise ValueError(f"unsafe prepared_filename path: {fname}")

        # Validate image exists
        if images_dir is not None:
            img_path = images_dir / fname
            if not img_path.is_file():
                raise ValueError(f"prepared image not found: {img_path}")


def load_prepared_manifest(manifest_path: Path) -> list[dict]:
    """Load and return the full manifest CSV as list of dicts."""
    df = pd.read_csv(manifest_path)
    return df.to_dict("records")


def load_preparation_metadata(metadata_path: Path) -> dict:
    """Load and validate the preparation metadata JSON."""
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
            _validate_result_rows(self._rows)
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
            "w",
            encoding="utf-8",
            dir=self.csv_path.parent,
            delete=False,
            suffix=".csv",
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
            "w",
            encoding="utf-8",
            dir=self.csv_path.parent,
            delete=False,
            suffix=".json",
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


def _validate_result_rows(rows: list[dict]) -> None:
    """Reject result tables whose resume key is not unique."""
    seen: set[str] = set()
    for row in rows:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("missing result sample_id")
        if sid in seen:
            raise ValueError(f"duplicate sample_id in results: {sid}")
        seen.add(sid)


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------


def ensure_complete_for_evaluation(
    input_rows: list[dict], result_rows: list[dict]
) -> None:
    """Verify that every prepared input has exactly one successful result.

    Raises ValueError with count of missing/failed rows.
    """
    _validate_result_rows(result_rows)
    input_ids = {r["sample_id"] for r in input_rows}
    result_by_id: dict[str, dict] = {}
    for r in result_rows:
        result_by_id[r["sample_id"]] = r

    missing = input_ids - set(result_by_id.keys())
    failed = [sid for sid, r in result_by_id.items() if r.get("status") == "failed"]

    if missing or failed:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing")
        if failed:
            parts.append(f"{len(failed)} failed")
        raise ValueError(
            f"Incomplete results: {', '.join(parts)}. "
            f"Re-run inference or use --reset for fresh results."
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _parse_bool_flag(value) -> bool:
    """Parse a boolean flag from manifest row (handles str/bool/int)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes")


def compute_evaluation(
    input_rows: list[dict],
    result_rows: list[dict],
    output_dir: Path,
) -> dict:
    """Compute combined evaluation across the full prepared cohort.

    Returns evaluation summary dict.
    """
    result_by_id = {r["sample_id"]: r for r in result_rows}

    # Build comparison table
    comparisons = []
    for inp in input_rows:
        sid = inp["sample_id"]
        rs_class = inp.get("roofsense_class", "")
        mapped_target = inp.get("mapped_remoteclip_class", "")
        accepted = EVALUATION_TARGET_MAP.get(rs_class, [mapped_target])
        split = inp.get("split", "")

        res = result_by_id.get(sid, {})
        predicted = res.get("predicted_class", "")
        confidence = float(res.get("confidence", 0))
        is_correct = predicted in accepted if predicted else False

        row_data = {
            "sample_id": sid,
            "roofsense_class": rs_class,
            "mapped_remoteclip_class": mapped_target,
            "accepted_remoteclip_targets": accepted,
            "predicted_class": predicted,
            "is_correct": is_correct,
            "confidence": confidence,
            "split": split,
            "status": res.get("status", "missing"),
        }
        # Copy diagnostic flags
        for flag in DIAGNOSTIC_FLAGS:
            row_data[flag] = _parse_bool_flag(inp.get(flag, False))
        comparisons.append(row_data)

    df = pd.DataFrame(comparisons)

    # Filter to done rows for accuracy
    done_mask = df["status"] == "done"
    done_df = df[done_mask]
    total = len(df)
    done_count = int(done_mask.sum())
    failed_count = int((df["status"] == "failed").sum())
    missing_count = int((df["status"] == "missing").sum())
    correct_count = int(done_df["is_correct"].sum()) if len(done_df) > 0 else 0
    accuracy = correct_count / done_count if done_count > 0 else 0.0

    # Per-native-class accuracy (on done rows only)
    per_class: dict[str, dict] = {}
    for rs_cls in sorted(EVALUATION_TARGET_MAP.keys()):
        subset = done_df[done_df["roofsense_class"] == rs_cls]
        if len(subset) > 0:
            cls_correct = int(subset["is_correct"].sum())
            per_class[rs_cls] = {
                "total": len(subset),
                "correct": cls_correct,
                "accuracy": float(cls_correct / len(subset)),
                "accepted_targets": EVALUATION_TARGET_MAP[rs_cls],
            }

    # Confusion matrix: rows = native RoofSense classes, columns = predicted RemoteCLIP
    native_classes = sorted(EVALUATION_TARGET_MAP.keys())
    pred_classes = sorted(set(done_df["predicted_class"].unique()) - {""})
    all_pred_classes = sorted(set(pred_classes) | set(MATERIAL_CLASSES))

    from sklearn.metrics import classification_report

    cm_data = {}
    for rs_cls in native_classes:
        cm_data[rs_cls] = {}
        subset = done_df[done_df["roofsense_class"] == rs_cls]
        for pred_cls in all_pred_classes:
            count = int((subset["predicted_class"] == pred_cls).sum())
            cm_data[rs_cls][pred_cls] = count

    cm_df = pd.DataFrame(cm_data).T.fillna(0).astype(int)

    # Classification report on mapped targets
    if len(done_df) > 0:
        report = classification_report(
            done_df["roofsense_class"],
            done_df.apply(
                lambda r: r["roofsense_class"] if r["is_correct"] else f"wrong:{r['predicted_class']}",
                axis=1,
            ),
            labels=native_classes,
            zero_division=0,
        )
    else:
        report = "No completed results."

    # Diagnostic flag breakdowns
    diagnostic_breakdowns: dict[str, dict] = {}
    for flag in DIAGNOSTIC_FLAGS:
        flag_true = done_df[done_df[flag] == True]
        flag_false = done_df[done_df[flag] == False]
        diagnostic_breakdowns[flag] = {
            "True": {
                "total": len(flag_true),
                "correct": int(flag_true["is_correct"].sum()) if len(flag_true) > 0 else 0,
                "accuracy": float(flag_true["is_correct"].mean()) if len(flag_true) > 0 else 0.0,
            },
            "False": {
                "total": len(flag_false),
                "correct": int(flag_false["is_correct"].sum()) if len(flag_false) > 0 else 0,
                "accuracy": float(flag_false["is_correct"].mean()) if len(flag_false) > 0 else 0.0,
            },
        }

    # Split counts (provenance only)
    split_counts: dict[str, int] = {}
    for split_name, group in df.groupby("split"):
        split_counts[split_name] = len(group)

    # Native class to mapped target counts
    native_to_target_counts: dict[str, dict[str, int]] = {}
    for rs_cls in native_classes:
        subset = done_df[done_df["roofsense_class"] == rs_cls]
        target_counts = Counter(subset["predicted_class"])
        native_to_target_counts[rs_cls] = dict(target_counts)

    summary = {
        "accepted_target_accuracy": float(accuracy),
        "total": total,
        "done_count": done_count,
        "correct_count": correct_count,
        "failed_count": failed_count,
        "missing_count": missing_count,
        "per_class_accuracy": per_class,
        "native_to_target_counts": native_to_target_counts,
        "diagnostic_breakdowns": diagnostic_breakdowns,
        "split_counts": split_counts,
        "confusion_matrix": cm_df.to_dict(),
    }

    # Write evaluation report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("RoofSense Frozen-Weights Inference Evaluation Report\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Total prepared: {total}\n")
        f.write(f"Done: {done_count}, Failed: {failed_count}, Missing: {missing_count}\n\n")
        f.write(f"Accepted-target accuracy: {accuracy:.4f} ({correct_count}/{done_count})\n\n")

        f.write("Per-native-class accuracy:\n")
        for rs_cls in sorted(per_class):
            stats = per_class[rs_cls]
            f.write(
                f"  {rs_cls}: {stats['accuracy']:.4f} "
                f"({stats['correct']}/{stats['total']}) "
                f"[accepted: {', '.join(stats['accepted_targets'])}]\n"
            )

        f.write("\nNative class → predicted RemoteCLIP class counts:\n")
        for rs_cls in sorted(native_to_target_counts):
            targets = native_to_target_counts[rs_cls]
            f.write(f"  {rs_cls}:\n")
            for t_cls, cnt in sorted(targets.items()):
                f.write(f"    {t_cls}: {cnt}\n")

        f.write("\nDiagnostic flag breakdowns:\n")
        for flag, breakdown in sorted(diagnostic_breakdowns.items()):
            f.write(f"  {flag}:\n")
            for label, stats in breakdown.items():
                f.write(
                    f"    {label}: total={stats['total']}, "
                    f"correct={stats['correct']}, "
                    f"accuracy={stats['accuracy']:.4f}\n"
                )

        f.write("\nSplit counts (provenance only):\n")
        for split_name in sorted(split_counts):
            f.write(f"  {split_name}: {split_counts[split_name]}\n")

        f.write("\nConfusion matrix (rows=native RoofSense, cols=predicted RemoteCLIP):\n")
        f.write(cm_df.to_string())
        f.write("\n\nClassification report:\n")
        f.write(report)

        f.write(f"\n\nCount reconciliation:\n")
        f.write(f"  prepared: {total}\n")
        f.write(f"  done: {done_count}\n")
        f.write(f"  failed: {failed_count}\n")
        f.write(f"  missing: {missing_count}\n")

    # Write evaluation summary JSON
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")

    # Write confusion matrix CSV
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    print(f"\nEvaluation report: {report_path}")
    print(f"Evaluation summary: {summary_path}")
    print(f"Confusion matrix: {cm_path}")

    return summary


# ---------------------------------------------------------------------------
# Main inference logic
# ---------------------------------------------------------------------------


def _build_result_row(
    sid: str,
    fname: str,
    rs_class: str,
    mapped_target: str,
    accepted_targets: list[str],
    split: str,
    diagnostic_values: dict,
    *,
    predicted_class: str = "",
    confidence: float = 0.0,
    probs: dict[str, float] | None = None,
    status: str = "done",
    error: str = "",
    started_at: str = "",
    finished_at: str = "",
) -> dict:
    """Build a result row dict with all required columns."""
    if probs is None:
        probs = {f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}

    row = {
        "sample_id": sid,
        "prepared_filename": fname,
        "roofsense_class": rs_class,
        "mapped_remoteclip_class": mapped_target,
        "accepted_remoteclip_targets": str(accepted_targets),
        "split": split,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        **probs,
        "status": status,
        "error": error,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    # Copy diagnostic flags
    for flag in DIAGNOSTIC_FLAGS:
        row[flag] = diagnostic_values.get(flag, False)
    return row


def run_inference(
    prepared_dir: Path,
    weights_path: Path,
    output_dir: Path,
    *,
    reset: bool = False,
    device: str = "auto",
) -> None:
    """Run RemoteCLIP inference on prepared RoofSense JPEGs with CSV resume."""
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
    images_dir = prepared_dir / "images"

    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)
    if not metadata_path.exists():
        print(f"Error: preparation metadata not found at {metadata_path}", file=sys.stderr)
        sys.exit(1)

    # Load and filter
    all_rows = load_prepared_manifest(manifest_path)
    prepared_rows = filter_prepared_rows(all_rows)
    print(f"Loaded {len(all_rows)} rows from manifest, {len(prepared_rows)} prepared")

    if not prepared_rows:
        print("Error: no prepared rows in manifest", file=sys.stderr)
        sys.exit(1)

    # Validate
    validate_prepared_manifest(all_rows, images_dir)
    prep_metadata = load_preparation_metadata(metadata_path)

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
    cohort_fingerprint = json_fingerprint(
        sorted(
            [(r["sample_id"], r["mapped_remoteclip_class"]) for r in prepared_rows]
        )
    )
    accepted_target_fingerprint = json_fingerprint(EVALUATION_TARGET_MAP)

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
        "cohort_fingerprint": cohort_fingerprint,
        "checkpoint_sha256": ckpt_sha,
        "prompt_set_fingerprint": prompt_fingerprint,
        "accepted_target_fingerprint": accepted_target_fingerprint,
        "accepted_target_map_version": EVALUATION_TARGET_MAP_VERSION,
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
    pending = [r for r in prepared_rows if r["sample_id"] not in done_ids]
    already_done = len(prepared_rows) - len(pending)
    print(f"Total: {len(prepared_rows)}, already done: {already_done}, pending: {len(pending)}")

    if not pending:
        print("All rows already completed. Running evaluation...")
        result_rows = pd.read_csv(results_csv).to_dict("records")
        ensure_complete_for_evaluation(prepared_rows, result_rows)
        compute_evaluation(prepared_rows, result_rows, output_dir)
        return

    # Load model
    print("Loading RemoteCLIP model...")
    model, tokenizer, preprocess_val = load_finetuned_remoteclip(weights_path, device)
    print("Model loaded.")
    tokenized_prompts = tokenizer(MATERIAL_PROMPTS).to(device)

    # Process pending
    for i, row in enumerate(pending):
        sid = row["sample_id"]
        fname = row.get("prepared_filename", "")
        rs_class = row.get("roofsense_class", "")
        mapped_target = row.get("mapped_remoteclip_class", "")
        accepted_targets = EVALUATION_TARGET_MAP.get(rs_class, [mapped_target])
        split = row.get("split", "")

        # Collect diagnostic flags
        diagnostic_values = {}
        for flag in DIAGNOSTIC_FLAGS:
            diagnostic_values[flag] = _parse_bool_flag(row.get(flag, False))

        now_start = datetime.now(timezone.utc).isoformat()

        try:
            # Validate path safety
            img_path = images_dir / fname
            if not img_path.resolve().is_relative_to(images_dir.resolve()):
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
                probs_tensor = similarities.softmax(dim=-1).squeeze(0)

            top_idx = probs_tensor.argmax().item()
            predicted_class = MATERIAL_CLASSES[top_idx]
            confidence = float(probs_tensor[top_idx].item())

            probs_dict = {
                f"prob_{cls}": round(float(probs_tensor[j].item()), 6)
                for j, cls in enumerate(MATERIAL_CLASSES)
            }

            result_row = _build_result_row(
                sid=sid,
                fname=fname,
                rs_class=rs_class,
                mapped_target=mapped_target,
                accepted_targets=accepted_targets,
                split=split,
                diagnostic_values=diagnostic_values,
                predicted_class=predicted_class,
                confidence=confidence,
                probs=probs_dict,
                status="done",
                started_at=now_start,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            store.upsert(result_row)

        except Exception as exc:
            now_end = datetime.now(timezone.utc).isoformat()
            result_row = _build_result_row(
                sid=sid,
                fname=fname,
                rs_class=rs_class,
                mapped_target=mapped_target,
                accepted_targets=accepted_targets,
                split=split,
                diagnostic_values=diagnostic_values,
                status="failed",
                error=str(exc),
                started_at=now_start,
                finished_at=now_end,
            )
            store.upsert(result_row)

        # Diagnostic progress
        processed = i + 1
        current_done = len(store.done_ids())
        current_failed = sum(
            1 for r in store._rows if r.get("status") == "failed"
        )
        if processed % 100 == 0 or processed == len(pending):
            print(
                f"  Processed {processed}/{len(pending)}, "
                f"done={current_done}, failed={current_failed}"
            )

    # Final evaluation
    print("\nInference complete. Running evaluation...")
    result_rows = pd.read_csv(results_csv).to_dict("records")

    done_count = sum(1 for r in result_rows if r.get("status") == "done")
    failed_count = sum(1 for r in result_rows if r.get("status") == "failed")
    print(f"Results: done={done_count}, failed={failed_count}")

    try:
        ensure_complete_for_evaluation(prepared_rows, result_rows)
    except ValueError as exc:
        raise IncompleteInferenceError(str(exc)) from exc
    compute_evaluation(prepared_rows, result_rows, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RemoteCLIP inference on prepared RoofSense crops."
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        required=True,
        help="Directory containing manifest.csv, preparation_metadata.json, and images/",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to fine-tuned RemoteCLIP checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Experiment output directory for results and evaluation",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove existing output directory before starting",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference (default: auto)",
    )
    args = parser.parse_args()

    try:
        run_inference(
            prepared_dir=Path(args.prepared_dir),
            weights_path=Path(args.weights),
            output_dir=Path(args.output_dir),
            reset=args.reset,
            device=args.device,
        )
    except IncompleteInferenceError as exc:
        print(f"Partial results saved: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
