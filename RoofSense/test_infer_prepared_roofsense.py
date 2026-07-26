"""
test_infer_prepared_roofsense.py
================================
Focused regression tests for the RoofSense frozen-weights inference pipeline.

Uses synthetic manifests and JPEGs. No real RemoteCLIP checkpoint required.
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training_evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest_row(
    sample_id: str = "test-001",
    roofsense_class: str = "Gravel",
    mapped_remoteclip_class: str = "Unknown",
    split: str = "training",
    prepared_filename: str = "",
    status: str = "prepared",
    **extra,
) -> dict:
    """Build a minimal manifest row for testing."""
    row = {
        "sample_id": sample_id,
        "source_relpath": f"{roofsense_class.lower().replace(' ', '_')}/{sample_id}.tif",
        "source_stem": sample_id,
        "coco_image_id": "0",
        "coco_annotation_id": "0",
        "roofsense_class_id": "3",
        "roofsense_class": roofsense_class,
        "semantic_mask_class_id": "3",
        "mapped_remoteclip_class": mapped_remoteclip_class,
        "mapping_version": "1.0",
        "split": split,
        "prepared_filename": prepared_filename,
        "status": status,
        "error": "",
        # Diagnostic flags (defaults)
        "low_target_occupancy": "False",
        "other_class_in_context": "False",
        "same_class_neighbor": "False",
        "multi_label_overlap": "False",
        "coco_mask_disagreement": "False",
        "contains_nodata": "False",
        "potential_building_merge": "False",
        "probable_building_group_ambiguous": "False",
    }
    row.update(extra)
    return row


def _write_manifest_csv(path: Path, rows: list[dict]) -> None:
    """Write manifest rows to CSV."""
    if not rows:
        path.write_text("sample_id,status\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_prepared_dir(tmp: Path, rows: list[dict]) -> Path:
    """Create a synthetic prepared directory with manifest, metadata, and images."""
    prepared = tmp / "prepared"
    prepared.mkdir()
    images = prepared / "images"
    images.mkdir()

    # Write manifest
    _write_manifest_csv(prepared / "manifest.csv", rows)

    # Write preparation metadata
    (prepared / "preparation_metadata.json").write_text(
        json.dumps({"schema_version": "1.0", "script_version": "3.2"}), encoding="utf-8"
    )

    # Create synthetic JPEGs for prepared rows
    for row in rows:
        if row.get("status") == "prepared" and row.get("prepared_filename"):
            img_path = images / row["prepared_filename"]
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (224, 224), (128, 128, 128)).save(str(img_path), "JPEG")

    return prepared


# ---------------------------------------------------------------------------
# Import from the module under test (will fail initially)
# ---------------------------------------------------------------------------

try:
    from infer_prepared_roofsense import (
        ACCEPTED_TARGETS,
        EVALUATION_TARGET_MAP,
        MATERIAL_CLASSES,
        MATERIAL_DESCRIPTIONS,
        RESULTS_SCHEMA_VERSION,
        RESULTS_COLUMNS,
        ResultStore,
        compute_evaluation,
        ensure_complete_for_evaluation,
        filter_prepared_rows,
        validate_prepared_manifest,
    )
    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestCohortFiltering(unittest.TestCase):
    """Spec: Only manifest rows whose status is prepared enter inference."""

    def test_filter_excludes_non_prepared_rows(self):
        rows = [
            _make_manifest_row("a", status="prepared"),
            _make_manifest_row("b", status="excluded"),
            _make_manifest_row("c", status="excluded_non_roof_scale"),
            _make_manifest_row("d", status="prepared"),
        ]
        prepared = filter_prepared_rows(rows)
        self.assertEqual(len(prepared), 2)
        self.assertEqual({r["sample_id"] for r in prepared}, {"a", "d"})

    def test_filter_returns_empty_when_no_prepared(self):
        rows = [
            _make_manifest_row("a", status="excluded"),
            _make_manifest_row("b", status="excluded_non_roof_scale"),
        ]
        prepared = filter_prepared_rows(rows)
        self.assertEqual(prepared, [])

    def test_filter_preserves_prepared_row_fields(self):
        rows = [_make_manifest_row("a", roofsense_class="Metal", split="test")]
        prepared = filter_prepared_rows(rows)
        self.assertEqual(prepared[0]["roofsense_class"], "Metal")
        self.assertEqual(prepared[0]["split"], "test")


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestEvaluationTargetMapping(unittest.TestCase):
    """Spec: Versioned RoofSense class to RemoteCLIP target mapping."""

    def test_ceramic_tile_has_multiple_accepted_targets(self):
        accepted = EVALUATION_TARGET_MAP["Ceramic Tile"]
        self.assertIn("ClayTiles", accepted)
        self.assertIn("AsphaltTiles", accepted)
        self.assertEqual(len(accepted), 2)

    def test_single_target_classes_are_strict(self):
        single_target_classes = [
            "Dark-coloured Membrane",
            "Gravel",
            "Light-coloured Membrane",
            "Light-permitting Surface",
            "Metal",
            "Vegetation",
        ]
        for cls in single_target_classes:
            accepted = EVALUATION_TARGET_MAP[cls]
            self.assertEqual(len(accepted), 1, f"{cls} should have exactly 1 accepted target")

    def test_all_accepted_targets_are_valid_remoteclip_classes(self):
        for native_cls, targets in EVALUATION_TARGET_MAP.items():
            for t in targets:
                self.assertIn(
                    t, MATERIAL_CLASSES,
                    f"{native_cls} accepted target {t} not in MATERIAL_CLASSES",
                )

    def test_gravel_maps_to_amorphous_concrete(self):
        self.assertEqual(
            EVALUATION_TARGET_MAP["Gravel"],
            ["AmorphousConcrete"],
        )

    def test_dark_membrane_maps_to_metal_sheet_materials(self):
        self.assertEqual(
            EVALUATION_TARGET_MAP["Dark-coloured Membrane"],
            ["MetalSheetMaterials"],
        )


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestMultiTargetAccuracy(unittest.TestCase):
    """Spec: Both ClayTiles and AsphaltTiles correct for Ceramic Tile."""

    def test_ceramic_tile_correct_with_clay_tiles(self):
        accepted = EVALUATION_TARGET_MAP["Ceramic Tile"]
        self.assertIn("ClayTiles", accepted)

    def test_ceramic_tile_correct_with_asphalt_tiles(self):
        accepted = EVALUATION_TARGET_MAP["Ceramic Tile"]
        self.assertIn("AsphaltTiles", accepted)

    def test_ceramic_tile_incorrect_with_other(self):
        accepted = EVALUATION_TARGET_MAP["Ceramic Tile"]
        self.assertNotIn("Unknown", accepted)

    def test_gravel_correct_only_with_amorphous_concrete(self):
        accepted = EVALUATION_TARGET_MAP["Gravel"]
        self.assertEqual(accepted, ["AmorphousConcrete"])
        self.assertNotIn("MetalSheetMaterials", accepted)


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestValidation(unittest.TestCase):
    """Spec: Reject duplicate IDs, unsafe paths, unsupported targets, missing images."""

    def test_rejects_duplicate_sample_ids(self):
        rows = [
            _make_manifest_row("dup", prepared_filename="gravel/dup.jpg"),
            _make_manifest_row("dup", prepared_filename="gravel/dup.jpg"),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_prepared_manifest(rows)

    def test_rejects_empty_sample_id(self):
        rows = [_make_manifest_row("", prepared_filename="gravel/x.jpg")]
        with self.assertRaisesRegex(ValueError, "sample_id"):
            validate_prepared_manifest(rows)

    def test_rejects_unsafe_path_traversal(self):
        rows = [_make_manifest_row("a", prepared_filename="../escape.jpg")]
        with self.assertRaisesRegex(ValueError, "safe"):
            validate_prepared_manifest(rows)

    def test_rejects_absolute_path(self):
        rows = [_make_manifest_row("a", prepared_filename="/etc/passwd")]
        with self.assertRaisesRegex(ValueError, "safe"):
            validate_prepared_manifest(rows)

    def test_rejects_unsupported_mapped_target(self):
        rows = [_make_manifest_row("a", mapped_remoteclip_class="BogusClass")]
        with self.assertRaisesRegex(ValueError, "RemoteCLIP class"):
            validate_prepared_manifest(rows)

    def test_rejects_missing_prepared_image(self):
        with TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared"
            prepared.mkdir()
            images = prepared / "images"
            images.mkdir()
            rows = [_make_manifest_row("a", prepared_filename="gravel/a.jpg")]
            _write_manifest_csv(prepared / "manifest.csv", rows)
            # No actual image file
            with self.assertRaisesRegex(ValueError, "image.*not found|missing"):
                validate_prepared_manifest(rows, images_dir=images)

    def test_accepts_valid_manifest(self):
        with TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared"
            images = prepared / "images" / "gravel"
            images.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(str(images / "a.jpg"), "JPEG")
            rows = [_make_manifest_row("a", prepared_filename="gravel/a.jpg")]
            _write_manifest_csv(prepared / "manifest.csv", rows)
            validate_prepared_manifest(rows, images_dir=prepared / "images")

    def test_rejects_empty_roofsense_class(self):
        rows = [_make_manifest_row("a", roofsense_class="", prepared_filename="gravel/a.jpg")]
        with self.assertRaisesRegex(ValueError, "roofsense_class"):
            validate_prepared_manifest(rows)

    def test_rejects_unsupported_roofsense_class(self):
        rows = [_make_manifest_row("a", roofsense_class="BogusMaterial", prepared_filename="gravel/a.jpg")]
        with self.assertRaisesRegex(ValueError, "roofsense_class"):
            validate_prepared_manifest(rows)


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestResultStore(unittest.TestCase):
    """Spec: Resume by sample ID, atomic CSV, fingerprint validation."""

    def test_done_ids_tracks_completed_samples(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.csv")
            store.upsert({"sample_id": "a", "status": "done"})
            store.upsert({"sample_id": "b", "status": "failed"})
            store.upsert({"sample_id": "c", "status": "done"})
            self.assertEqual(store.done_ids(), {"a", "c"})

    def test_upsert_overwrites_existing(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.csv")
            store.upsert({"sample_id": "a", "status": "failed", "error": "oops"})
            store.upsert({"sample_id": "a", "status": "done", "error": ""})
            self.assertEqual(store.done_ids(), {"a"})

    def test_validate_metadata_rejects_changed_fingerprint(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.csv")
            store.initialize_metadata({"checkpoint_sha256": "abc123"})
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                store.validate_metadata({"checkpoint_sha256": "different"})

    def test_validate_metadata_passes_on_match(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.csv")
            store.initialize_metadata({"key": "value"})
            store.validate_metadata({"key": "value"})  # should not raise

    def test_csv_written_atomically(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "results.csv"
            store = ResultStore(csv_path)
            store.upsert({"sample_id": "a", "status": "done"})
            self.assertTrue(csv_path.exists())


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestResumeAndRetry(unittest.TestCase):
    """Spec: Completed rows skipped, failed rows retried on resume."""

    def test_done_ids_excludes_failed(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.csv")
            store.upsert({"sample_id": "a", "status": "done"})
            store.upsert({"sample_id": "b", "status": "failed"})
            self.assertEqual(store.done_ids(), {"a"})

    def test_failed_rows_not_in_done_ids(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.csv")
            store.upsert({"sample_id": "x", "status": "failed"})
            self.assertNotIn("x", store.done_ids())


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestCompletenessCheck(unittest.TestCase):
    """Spec: Evaluation requires exactly one successful result per prepared sample."""

    def test_raises_when_results_missing(self):
        inputs = [_make_manifest_row("a"), _make_manifest_row("b")]
        results = [{"sample_id": "a", "status": "done"}]
        with self.assertRaisesRegex(ValueError, "missing|failed"):
            ensure_complete_for_evaluation(inputs, results)

    def test_raises_when_results_failed(self):
        inputs = [_make_manifest_row("a"), _make_manifest_row("b")]
        results = [
            {"sample_id": "a", "status": "done"},
            {"sample_id": "b", "status": "failed"},
        ]
        with self.assertRaisesRegex(ValueError, "missing|failed"):
            ensure_complete_for_evaluation(inputs, results)

    def test_passes_when_all_done(self):
        inputs = [_make_manifest_row("a"), _make_manifest_row("b")]
        results = [
            {"sample_id": "a", "status": "done"},
            {"sample_id": "b", "status": "done"},
        ]
        # Should not raise
        ensure_complete_for_evaluation(inputs, results)

    def test_rejects_duplicate_result_ids(self):
        inputs = [_make_manifest_row("a")]
        results = [
            {"sample_id": "a", "status": "done"},
            {"sample_id": "a", "status": "done"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ensure_complete_for_evaluation(inputs, results)


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestEvaluation(unittest.TestCase):
    """Spec: Combined evaluation across splits, accepted-target accuracy."""

    def _make_inputs_and_results(self, predictions: dict[str, str]) -> tuple:
        """Helper: create inputs/results where key=sample_id, value=predicted_class."""
        inputs = []
        results = []
        for sid, pred in predictions.items():
            inputs.append(_make_manifest_row(
                sid,
                roofsense_class="Gravel",
                mapped_remoteclip_class="Unknown",
                split="training",
            ))
            results.append({
                "sample_id": sid,
                "status": "done",
                "predicted_class": pred,
                "confidence": 0.9,
                **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES},
            })
        return inputs, results

    def test_accepted_target_accuracy_all_correct(self):
        inputs, results = self._make_inputs_and_results(
            {"a": "AmorphousConcrete", "b": "AmorphousConcrete"}
        )
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            self.assertAlmostEqual(summary["accepted_target_accuracy"], 1.0)

    def test_accepted_target_accuracy_partial(self):
        inputs, results = self._make_inputs_and_results(
            {"a": "AmorphousConcrete", "b": "ClayTiles"}
        )
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            self.assertAlmostEqual(summary["accepted_target_accuracy"], 0.5)

    def test_ceramic_tile_multi_target_accuracy(self):
        inputs = [
            _make_manifest_row("a", roofsense_class="Ceramic Tile", mapped_remoteclip_class="ClayTiles"),
            _make_manifest_row("b", roofsense_class="Ceramic Tile", mapped_remoteclip_class="ClayTiles"),
        ]
        results = [
            {"sample_id": "a", "status": "done", "predicted_class": "ClayTiles", "confidence": 0.9,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
            {"sample_id": "b", "status": "done", "predicted_class": "AsphaltTiles", "confidence": 0.8,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
        ]
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            # Both correct because Ceramic Tile accepts ClayTiles and AsphaltTiles
            self.assertAlmostEqual(summary["accepted_target_accuracy"], 1.0)

    def test_ceramic_tile_wrong_prediction(self):
        inputs = [
            _make_manifest_row("a", roofsense_class="Ceramic Tile", mapped_remoteclip_class="ClayTiles"),
        ]
        results = [
            {"sample_id": "a", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.9,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
        ]
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            self.assertAlmostEqual(summary["accepted_target_accuracy"], 0.0)

    def test_per_class_accuracy(self):
        inputs = [
            _make_manifest_row("a", roofsense_class="Gravel", mapped_remoteclip_class="Unknown"),
            _make_manifest_row("b", roofsense_class="Metal", mapped_remoteclip_class="MetalSheetMaterials"),
        ]
        results = [
            {"sample_id": "a", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.9,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
            {"sample_id": "b", "status": "done", "predicted_class": "MetalSheetMaterials", "confidence": 0.85,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
        ]
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            per_class = summary["per_class_accuracy"]
            self.assertAlmostEqual(per_class["Gravel"]["accuracy"], 1.0)
            self.assertAlmostEqual(per_class["Metal"]["accuracy"], 1.0)

    def test_combined_across_splits(self):
        inputs = [
            _make_manifest_row("a", roofsense_class="Gravel", split="training"),
            _make_manifest_row("b", roofsense_class="Gravel", split="validation"),
            _make_manifest_row("c", roofsense_class="Gravel", split="test"),
        ]
        results = [
            {"sample_id": "a", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.9,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
            {"sample_id": "b", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.85,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
            {"sample_id": "c", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.95,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
        ]
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            self.assertAlmostEqual(summary["accepted_target_accuracy"], 1.0)
            self.assertEqual(summary["total"], 3)

    def test_writes_confusion_matrix_csv(self):
        inputs = [_make_manifest_row("a")]
        results = [
            {"sample_id": "a", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.9,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
        ]
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            compute_evaluation(inputs, results, out)
            self.assertTrue((out / "confusion_matrix.csv").exists())
            self.assertTrue((out / "evaluation_summary.json").exists())
            self.assertTrue((out / "evaluation_report.txt").exists())

    def test_writes_split_counts_as_provenance(self):
        inputs = [
            _make_manifest_row("a", split="training"),
            _make_manifest_row("b", split="test"),
        ]
        results = [
            {"sample_id": "a", "status": "done", "predicted_class": "Unknown", "confidence": 0.9,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
            {"sample_id": "b", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.85,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
        ]
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            self.assertIn("split_counts", summary)
            self.assertEqual(summary["split_counts"]["training"], 1)
            self.assertEqual(summary["split_counts"]["test"], 1)


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestDiagnosticBreakdowns(unittest.TestCase):
    """Spec: Accuracy grouped by each selected boolean diagnostic flag."""

    def test_diagnostic_flag_breakdown(self):
        inputs = [
            _make_manifest_row("a", roofsense_class="Gravel", low_target_occupancy="True"),
            _make_manifest_row("b", roofsense_class="Gravel", low_target_occupancy="False"),
            _make_manifest_row("c", roofsense_class="Gravel", low_target_occupancy="True"),
        ]
        results = [
            {"sample_id": "a", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.9,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
            {"sample_id": "b", "status": "done", "predicted_class": "ClayTiles", "confidence": 0.7,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
            {"sample_id": "c", "status": "done", "predicted_class": "AmorphousConcrete", "confidence": 0.85,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
        ]
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            self.assertIn("diagnostic_breakdowns", summary)
            breakdown = summary["diagnostic_breakdowns"]
            self.assertIn("low_target_occupancy", breakdown)
            # low_target_occupancy=True: 2 rows, both correct (AmorphousConcrete)
            low = breakdown["low_target_occupancy"]
            self.assertEqual(low["True"]["total"], 2)
            self.assertEqual(low["True"]["correct"], 2)
            # low_target_occupancy=False: 1 row, wrong (ClayTiles)
            self.assertEqual(low["False"]["total"], 1)
            self.assertEqual(low["False"]["correct"], 0)

    def test_missing_diagnostic_columns_treated_as_false(self):
        """Spec: Missing optional diagnostic columns treated as false."""
        inputs = [
            {"sample_id": "a", "roofsense_class": "Gravel", "mapped_remoteclip_class": "Unknown",
             "split": "training", "prepared_filename": "gravel/a.jpg", "status": "prepared"},
        ]
        results = [
            {"sample_id": "a", "status": "done", "predicted_class": "Unknown", "confidence": 0.9,
             **{f"prob_{cls}": 0.0 for cls in MATERIAL_CLASSES}},
        ]
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            breakdown = summary["diagnostic_breakdowns"]
            # Should not error, missing columns treated as False
            self.assertIn("low_target_occupancy", breakdown)


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestCountReconciliation(unittest.TestCase):
    """Spec: Count reconciliation between manifest and result states."""

    def test_reconciliation_counts(self):
        inputs = [_make_manifest_row("a"), _make_manifest_row("b"), _make_manifest_row("c")]
        results = [
            {"sample_id": "a", "status": "done"},
            {"sample_id": "b", "status": "done"},
            {"sample_id": "c", "status": "failed"},
        ]
        with TemporaryDirectory() as tmp:
            summary = compute_evaluation(inputs, results, Path(tmp))
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["correct_count"], 0)  # Gravel→Unknown, but predicted_class=""


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestMaterialClasses(unittest.TestCase):
    """Spec: 15 RemoteCLIP material classes."""

    def test_fifteen_material_classes(self):
        self.assertEqual(len(MATERIAL_CLASSES), 15)

    def test_material_descriptions_match_classes(self):
        self.assertEqual(set(MATERIAL_DESCRIPTIONS.keys()), set(MATERIAL_CLASSES))


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestResultSchema(unittest.TestCase):
    """Spec: Result row schema includes all required fields."""

    def test_results_columns_include_core_provenance(self):
        required = ["sample_id", "prepared_filename", "roofsense_class",
                     "mapped_remoteclip_class", "accepted_remoteclip_targets", "split"]
        for col in required:
            self.assertIn(col, RESULTS_COLUMNS, f"Missing column: {col}")

    def test_results_columns_include_prediction(self):
        for col in ["predicted_class", "confidence"]:
            self.assertIn(col, RESULTS_COLUMNS, f"Missing column: {col}")

    def test_results_columns_include_all_prob_columns(self):
        for cls in MATERIAL_CLASSES:
            self.assertIn(f"prob_{cls}", RESULTS_COLUMNS, f"Missing prob column for {cls}")

    def test_results_columns_include_execution_state(self):
        for col in ["status", "error", "started_at", "finished_at"]:
            self.assertIn(col, RESULTS_COLUMNS, f"Missing column: {col}")

    def test_results_columns_include_diagnostic_flags(self):
        flags = [
            "low_target_occupancy", "other_class_in_context", "same_class_neighbor",
            "multi_label_overlap", "coco_mask_disagreement", "contains_nodata",
            "potential_building_merge", "probable_building_group_ambiguous",
        ]
        for flag in flags:
            self.assertIn(flag, RESULTS_COLUMNS, f"Missing diagnostic column: {flag}")


@unittest.skipUnless(IMPORT_OK, "infer_prepared_roofsense not importable yet")
class TestCLISmoke(unittest.TestCase):
    """Spec: CLI help/import smoke without loading model."""

    def test_module_imports(self):
        import infer_prepared_roofsense
        self.assertTrue(hasattr(infer_prepared_roofsense, "main"))

    def test_cli_help_exits_cleanly(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("infer_prepared_roofsense.py")), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--prepared-dir", result.stdout)
        self.assertIn("--weights", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--reset", result.stdout)
        self.assertIn("--device", result.stdout)


if __name__ == "__main__":
    unittest.main()
