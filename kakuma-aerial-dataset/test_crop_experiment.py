"""
test_crop_experiment.py
======================
Local regression tests for the alpha-derived crop inference pipeline.

Placed beside the experiment code because repository-level test/ and tests/
are ignored by policy.
"""

import sys
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training_evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Task 1: Runtime spec tests
# ---------------------------------------------------------------------------

from remoteclip_runtime import PREPROCESSING_ID, preprocessing_spec


class RuntimeSpecTests(unittest.TestCase):
    def test_preprocessing_spec_matches_finetuning_clip_normalization(self):
        self.assertEqual(
            preprocessing_spec(),
            {
                "resize": 224,
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
            },
        )
        self.assertEqual(PREPROCESSING_ID, "remoteclip-vit-l-14-default-val-v1")


# ---------------------------------------------------------------------------
# Task 2: Crop and ID primitive tests
# ---------------------------------------------------------------------------

from crop_experiment import (
    crop_from_alpha,
    make_sample_id,
    square_pad_rgb,
    write_manifest,
    write_prepared_jpeg,
)


class CropTests(unittest.TestCase):
    def test_alpha_box_is_expanded_by_eight_and_clipped(self):
        rgba = np.zeros((10, 12, 4), dtype=np.uint8)
        rgba[..., :3] = [10, 20, 30]
        rgba[2:5, 1:4, 3] = 1
        crop = crop_from_alpha(Image.fromarray(rgba, "RGBA"), padding=8)
        self.assertEqual(crop.size, (12, 10))
        self.assertEqual(crop.mode, "RGB")

    def test_alpha_box_rejects_an_empty_mask(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "empty alpha footprint"):
            crop_from_alpha(Image.fromarray(rgba, "RGBA"), padding=8)

    def test_square_pad_replicates_the_nearest_edge(self):
        image = Image.new("RGB", (2, 4), (9, 8, 7))
        padded = square_pad_rgb(image)
        self.assertEqual(padded.size, (4, 4))
        self.assertEqual(padded.getpixel((0, 0)), (9, 8, 7))

    def test_same_basename_in_different_class_paths_has_distinct_ids(self):
        self.assertNotEqual(
            make_sample_id("metal_sheet/tile_1.tif"),
            make_sample_id("thatch/tile_1.tif"),
        )


class WriterTests(unittest.TestCase):
    def test_writer_emits_rgb_jpeg_and_minimal_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = write_prepared_jpeg(
                Image.new("RGB", (4, 4)), root, "roof--abc"
            )
            with Image.open(image_path) as result:
                self.assertEqual(result.format, "JPEG")
                self.assertEqual(result.mode, "RGB")
            write_manifest(
                root / "manifest.csv",
                [
                    {
                        "sample_id": "roof--abc",
                        "source_relpath": "thatch/a.tif",
                        "prepared_filename": image_path.name,
                        "gt_class": "thatch",
                        "status": "prepared",
                        "error": "",
                    }
                ],
            )
            self.assertEqual(
                (root / "manifest.csv").read_text().splitlines()[0],
                "sample_id,source_relpath,prepared_filename,gt_class,status,error",
            )

    def test_writer_allows_identical_prepared_jpeg_to_be_written_twice(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGB", (4, 4), (127, 83, 41))
            first = write_prepared_jpeg(image, root, "roof--abc")
            second = write_prepared_jpeg(image, root, "roof--abc")
            self.assertEqual(first, second)


# ---------------------------------------------------------------------------
# Task 3: Preparation end-to-end test
# ---------------------------------------------------------------------------

from prepare_alpha_crops import prepare_dataset


class PreparationTests(unittest.TestCase):
    def test_prepare_dataset_keeps_class_directories_and_does_not_square_pad(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            for label in ("metal_sheet", "thatch"):
                folder = root / label
                folder.mkdir(parents=True)
                rgba = np.zeros((20, 40, 4), dtype=np.uint8)
                rgba[..., :3] = [20, 40, 60]
                rgba[5:12, 6:13, 3] = 1
                Image.fromarray(rgba, "RGBA").save(folder / "same_name.tif")
            output = Path(tmp) / "prepared"
            summary = prepare_dataset(root, output, reset=True, contact_sheet_size=4)
            self.assertEqual(summary.prepared, 2)
            metal_images = list((output / "images" / "metal_sheet").glob("*.jpg"))
            thatch_images = list((output / "images" / "thatch").glob("*.jpg"))
            self.assertEqual(len(metal_images), 1)
            self.assertEqual(len(thatch_images), 1)
            with Image.open(metal_images[0]) as result:
                self.assertEqual(result.size, (21, 20))
            self.assertTrue((output / "manifest.csv").exists())
            self.assertTrue((output / "contact_sheet.jpg").exists())


# ---------------------------------------------------------------------------
# Task 4: Result store and resume tests
# ---------------------------------------------------------------------------

from infer_prepared_crops import (
    ResultStore,
    ensure_complete_for_evaluation,
    run_inference,
    validate_input_manifest,
)


class ResumeTests(unittest.TestCase):
    def test_resume_keys_results_by_sample_id_not_basename(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.csv")
            store.upsert({"sample_id": "metal--1", "status": "done"})
            store.upsert({"sample_id": "thatch--2", "status": "done"})
            self.assertEqual(set(store.done_ids()), {"metal--1", "thatch--2"})

    def test_resume_rejects_changed_checkpoint_fingerprint(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.csv")
            store.initialize_metadata({"checkpoint_sha256": "old"})
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                store.validate_metadata({"checkpoint_sha256": "new"})

    def test_manifest_rejects_duplicate_sample_ids(self):
        rows = [
            {
                "sample_id": "same",
                "status": "prepared",
                "gt_class": "thatch",
                "prepared_filename": "thatch/same.jpg",
            },
            {
                "sample_id": "same",
                "status": "prepared",
                "gt_class": "thatch",
                "prepared_filename": "thatch/same.jpg",
            },
        ]
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            validate_input_manifest(rows)

    def test_manifest_rejects_invalid_label_and_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "unsupported gt_class"):
            validate_input_manifest([
                {"sample_id": "a", "status": "prepared", "gt_class": "bad", "prepared_filename": "a.jpg"}
            ])
        with self.assertRaisesRegex(ValueError, "path traversal"):
            validate_input_manifest([
                {"sample_id": "a", "status": "prepared", "gt_class": "thatch", "prepared_filename": "../a.jpg"}
            ])

    def test_manifest_allows_class_relative_prepared_filename(self):
        validate_input_manifest([
            {
                "sample_id": "a",
                "status": "prepared",
                "gt_class": "thatch",
                "prepared_filename": "thatch/a.jpg",
            }
        ])

    def test_manifest_rejects_failed_rows_without_allow_failed(self):
        with self.assertRaisesRegex(ValueError, "failed preparation"):
            validate_input_manifest([
                {"sample_id": "a", "status": "failed", "gt_class": "thatch", "prepared_filename": ""}
            ])

    def test_result_store_rejects_duplicate_completed_ids(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text("sample_id,status\na,done\na,done\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                ResultStore(path)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_refuses_missing_or_failed_prepared_rows(self):
        inputs = [{"sample_id": "a"}, {"sample_id": "b"}]
        results = [
            {"sample_id": "a", "status": "done"},
            {"sample_id": "b", "status": "failed"},
        ]
        with self.assertRaisesRegex(ValueError, "1 failed"):
            ensure_complete_for_evaluation(inputs, results)

    def test_evaluation_rejects_duplicate_result_ids(self):
        inputs = [{"sample_id": "a"}]
        results = [
            {"sample_id": "a", "status": "done"},
            {"sample_id": "a", "status": "done"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            ensure_complete_for_evaluation(inputs, results)

    def test_allow_failed_persists_failure_then_reports_incomplete_run(self):
        with TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared"
            prepared.mkdir()
            (prepared / "manifest.csv").write_text(
                "sample_id,source_relpath,prepared_filename,gt_class,status,error\n"
                "a,thatch/a.tif,,thatch,failed,empty alpha footprint\n",
                encoding="utf-8",
            )
            (prepared / "preparation_metadata.json").write_text("{}\n", encoding="utf-8")
            weights = Path(tmp) / "unused.pth"
            weights.write_bytes(b"not loaded because every row failed")
            output = Path(tmp) / "results"
            with self.assertRaisesRegex(Exception, "incomplete inference"):
                run_inference(prepared, weights, output, allow_failed=True)
            self.assertIn("a", (output / "results.csv").read_text(encoding="utf-8"))

    def test_allow_failed_cli_writes_results_and_exits_nonzero(self):
        with TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared"
            prepared.mkdir()
            (prepared / "manifest.csv").write_text(
                "sample_id,source_relpath,prepared_filename,gt_class,status,error\n"
                "a,thatch/a.tif,,thatch,failed,empty alpha footprint\n",
                encoding="utf-8",
            )
            (prepared / "preparation_metadata.json").write_text("{}\n", encoding="utf-8")
            weights = Path(tmp) / "unused.pth"
            weights.write_bytes(b"not loaded because every row failed")
            output = Path(tmp) / "results"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("infer_prepared_crops.py")),
                    "--prepared-dir", str(prepared),
                    "--weights", str(weights),
                    "--output-dir", str(output),
                    "--allow-failed",
                    "--device", "cpu",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue((output / "results.csv").exists())


# ---------------------------------------------------------------------------
# Task 5: Static regression test for forbidden LAION loader configuration
# ---------------------------------------------------------------------------


class ForbiddenLoaderTests(unittest.TestCase):
    def test_finetuned_checkpoint_consumers_do_not_request_laion_pretrained_weights(self):
        paths = [
            REPO_ROOT / "training_evaluation" / "remoteclip_classify.py",
            REPO_ROOT / "xAI_notebooks" / "remoteclip_segmentation_overlap_batch.py",
            REPO_ROOT / "xAI_notebooks" / "remoteclip_segmentation_overlap_marimo.py",
            REPO_ROOT / "xAI_notebooks" / "remoteclip_xai_attribution_marimo.py",
        ]
        for path in paths:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('pretrained="laion2b_s32b_b82k"', text, path.name)
            self.assertNotIn("pretrained='laion2b_s32b_b82k'", text, path.name)


if __name__ == "__main__":
    unittest.main()
