import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xAI_notebooks"))

from attribution_helpers import batch_recovery


class BatchRecoveryTests(unittest.TestCase):
    def test_manifest_round_trip_preserves_job_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "batch_run_manifest.json"
            manifest = batch_recovery.load_manifest(manifest_path)
            batch_recovery.upsert_job(
                manifest,
                job_id="img1__transformer_explainability",
                image_id="img1.png",
                method_name="transformer_explainability",
                output_path=Path(tmpdir) / "img1.png",
            )
            batch_recovery.mark_job_running(manifest, "img1__transformer_explainability")
            batch_recovery.mark_job_done(manifest, "img1__transformer_explainability")
            batch_recovery.save_manifest(manifest_path, manifest)

            reloaded = batch_recovery.load_manifest(manifest_path)
            self.assertEqual(reloaded["jobs"]["img1__transformer_explainability"]["status"], "done")

    def test_resume_action_skips_only_when_done_and_output_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "img1__transformer_explainability.png"
            output_path.write_bytes(b"png")
            manifest = batch_recovery.load_manifest(Path(tmpdir) / "batch_run_manifest.json")
            batch_recovery.upsert_job(
                manifest,
                job_id="img1__transformer_explainability",
                image_id="img1.png",
                method_name="transformer_explainability",
                output_path=output_path,
            )

            self.assertEqual(
                batch_recovery.resolve_job_action(
                    manifest,
                    job_id="img1__transformer_explainability",
                    output_path=output_path,
                ),
                "run",
            )

            batch_recovery.mark_job_done(manifest, "img1__transformer_explainability")
            self.assertEqual(
                batch_recovery.resolve_job_action(
                    manifest,
                    job_id="img1__transformer_explainability",
                    output_path=output_path,
                ),
                "skip",
            )

            output_path.unlink()
            self.assertEqual(
                batch_recovery.resolve_job_action(
                    manifest,
                    job_id="img1__transformer_explainability",
                    output_path=output_path,
                ),
                "run",
            )

    def test_append_stats_row_writes_header_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "transformer_spatial_stats.csv"
            row_one = {"image_id": "a.png", "method": "transformer_explainability", "mass_center_25_square": 0.4}
            row_two = {"image_id": "b.png", "method": "transformer_explainability", "mass_center_25_square": 0.6}

            batch_recovery.append_stats_row(stats_path, row_one)
            batch_recovery.append_stats_row(stats_path, row_two)

            with stats_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["image_id"], "a.png")
            self.assertEqual(rows[1]["image_id"], "b.png")


if __name__ == "__main__":
    unittest.main()
