#!/usr/bin/env python3
"""Export complete source TIFF frames as class-organized RGB JPEGs.

Unlike ``prepare_alpha_crops.py``, this utility does not use alpha to locate a
building footprint. It discards alpha and preserves each TIFF's complete frame
for visual verification.

Usage:
    python prepare_full_tiff_jpegs.py --source-dir <path> --output-dir <path> [--reset] [--contact-sheet-size 32]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

from crop_experiment import make_sample_id, write_manifest, write_prepared_jpeg
from prepare_alpha_crops import (
    JPEG_QUALITY,
    SCHEMA_VERSION,
    PreparationSummary,
    build_contact_sheet,
    compute_source_inventory_fingerprint,
    discover_sources,
)


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    reset: bool = False,
    contact_sheet_size: int = 32,
) -> PreparationSummary:
    """Convert complete TIFF frames to RGB JPEGs, organized by ground-truth class."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if reset and output_dir.exists():
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    metadata_path = output_dir / "preparation_metadata.json"
    contact_sheet_path = output_dir / "contact_sheet.jpg"

    sources = discover_sources(source_dir)
    print(f"Discovered {len(sources)} source TIFFs under {source_dir}")
    if not sources:
        print("No source images found. Exiting.")
        sys.exit(1)

    manifest_rows = []
    prepared_count = 0
    failed_count = 0
    for i, src in enumerate(sources):
        source_relpath = src["source_relpath"]
        gt_class = src["gt_class"]
        sample_id = make_sample_id(source_relpath)
        try:
            image = Image.open(src["source_path"]).convert("RGB")
            class_images_dir = images_dir / gt_class
            prepared_filename = write_prepared_jpeg(
                image, class_images_dir, sample_id, quality=JPEG_QUALITY
            )
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "source_relpath": source_relpath,
                    "prepared_filename": prepared_filename.relative_to(images_dir).as_posix(),
                    "gt_class": gt_class,
                    "status": "prepared",
                    "error": "",
                }
            )
            prepared_count += 1
        except Exception as exc:
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "source_relpath": source_relpath,
                    "prepared_filename": "",
                    "gt_class": gt_class,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            failed_count += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(sources)}")

    write_manifest(manifest_path, manifest_rows)
    print(f"Wrote manifest: {manifest_path}")

    metadata = {
        "conversion": "full_frame_rgb_alpha_discarded",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jpeg_quality": JPEG_QUALITY,
        "schema_version": SCHEMA_VERSION,
        "source_inventory_fingerprint": compute_source_inventory_fingerprint(sources),
        "source_root": str(source_dir),
    }
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False, suffix=".json"
    ) as tmp:
        json.dump(metadata, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(metadata_path)
    print(f"Wrote metadata: {metadata_path}")

    build_contact_sheet(
        images_dir, manifest_rows, contact_sheet_path, max_images=contact_sheet_size
    )
    print(f"Wrote contact sheet: {contact_sheet_path}")

    summary = PreparationSummary(
        discovered=len(sources),
        prepared=prepared_count,
        failed=failed_count,
        output_dir=output_dir,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        contact_sheet_path=contact_sheet_path,
    )
    print("\nPreparation complete:")
    print(f"  Discovered: {summary.discovered}")
    print(f"  Prepared:   {summary.prepared}")
    print(f"  Failed:     {summary.failed}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare full-frame RGB JPEGs from TIFF sources, discarding alpha."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Source directory containing label subdirectories with .tif/.tiff images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for full-frame JPEGs, manifest, and metadata",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove existing output directory before starting",
    )
    parser.add_argument(
        "--contact-sheet-size",
        type=int,
        default=32,
        help="Maximum number of images on the contact sheet (default: 32)",
    )
    args = parser.parse_args()
    prepare_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        reset=args.reset,
        contact_sheet_size=args.contact_sheet_size,
    )


if __name__ == "__main__":
    main()
