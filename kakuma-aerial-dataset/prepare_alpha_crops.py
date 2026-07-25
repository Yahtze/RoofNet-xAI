#!/usr/bin/env python3
"""
prepare_alpha_crops.py
=====================
CLI that turns RGBA TIFF source images into a flat directory of padded RGB
JPEGs suitable for RemoteCLIP inference.

Reads only source TIFFs. Produces:
- One flat JPEG directory with collision-safe filenames
- A minimal CSV manifest
- A preparation_metadata.json provenance file
- A visual contact sheet for QA review

Usage:
    python prepare_alpha_crops.py --source-dir <path> --output-dir <path> [--reset] [--contact-sheet-size 32]
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
from PIL import Image, ImageDraw, ImageFont

from crop_experiment import (
    MANIFEST_COLUMNS,
    crop_from_alpha,
    make_sample_id,
    square_pad_rgb,
    write_manifest,
    write_prepared_jpeg,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PADDING = 8
JPEG_QUALITY = 95
ALPHA_THRESHOLD = 0
SUPPORTED_LABELS = ("metal_sheet", "thatch", "plastic", "other")
IMAGE_EXTENSIONS = (".tif", ".tiff")
SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PreparationSummary:
    """Result summary from prepare_dataset."""

    discovered: int
    prepared: int
    failed: int
    output_dir: Path
    manifest_path: Path
    metadata_path: Path
    contact_sheet_path: Path | None


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def discover_sources(source_dir: Path) -> list[dict[str, str]]:
    """Walk label directories and return list of source records.

    Each record has: source_path, source_relpath, gt_class
    """
    sources = []
    source_dir = Path(source_dir)

    for label_dir in sorted(source_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label not in SUPPORTED_LABELS:
            continue
        for img_path in sorted(label_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                relpath = f"{label}/{img_path.name}"
                sources.append(
                    {
                        "source_path": str(img_path),
                        "source_relpath": relpath,
                        "gt_class": label,
                    }
                )
    return sources


# ---------------------------------------------------------------------------
# Metadata fingerprint
# ---------------------------------------------------------------------------


def compute_source_inventory_fingerprint(sources: list[dict[str, str]]) -> str:
    """Compute a deterministic fingerprint of the source inventory."""
    import hashlib

    h = hashlib.sha256()
    for src in sorted(sources, key=lambda s: s["source_relpath"]):
        h.update(src["source_relpath"].encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------


def build_contact_sheet(
    image_dir: Path,
    manifest_rows: list[dict[str, str]],
    output_path: Path,
    max_images: int = 32,
) -> Path:
    """Build a contact sheet from a deterministic sample of prepared JPEGs.

    Stratified across available classes, ordered by sample_id.
    Overlays filename and gt_class on each thumbnail.
    """
    # Group by class
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        if row.get("status") != "prepared":
            continue
        gt = row.get("gt_class", "unknown")
        by_class.setdefault(gt, []).append(row)

    # Deterministic sample: sort by sample_id, take proportional from each class
    selected = []
    per_class = max(1, max_images // max(len(by_class), 1))
    for cls in sorted(by_class):
        rows = sorted(by_class[cls], key=lambda r: r["sample_id"])
        selected.extend(rows[:per_class])
    selected = selected[:max_images]
    selected.sort(key=lambda r: r["sample_id"])

    if not selected:
        # Create blank contact sheet
        img = Image.new("RGB", (400, 40), (255, 255, 255))
        img.save(str(output_path), "JPEG", quality=JPEG_QUALITY)
        return output_path

    # Layout: square grid
    thumb_size = 224
    n = len(selected)
    cols = int(np.ceil(np.sqrt(n)))
    rows_count = int(np.ceil(n / cols))

    sheet = Image.new("RGB", (cols * thumb_size, rows_count * thumb_size), (200, 200, 200))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, row in enumerate(selected):
        r, c = divmod(idx, cols)
        x0, y0 = c * thumb_size, r * thumb_size

        fname = row.get("prepared_filename", "")
        img_path = image_dir / fname if fname else None
        if img_path and img_path.exists():
            try:
                thumb = Image.open(img_path).convert("RGB")
                thumb = thumb.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x0, y0))
            except Exception:
                pass

        # Overlay text
        label = f"{row['sample_id'][:20]}\n{row['gt_class']}"
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 0), font=font)

    sheet.save(str(output_path), "JPEG", quality=JPEG_QUALITY)
    return output_path


# ---------------------------------------------------------------------------
# Main preparation logic
# ---------------------------------------------------------------------------


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    reset: bool = False,
    contact_sheet_size: int = 32,
) -> PreparationSummary:
    """Run the full preparation pipeline.

    Returns PreparationSummary with counts and output paths.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if reset and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    metadata_path = output_dir / "preparation_metadata.json"
    contact_sheet_path = output_dir / "contact_sheet.jpg"

    # Discover sources
    sources = discover_sources(source_dir)
    print(f"Discovered {len(sources)} source TIFFs under {source_dir}")

    if not sources:
        print("No source images found. Exiting.")
        sys.exit(1)

    # Process each source
    manifest_rows = []
    prepared_count = 0
    failed_count = 0

    for i, src in enumerate(sources):
        source_relpath = src["source_relpath"]
        gt_class = src["gt_class"]
        sample_id = make_sample_id(source_relpath)

        try:
            # Load RGBA
            rgba = Image.open(src["source_path"]).convert("RGBA")

            # Crop from alpha
            cropped = crop_from_alpha(rgba, padding=PADDING)

            # Square pad
            padded = square_pad_rgb(cropped)

            # Write JPEG
            prepared_filename = write_prepared_jpeg(
                padded, images_dir, sample_id, quality=JPEG_QUALITY
            )

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "source_relpath": source_relpath,
                    "prepared_filename": prepared_filename.name,
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

    # Write manifest
    write_manifest(manifest_path, manifest_rows)
    print(f"Wrote manifest: {manifest_path}")

    # Write metadata
    metadata = {
        "source_root": str(source_dir),
        "source_inventory_fingerprint": compute_source_inventory_fingerprint(sources),
        "padding": PADDING,
        "jpeg_quality": JPEG_QUALITY,
        "alpha_threshold": ALPHA_THRESHOLD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False, suffix=".json"
    ) as tmp:
        json.dump(metadata, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(metadata_path)
    print(f"Wrote metadata: {metadata_path}")

    # Build contact sheet
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

    print(f"\nPreparation complete:")
    print(f"  Discovered: {summary.discovered}")
    print(f"  Prepared:   {summary.prepared}")
    print(f"  Failed:     {summary.failed}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare alpha-cropped RGB JPEGs from RGBA TIFF sources."
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
        help="Output directory for prepared JPEGs, manifest, and metadata",
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
