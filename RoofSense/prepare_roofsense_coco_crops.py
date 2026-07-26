#!/usr/bin/env python3
"""
prepare_roofsense_coco_crops.py
================================
Prepare the RoofSense dataset for frozen RemoteCLIP inference by producing
one centered RGB JPEG chip per COCO roof annotation.

The COCO polygon is a *seed* identifying the target material. The crop
boundary comes from the connected nonblack RGB region containing that seed,
expanded to a square with padding. Constant (black) padding fills any region
that extends beyond the tile boundary.

Solar Panel annotations (COCO category 8) are excluded from the inference
cohort. They may appear visually inside crops as occlusion/context.

Usage:
    python prepare_roofsense_coco_crops.py --dataset-root <path> --output-dir <path> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from coco_crop_utils import (
    MANIFEST_COLUMNS,
    assign_probable_building_groups,
    crop_with_constant_boundary_padding,
    expand_rect_to_square,
    fingerprint_file,
    fingerprint_strings,
    make_sample_id,
    polygon_bbox,
    read_rgb_bands,
    write_json_metadata,
    write_manifest,
    write_prepared_jpeg,
)
from coco_parser import (
    COCO_CATEGORY_MAP,
    INVALID_COCO_ID,
    MASK_ID_TO_LABEL,
    MAPPING_VERSION,
    REMOTECLIP_MAPPING,
    SUPPORTED_COCO_IDS,
    CocoData,
    compute_mask_agreement,
    load_names,
    load_splits,
    rasterize_polygon_mask,
    read_semantic_mask,
)
from diagnostics import (
    LOW_TARGET_OCCUPANCY_THRESHOLD,
    MASK_DISAGREEMENT_THRESHOLD,
    compute_mask_agreement_flag,
    measure_crop_composition,
    build_contact_sheet_clean,
    build_contact_sheet_diagnostics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
SCRIPT_VERSION = "3.2"  # v3.2: allow elongated roof-scale annotations
TILE_SIZE = 512
SOLAR_PANEL_COCO_ID = 8
MIN_ROOF_SHORT_SIDE_PX = 16
MIN_ROOF_BBOX_AREA_PX = 2048
MAX_CONTAINED_AREA_RATIO = 0.25


def roof_scale_rejection_reason(
    bbox: tuple[int, int, int, int],
    annotation: dict,
    image_annotations: list[dict],
) -> str | None:
    """Return why an annotation is not a credible whole-roof proxy."""
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    area = width * height
    if min(width, height) < MIN_ROOF_SHORT_SIDE_PX:
        return f"bbox_short_side_lt_{MIN_ROOF_SHORT_SIDE_PX}"
    if area < MIN_ROOF_BBOX_AREA_PX:
        return f"bbox_area_lt_{MIN_ROOF_BBOX_AREA_PX}"

    for other in image_annotations:
        if other["id"] == annotation["id"]:
            continue
        category_id = other["category_id"]
        if category_id not in SUPPORTED_COCO_IDS or category_id == SOLAR_PANEL_COCO_ID:
            continue
        segmentation = other.get("segmentation", [])
        if not segmentation or not isinstance(segmentation[0], list):
            continue
        try:
            other_left, other_top, other_right, other_bottom = polygon_bbox(segmentation[0])
        except ValueError:
            continue
        other_area = (other_right - other_left) * (other_bottom - other_top)
        contained = (
            left >= other_left and top >= other_top
            and right <= other_right and bottom <= other_bottom
        )
        if contained and other_area > 0 and area / other_area <= MAX_CONTAINED_AREA_RATIO:
            return "contained_in_larger_material_region"
    return None


# ---------------------------------------------------------------------------
# Preparation logic
# ---------------------------------------------------------------------------


def run_preparation(
    dataset_root: Path,
    output_dir: Path,
    padding_px: int = 8,
    jpeg_quality: int = 95,
    contact_sheet_per_class: int = 4,
    reset: bool = False,
) -> dict:
    """Run the full RoofSense COCO-crop preparation pipeline."""
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    annotations_path = dataset_root / "annotations" / "annotations.json"
    splits_path = dataset_root / "splits.json"
    names_path = dataset_root / "names.json"

    # ------------------------------------------------------------------
    # Validate inputs exist
    # ------------------------------------------------------------------
    for p, label in [
        (images_dir, "images directory"),
        (masks_dir, "masks directory"),
        (annotations_path, "annotations.json"),
        (splits_path, "splits.json"),
        (names_path, "names.json"),
    ]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Check for incompatible resume
    # ------------------------------------------------------------------
    metadata_path = output_dir / "preparation_metadata.json"
    if output_dir.exists() and metadata_path.exists():
        if reset:
            _safe_reset(output_dir)
        else:
            _check_compatible_resume(metadata_path, dataset_root, padding_px, jpeg_quality)
    elif reset and output_dir.exists():
        _safe_reset(output_dir)

    # ------------------------------------------------------------------
    # Discover source TIFFs (reject sidecars)
    # ------------------------------------------------------------------
    tiff_files = sorted(images_dir.glob("*.tif"))
    tiff_files = [f for f in tiff_files if not f.name.endswith(".tif.aux.xml")]
    print(f"Discovered {len(tiff_files)} source TIFFs (excluded .tif.aux.xml sidecars)")

    tiff_names = {f.stem + ".tif" for f in tiff_files}

    # ------------------------------------------------------------------
    # Load COCO annotations
    # ------------------------------------------------------------------
    coco = CocoData(annotations_path)
    coco_errors = coco.validate()
    if coco_errors:
        print("ERROR: COCO validation failed:")
        for e in coco_errors:
            print(f"  {e}")
        sys.exit(1)

    print(f"COCO: {len(coco.images)} images, {len(coco.annotations)} annotations")

    # ------------------------------------------------------------------
    # Load splits and names
    # ------------------------------------------------------------------
    splits = load_splits(splits_path)
    names = load_names(names_path)
    total_split_files = sum(len(v) for v in splits.values())
    print(f"Splits: {total_split_files} files ({', '.join(f'{k}={len(v)}' for k, v in splits.items())})")

    # ------------------------------------------------------------------
    # Reconcile: 300 TIFFs == 300 COCO images == 300 split assignments
    # ------------------------------------------------------------------
    coco_tiff_names = set()
    unmatched_coco = []
    for img in coco.images.values():
        tif_name = img["file_name"].replace(".png", ".tif")
        if tif_name in tiff_names:
            coco_tiff_names.add(tif_name)
        else:
            unmatched_coco.append(img["file_name"])

    if unmatched_coco:
        print(f"ERROR: {len(unmatched_coco)} COCO images have no matching TIFF:")
        for f in unmatched_coco[:10]:
            print(f"  {f}")
        sys.exit(1)

    tiff_without_coco = tiff_names - coco_tiff_names
    if tiff_without_coco:
        print(f"ERROR: {len(tiff_without_coco)} TIFFs have no COCO image record")
        sys.exit(1)

    tif_split_map: dict[str, str] = {}
    for split_name, files in splits.items():
        for f in files:
            tif_split_map[f] = split_name

    # ------------------------------------------------------------------
    # Count annotations by category
    # ------------------------------------------------------------------
    cat_counts = Counter(a["category_id"] for a in coco.annotations.values())
    invalid_count = cat_counts.get(INVALID_COCO_ID, 0)
    solar_panel_count = cat_counts.get(SOLAR_PANEL_COCO_ID, 0)
    excluded_count = invalid_count + solar_panel_count
    non_roof_scale_count = 0
    supported_ann_count = sum(
        v for k, v in cat_counts.items()
        if k in SUPPORTED_COCO_IDS and k != SOLAR_PANEL_COCO_ID
    )
    print(f"Annotations: {invalid_count} Invalid, {solar_panel_count} Solar Panel (excluded), "
          f"{supported_ann_count} supported")

    # ------------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------------
    prepared_images_dir = output_dir / "images"
    for cat_id, (_, _, dirname) in COCO_CATEGORY_MAP.items():
        if dirname is not None and cat_id != SOLAR_PANEL_COCO_ID:
            (prepared_images_dir / dirname).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Prepare each annotation
    # ------------------------------------------------------------------
    manifest_rows = []
    prepared_count = 0
    failed_count = 0
    flagged_count = 0

    image_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[str, np.ndarray | None] = {}

    seen_sample_ids: set[str] = set()
    seen_filenames: set[str] = set()

    annotation_items = sorted(coco.annotations.items(), key=lambda x: x[0])
    total = len(annotation_items)

    for i, (ann_id, ann) in enumerate(annotation_items):
        cat_id = ann["category_id"]

        img_record = coco.images.get(ann["image_id"])
        excluded_provenance = {}
        if img_record is not None:
            excluded_tiff = coco.resolve_tiff_path(img_record, images_dir)
            excluded_relpath = str(excluded_tiff.relative_to(dataset_root))
            excluded_split = coco.resolve_split(img_record, splits) or ""
            excluded_provenance = {
                "sample_id": make_sample_id(
                    excluded_relpath, ann["image_id"], ann_id, cat_id
                ),
                "source_relpath": excluded_relpath,
                "source_stem": excluded_tiff.stem,
                "split": excluded_split,
            }

        # ------------------------------------------------------------------
        # Handle Invalid category
        # ------------------------------------------------------------------
        if cat_id == INVALID_COCO_ID:
            cat_info = COCO_CATEGORY_MAP[cat_id]
            manifest_rows.append(_make_excluded_row(
                ann, cat_id, cat_info[1], "Invalid category (COCO ID 4)",
                **excluded_provenance,
            ))
            continue

        # ------------------------------------------------------------------
        # Handle Solar Panel exclusion
        # ------------------------------------------------------------------
        if cat_id == SOLAR_PANEL_COCO_ID:
            cat_info = COCO_CATEGORY_MAP[cat_id]
            manifest_rows.append(_make_excluded_row(
                ann, cat_id, cat_info[1],
                "Solar Panel excluded from inference cohort",
                **excluded_provenance,
            ))
            continue

        # ------------------------------------------------------------------
        # Validate supported category
        # ------------------------------------------------------------------
        if cat_id not in SUPPORTED_COCO_IDS:
            manifest_rows.append(_make_failed_row(
                ann, cat_id, "unknown", "", "",
                f"Unsupported COCO category ID {cat_id}"
            ))
            failed_count += 1
            continue

        cat_info = COCO_CATEGORY_MAP[cat_id]
        mask_id, roofsense_label, dirname = cat_info
        mapped_target = REMOTECLIP_MAPPING[roofsense_label]

        # ------------------------------------------------------------------
        # Resolve image record and source files
        # ------------------------------------------------------------------
        img_record = coco.images.get(ann["image_id"])
        if img_record is None:
            manifest_rows.append(_make_failed_row(
                ann, cat_id, roofsense_label, mask_id, mapped_target,
                f"Image ID {ann['image_id']} not found in COCO"
            ))
            failed_count += 1
            continue

        tiff_path = coco.resolve_tiff_path(img_record, images_dir)
        source_relpath = str(tiff_path.relative_to(dataset_root))
        source_stem = tiff_path.stem

        if not tiff_path.exists():
            manifest_rows.append(_make_failed_row(
                ann, cat_id, roofsense_label, mask_id, mapped_target,
                f"Source TIFF not found: {tiff_path}",
                source_relpath=source_relpath, source_stem=source_stem,
            ))
            failed_count += 1
            continue

        split = coco.resolve_split(img_record, splits)
        if split is None:
            manifest_rows.append(_make_failed_row(
                ann, cat_id, roofsense_label, mask_id, mapped_target,
                f"No split assignment for {tiff_path.name}",
                source_relpath=source_relpath, source_stem=source_stem,
            ))
            failed_count += 1
            continue

        sample_id = make_sample_id(source_relpath, ann["image_id"], ann_id, cat_id)
        if sample_id in seen_sample_ids:
            manifest_rows.append(_make_failed_row(
                ann, cat_id, roofsense_label, mask_id, mapped_target,
                f"Duplicate sample_id: {sample_id}",
                sample_id=sample_id, source_relpath=source_relpath,
                source_stem=source_stem, split=split,
            ))
            failed_count += 1
            continue
        seen_sample_ids.add(sample_id)

        # ------------------------------------------------------------------
        # Validate geometry
        # ------------------------------------------------------------------
        segmentation = ann.get("segmentation", [])
        if not segmentation or not isinstance(segmentation[0], list):
            manifest_rows.append(_make_failed_row(
                ann, cat_id, roofsense_label, mask_id, mapped_target,
                "Empty or invalid segmentation",
                sample_id=sample_id, source_relpath=source_relpath,
                source_stem=source_stem, split=split,
            ))
            failed_count += 1
            continue

        try:
            seg_flat = segmentation[0]
            poly_left, poly_top, poly_right, poly_bottom = polygon_bbox(seg_flat)
        except ValueError as e:
            manifest_rows.append(_make_failed_row(
                ann, cat_id, roofsense_label, mask_id, mapped_target,
                f"Invalid polygon geometry: {e}",
                sample_id=sample_id, source_relpath=source_relpath,
                source_stem=source_stem, split=split,
            ))
            failed_count += 1
            continue

        tile_annotations = coco.image_annotations.get(ann["image_id"], [])
        roof_scale_reason = roof_scale_rejection_reason(
            (poly_left, poly_top, poly_right, poly_bottom),
            ann,
            tile_annotations,
        )
        if roof_scale_reason is not None:
            row = _make_excluded_row(
                ann,
                cat_id,
                roofsense_label,
                roof_scale_reason,
                sample_id=sample_id,
                source_relpath=source_relpath,
                source_stem=source_stem,
                split=split,
                status="excluded_non_roof_scale",
            )
            row.update({
                "semantic_mask_class_id": mask_id,
                "mapped_remoteclip_class": mapped_target,
                "polygon_bbox_x": poly_left,
                "polygon_bbox_y": poly_top,
                "polygon_bbox_width": poly_right - poly_left,
                "polygon_bbox_height": poly_bottom - poly_top,
                "crop_method": "annotation_bbox",
            })
            manifest_rows.append(row)
            non_roof_scale_count += 1
            excluded_count += 1
            continue

        try:
            # ------------------------------------------------------------------
            # Load source image (RGB bands 1-3 only)
            # ------------------------------------------------------------------
            tiff_key = str(tiff_path)
            if tiff_key not in image_cache:
                rgb = read_rgb_bands(tiff_path)
                if rgb.shape[0] != TILE_SIZE or rgb.shape[1] != TILE_SIZE:
                    raise ValueError(
                        f"Unexpected image dimensions {rgb.shape[:2]}, expected {TILE_SIZE}x{TILE_SIZE}"
                    )
                image_cache[tiff_key] = rgb
            rgb = image_cache[tiff_key]

            # ------------------------------------------------------------------
            # Treat the material annotation as the building proxy. Expand its
            # bounding box to a square, then add fixed visual context.
            # ------------------------------------------------------------------
            sq_left, sq_top, sq_right, sq_bottom = expand_rect_to_square(
                poly_left, poly_top, poly_right, poly_bottom,
                TILE_SIZE, TILE_SIZE,
            )
            # Add padding (may go beyond tile boundary)
            crop_left = sq_left - padding_px
            crop_top = sq_top - padding_px
            crop_right = sq_right + padding_px
            crop_bottom = sq_bottom + padding_px

            # Target square size before clipping
            target_size = max(crop_right - crop_left, crop_bottom - crop_top)

            # Extract crop with constant (black) boundary padding
            crop_rgb = crop_with_constant_boundary_padding(
                rgb, crop_left, crop_top, crop_right, crop_bottom, target_size,
            )
            crop_img = Image.fromarray(crop_rgb, "RGB")

            # Actual crop dimensions after boundary padding
            crop_width = crop_right - crop_left
            crop_height = crop_bottom - crop_top
            square_size = target_size

            # ------------------------------------------------------------------
            # Rasterize target polygon for crop-relative diagnostics
            # ------------------------------------------------------------------
            # Offset for crop-relative coordinates
            target_mask = rasterize_polygon_mask(
                seg_flat, crop_width, crop_height,
                left=crop_left, top=crop_top,
            )

            # ------------------------------------------------------------------
            # Find all annotations intersecting this crop rectangle
            # ------------------------------------------------------------------
            other_ann_masks: dict[int, tuple[np.ndarray, int]] = {}
            for other_ann in tile_annotations:
                other_ann_id = other_ann["id"]
                if other_ann_id == ann_id:
                    other_ann_masks[ann_id] = (target_mask, cat_id)
                    continue
                other_seg = other_ann.get("segmentation", [])
                if not other_seg or not isinstance(other_seg[0], list):
                    continue
                other_flat = other_seg[0]
                try:
                    o_left, o_top, o_right, o_bottom = polygon_bbox(other_flat)
                except ValueError:
                    continue
                if (o_right <= crop_left or o_left >= crop_right or
                    o_bottom <= crop_top or o_top >= crop_bottom):
                    continue
                other_mask = rasterize_polygon_mask(
                    other_flat, crop_width, crop_height,
                    left=crop_left, top=crop_top,
                )
                other_ann_masks[other_ann_id] = (other_mask, other_ann["category_id"])

            # ------------------------------------------------------------------
            # Semantic mask agreement
            # ------------------------------------------------------------------
            mask_path = masks_dir / tiff_path.name
            if tiff_key not in mask_cache:
                if mask_path.exists():
                    mask_cache[tiff_key] = read_semantic_mask(mask_path)
                else:
                    mask_cache[tiff_key] = None

            sem_mask = mask_cache.get(tiff_key)
            if sem_mask is not None:
                # Clip target mask to tile-interior portion (semantic mask
                # only covers the tile, not the constant-boundary-padding zone)
                tile_clip_top = max(0, crop_top) - crop_top
                tile_clip_left = max(0, crop_left) - crop_left
                tile_clip_bottom = min(TILE_SIZE, crop_bottom) - crop_top
                tile_clip_right = min(TILE_SIZE, crop_right) - crop_left
                clipped_target_mask = target_mask[
                    tile_clip_top:tile_clip_bottom,
                    tile_clip_left:tile_clip_right,
                ]
                total_valid, agreeing, agreement_fraction = compute_mask_agreement(
                    sem_mask, clipped_target_mask, cat_id,
                    max(0, crop_left), max(0, crop_top),
                    tile_clip_right - tile_clip_left, tile_clip_bottom - tile_clip_top,
                )
            else:
                agreement_fraction = 0.0
            coco_mask_agree, coco_mask_disagree = compute_mask_agreement_flag(agreement_fraction)

            # ------------------------------------------------------------------
            # Crop composition diagnostics
            # ------------------------------------------------------------------
            crop_for_nodata = rgb[
                max(0, crop_top):min(TILE_SIZE, crop_bottom),
                max(0, crop_left):min(TILE_SIZE, crop_right),
            ]
            if crop_for_nodata.size > 0:
                nodata_mask = np.all(crop_for_nodata == 0, axis=2)
                # Pad nodata mask to match crop dimensions
                full_nodata = np.ones((crop_height, crop_width), dtype=bool)
                src_top = max(0, crop_top) - crop_top
                src_left = max(0, crop_left) - crop_left
                full_nodata[src_top:src_top + nodata_mask.shape[0],
                            src_left:src_left + nodata_mask.shape[1]] = nodata_mask
                nodata_mask = full_nodata
            else:
                nodata_mask = np.zeros((crop_height, crop_width), dtype=bool)

            composition = measure_crop_composition(
                crop_width, crop_height,
                target_mask, other_ann_masks,
                ann_id, cat_id,
                nodata_mask,
            )

            composition["coco_mask_agreement"] = coco_mask_agree
            composition["coco_mask_disagreement"] = coco_mask_disagree

            composition["shared_annotation_count"] = len(other_ann_masks)
            composition["multi_material_component"] = False
            composition["potential_building_merge"] = False

            # Count flags
            row_flagged = any([
                composition["multi_label_overlap"],
                composition["other_class_in_context"],
                composition["same_class_neighbor"],
                composition["low_target_occupancy"],
                coco_mask_disagree,
                composition["contains_nodata"],
            ])
            if row_flagged:
                flagged_count += 1

            # ------------------------------------------------------------------
            # Write JPEG
            # ------------------------------------------------------------------
            prepared_filename_base = f"{sample_id}.jpg"
            if prepared_filename_base in seen_filenames:
                raise FileExistsError(f"Duplicate filename: {prepared_filename_base}")
            seen_filenames.add(prepared_filename_base)

            class_dir = prepared_images_dir / dirname
            jpeg_path = write_prepared_jpeg(crop_img, class_dir, sample_id, jpeg_quality)
            prepared_filename = f"{dirname}/{prepared_filename_base}"

            # ------------------------------------------------------------------
            # Build manifest row
            # ------------------------------------------------------------------
            row = {
                "sample_id": sample_id,
                "source_relpath": source_relpath,
                "source_stem": source_stem,
                "coco_image_id": ann["image_id"],
                "coco_annotation_id": ann_id,
                "roofsense_class_id": cat_id,
                "roofsense_class": roofsense_label,
                "semantic_mask_class_id": mask_id,
                "mapped_remoteclip_class": mapped_target,
                "mapping_version": MAPPING_VERSION,
                "split": split,
                "prepared_filename": prepared_filename,
                "crop_method": "annotation_bbox",
                "polygon_bbox_x": poly_left,
                "polygon_bbox_y": poly_top,
                "polygon_bbox_width": poly_right - poly_left,
                "polygon_bbox_height": poly_bottom - poly_top,
                "footprint_left": poly_left,
                "footprint_top": poly_top,
                "footprint_right": poly_right,
                "footprint_bottom": poly_bottom,
                "crop_left": crop_left,
                "crop_top": crop_top,
                "crop_right": crop_right,
                "crop_bottom": crop_bottom,
                "padding_px": padding_px,
                "square_size": square_size,
                "component_pixel_count": "",
                "status": "prepared",
                "error": "",
            }
            row.update(composition)
            manifest_rows.append(row)
            prepared_count += 1

        except Exception as exc:
            manifest_rows.append(_make_failed_row(
                ann, cat_id, roofsense_label, mask_id, mapped_target,
                str(exc),
                sample_id=sample_id, source_relpath=source_relpath,
                source_stem=source_stem, split=split,
            ))
            failed_count += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total}")

    # ------------------------------------------------------------------
    # Group near-duplicate annotation crops that probably depict one building.
    # ------------------------------------------------------------------
    assign_probable_building_groups(manifest_rows, iou_threshold=0.8)

    # ------------------------------------------------------------------
    # Write manifest (atomic)
    # ------------------------------------------------------------------
    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"Wrote manifest: {manifest_path}")

    manifest_fingerprint = fingerprint_file(manifest_path)

    # ------------------------------------------------------------------
    # Build contact sheets
    # ------------------------------------------------------------------
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    build_contact_sheet_clean(
        prepared_images_dir, manifest_rows, contact_sheet_path,
        per_class=contact_sheet_per_class,
    )
    print(f"Wrote contact sheet: {contact_sheet_path}")

    contact_sheet_diag_path = output_dir / "contact_sheet_diagnostics.jpg"
    build_contact_sheet_diagnostics(
        prepared_images_dir, dataset_root, coco,
        manifest_rows, contact_sheet_diag_path,
        per_class=contact_sheet_per_class,
    )
    print(f"Wrote diagnostics contact sheet: {contact_sheet_diag_path}")

    # ------------------------------------------------------------------
    # Preparation metadata (atomic)
    # ------------------------------------------------------------------
    ann_fp = fingerprint_file(annotations_path)
    splits_fp = fingerprint_file(splits_path)
    names_fp = fingerprint_file(names_path)

    tiff_identities = sorted(
        f"{f.relative_to(dataset_root)}:{fingerprint_file(f)}" for f in tiff_files
    )
    tiff_fp = fingerprint_strings(tiff_identities)

    class_counts: dict[str, dict[str, int]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for row in manifest_rows:
        cls = row.get("roofsense_class", "unknown")
        sp = row.get("split", "unknown")
        status = row.get("status", "unknown")
        class_counts.setdefault(cls, Counter())[status] += 1
        split_counts.setdefault(sp, Counter())[status] += 1

    flag_combos: Counter[tuple[str, ...]] = Counter()
    for row in manifest_rows:
        if row.get("status") != "prepared":
            continue
        active = []
        for f in ("multi_label_overlap", "other_class_in_context", "same_class_neighbor",
                   "low_target_occupancy", "coco_mask_disagreement", "contains_nodata",
                   "multi_material_component", "potential_building_merge"):
            val = row.get(f, False)
            is_flagged = val if isinstance(val, bool) else str(val).lower() in ("true", "1")
            if is_flagged:
                active.append(f)
        if active:
            flag_combos[tuple(active)] += 1

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "fingerprints": {
            "annotations_json": ann_fp,
            "splits_json": splits_fp,
            "names_json": names_fp,
            "source_tiffs": tiff_fp,
            "manifest": manifest_fingerprint,
        },
        "settings": {
            "padding_px": padding_px,
            "jpeg_quality": jpeg_quality,
            "tile_size": TILE_SIZE,
            "rgb_bands": [1, 2, 3],
            "non_finite_replacement": 0,
            "crop_method": "annotation_bbox",
            "probable_building_iou_threshold": 0.8,
            "minimum_roof_short_side_px": MIN_ROOF_SHORT_SIDE_PX,
            "minimum_roof_bbox_area_px": MIN_ROOF_BBOX_AREA_PX,
            "maximum_contained_area_ratio": MAX_CONTAINED_AREA_RATIO,
            "square_expansion": "source_coordinates",
            "boundary_padding": "constant_black",
            "polygon_rasterization": "PIL_ImageDraw_polygon",
            "diagnostic_thresholds": {
                "low_target_occupancy": 0.20,
                "mask_disagreement": 0.90,
            },
        },
        "class_mapping": {
            "native": {str(k): v[1] for k, v in COCO_CATEGORY_MAP.items()},
            "remoteclip": REMOTECLIP_MAPPING,
            "mapping_version": MAPPING_VERSION,
        },
        "exclusions": {
            "invalid_coco_id_4": invalid_count,
            "solar_panel_coco_id_8": solar_panel_count,
            "non_roof_scale": non_roof_scale_count,
            "total_excluded": excluded_count,
        },
        "counts": {
            "source_tiffs": len(tiff_files),
            "coco_images": len(coco.images),
            "coco_annotations": len(coco.annotations),
            "invalid_excluded": invalid_count,
            "solar_panel_excluded": solar_panel_count,
            "non_roof_scale_excluded": non_roof_scale_count,
            "supported_annotations": supported_ann_count,
            "prepared": prepared_count,
            "flagged": flagged_count,
            "excluded": excluded_count,
            "failed": failed_count,
            "probable_building_groups": len({
                row.get("probable_building_group_id")
                for row in manifest_rows if row.get("status") == "prepared"
            }),
            "ambiguous_probable_building_groups": len({
                row.get("probable_building_group_id")
                for row in manifest_rows
                if row.get("status") == "prepared"
                and row.get("probable_building_group_ambiguous")
            }),
        },
        "per_class_counts": {k: dict(v) for k, v in sorted(class_counts.items())},
        "per_split_counts": {k: dict(v) for k, v in sorted(split_counts.items())},
        "flag_combinations": {",".join(k): v for k, v in flag_combos.most_common(20)},
    }

    write_json_metadata(metadata_path, metadata)
    print(f"Wrote metadata: {metadata_path}")

    # ------------------------------------------------------------------
    # Console diagnostics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Source TIFFs:        {len(tiff_files)}")
    print(f"  COCO images:         {len(coco.images)}")
    print(f"  COCO annotations:    {len(coco.annotations)}")
    print(f"  Invalid (excluded):  {invalid_count}")
    print(f"  Solar Panel (excl):  {solar_panel_count}")
    print(f"  Non-roof-scale:      {non_roof_scale_count}")
    print(f"  Supported:           {supported_ann_count}")
    print(f"  Prepared:            {prepared_count}")
    print(f"  Flagged:             {flagged_count}")
    print(f"  Failed:              {failed_count}")
    print(f"  Manifest:            {manifest_path}")
    print(f"  Metadata:            {metadata_path}")
    print(f"  Contact sheet:       {contact_sheet_path}")
    print(f"  Diag contact sheet:  {contact_sheet_diag_path}")

    print(f"\nPer-class breakdown:")
    for cls in sorted(class_counts):
        counts = class_counts[cls]
        print(f"  {cls}: {dict(counts)}")

    print(f"\nPer-split breakdown:")
    for sp in sorted(split_counts):
        counts = split_counts[sp]
        print(f"  {sp}: {dict(counts)}")

    if flag_combos:
        print(f"\nTop flag combinations:")
        for flags, count in flag_combos.most_common(10):
            print(f"  {','.join(flags)}: {count}")

    if prepared_count > 0:
        fractions = [float(r.get("target_crop_fraction", 0)) for r in manifest_rows if r.get("status") == "prepared"]
        print(f"\nTarget occupancy distribution (prepared):")
        print(f"  min={min(fractions):.4f} median={np.median(fractions):.4f} max={max(fractions):.4f}")

    has_failures = failed_count > 0
    if has_failures:
        print(f"\nWARNING: {failed_count} annotations failed. See manifest for details.")
        sys.exit(1)

    return {
        "discovered": len(tiff_files),
        "prepared": prepared_count,
        "excluded": excluded_count,
        "flagged": flagged_count,
        "failed": failed_count,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
    }


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _make_excluded_row(
    ann: dict,
    cat_id: int,
    label: str,
    reason: str,
    sample_id: str = "",
    source_relpath: str = "",
    source_stem: str = "",
    split: str = "",
    status: str = "excluded",
) -> dict:
    """Build a manifest row for an excluded annotation."""
    return {
        "sample_id": sample_id,
        "source_relpath": source_relpath,
        "source_stem": source_stem,
        "coco_image_id": ann["image_id"],
        "coco_annotation_id": ann["id"],
        "roofsense_class_id": cat_id,
        "roofsense_class": label,
        "semantic_mask_class_id": "",
        "mapped_remoteclip_class": "",
        "mapping_version": MAPPING_VERSION,
        "split": split,
        "prepared_filename": "",
        "status": status,
        "error": reason,
    }


def _make_failed_row(
    ann: dict, cat_id: int, label: str, mask_id, mapped: str, error: str,
    sample_id: str = "", source_relpath: str = "", source_stem: str = "",
    split: str = "",
) -> dict:
    """Build a manifest row for a failed annotation."""
    return {
        "sample_id": sample_id,
        "source_relpath": source_relpath,
        "source_stem": source_stem,
        "coco_image_id": ann["image_id"],
        "coco_annotation_id": ann["id"],
        "roofsense_class_id": cat_id,
        "roofsense_class": label,
        "semantic_mask_class_id": mask_id,
        "mapped_remoteclip_class": mapped,
        "mapping_version": MAPPING_VERSION,
        "split": split,
        "prepared_filename": "",
        "status": "failed",
        "error": error,
    }


# ---------------------------------------------------------------------------
# Resume / reset helpers
# ---------------------------------------------------------------------------


def _safe_reset(output_dir: Path) -> None:
    """Remove recognized generated files inside the output directory."""
    import shutil

    recognized = {
        "images",
        "manifest.csv",
        "preparation_metadata.json",
        "contact_sheet.jpg",
        "contact_sheet_diagnostics.jpg",
    }

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return

    metadata_path = output_dir / "preparation_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Refusing reset: {output_dir} is not a recognized preparation output"
        ) from exc
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or not metadata.get("script_version")
        or metadata.get("dataset_root") is None
    ):
        raise RuntimeError(
            f"Refusing reset: {output_dir} is not a recognized preparation output"
        )

    existing = set(p.name for p in output_dir.iterdir())
    unrecognized = {f for f in existing - recognized if not f.startswith(".")}
    if unrecognized:
        print(f"WARNING: Output directory contains unrecognized files: {unrecognized}")
        print("Proceeding with removal of recognized files only.")

    for item in recognized:
        item_path = output_dir / item
        if item_path.exists():
            if item_path.is_dir():
                shutil.rmtree(item_path)
            else:
                item_path.unlink()

    print(f"Reset: removed recognized files from {output_dir}")


def _check_compatible_resume(
    metadata_path: Path,
    dataset_root: Path,
    padding_px: int,
    jpeg_quality: int,
) -> None:
    """Check if existing preparation metadata is compatible."""
    existing = json.loads(metadata_path.read_text())

    errors = []
    expected_settings = {
        "padding_px": padding_px,
        "jpeg_quality": jpeg_quality,
        "tile_size": TILE_SIZE,
        "rgb_bands": [1, 2, 3],
        "crop_method": "annotation_bbox",
        "probable_building_iou_threshold": 0.8,
        "minimum_roof_short_side_px": MIN_ROOF_SHORT_SIDE_PX,
        "minimum_roof_bbox_area_px": MIN_ROOF_BBOX_AREA_PX,
        "maximum_contained_area_ratio": MAX_CONTAINED_AREA_RATIO,
        "square_expansion": "source_coordinates",
        "boundary_padding": "constant_black",
        "diagnostic_thresholds": {
            "low_target_occupancy": LOW_TARGET_OCCUPANCY_THRESHOLD,
            "mask_disagreement": MASK_DISAGREEMENT_THRESHOLD,
        },
    }
    if existing.get("dataset_root") != str(dataset_root.resolve()):
        errors.append(f"dataset_root mismatch: {existing.get('dataset_root')} vs {dataset_root}")
    if existing.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if existing.get("script_version") != SCRIPT_VERSION:
        errors.append("script_version mismatch")
    for key, expected in expected_settings.items():
        if existing.get("settings", {}).get(key) != expected:
            errors.append(f"settings.{key} mismatch")
    mapping = existing.get("class_mapping", {})
    if mapping.get("remoteclip") != REMOTECLIP_MAPPING:
        errors.append("RemoteCLIP mapping mismatch")
    if mapping.get("mapping_version") != MAPPING_VERSION:
        errors.append("mapping_version mismatch")

    annotations_path = dataset_root / "annotations" / "annotations.json"
    splits_path = dataset_root / "splits.json"
    names_path = dataset_root / "names.json"
    tiff_files = sorted((dataset_root / "images").glob("*.tif"))
    current_fingerprints = {
        "annotations_json": fingerprint_file(annotations_path),
        "splits_json": fingerprint_file(splits_path),
        "names_json": fingerprint_file(names_path),
        "source_tiffs": fingerprint_strings([
            f"{path.relative_to(dataset_root)}:{fingerprint_file(path)}"
            for path in tiff_files
        ]),
    }
    for key, expected in current_fingerprints.items():
        if existing.get("fingerprints", {}).get(key) != expected:
            errors.append(f"fingerprints.{key} mismatch")

    if errors:
        print("ERROR: Incompatible existing preparation found:")
        for e in errors:
            print(f"  {e}")
        print("Choose a new output directory or use --reset to overwrite.")
        sys.exit(1)

    print("Existing preparation is compatible. Use --reset to regenerate.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare RoofSense COCO-crop RGB JPEGs for RemoteCLIP inference."
    )
    parser.add_argument(
        "--dataset-root", type=str, required=True,
        help="Path to RoofSense dataset root",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for prepared JPEGs, manifest, and metadata",
    )
    parser.add_argument(
        "--padding-px", type=int, default=8,
        help="Context padding in pixels (default: 8)",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG quality (default: 95)",
    )
    parser.add_argument(
        "--contact-sheet-per-class", type=int, default=4,
        help="Samples per class on contact sheets (default: 4)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Remove existing output before starting",
    )
    args = parser.parse_args()

    run_preparation(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        padding_px=args.padding_px,
        jpeg_quality=args.jpeg_quality,
        contact_sheet_per_class=args.contact_sheet_per_class,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
