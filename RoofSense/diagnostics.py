"""
diagnostics.py
==============
Crop composition diagnostics, ambiguity flagging, and contact-sheet generation
for the RoofSense COCO-crop preparation pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coco_crop_utils import write_prepared_jpeg

# ---------------------------------------------------------------------------
# Diagnostic thresholds (QA defaults, not assertions)
# ---------------------------------------------------------------------------

LOW_TARGET_OCCUPANCY_THRESHOLD = 0.20
MASK_DISAGREEMENT_THRESHOLD = 0.90  # flag if agreement < this


# ---------------------------------------------------------------------------
# Crop composition measurement
# ---------------------------------------------------------------------------


def measure_crop_composition(
    crop_width: int,
    crop_height: int,
    target_mask: np.ndarray,
    all_annotation_masks: dict[int, tuple[np.ndarray, int]],
    target_annotation_id: int,
    target_category_id: int,
    nodata_mask: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    """Measure crop composition and assign diagnostic flags.

    Parameters
    ----------
    crop_width, crop_height : int
        Dimensions of the unpadded crop rectangle.
    target_mask : np.ndarray
        Boolean mask of the target polygon within the crop.
    all_annotation_masks : dict
        {annotation_id: (mask, category_id)} for all annotations
        intersecting the crop.
    target_annotation_id : int
        The annotation ID being measured.
    target_category_id : int
        COCO category ID of the target annotation.
    nodata_mask : np.ndarray or None
        Boolean mask where True = nodata pixel in the source RGB.

    Returns
    -------
    dict with all measurement and flag fields.
    """
    crop_area = crop_width * crop_height
    if crop_area == 0:
        raise ValueError("Zero-area crop")

    # Target measurements
    target_pixels = int(target_mask.sum())
    target_fraction = target_pixels / crop_area

    # Collect other annotations
    same_class_other_masks = []
    different_class_masks = []
    same_class_ann_ids = []
    different_class_ann_ids = []

    for ann_id, (mask, cat_id) in all_annotation_masks.items():
        if ann_id == target_annotation_id:
            continue
        if cat_id == target_category_id:
            same_class_other_masks.append(mask)
            same_class_ann_ids.append(ann_id)
        else:
            different_class_masks.append(mask)
            different_class_ann_ids.append(ann_id)

    # Same-class other area (union to avoid double-counting)
    if same_class_other_masks:
        same_union = np.zeros_like(target_mask)
        for m in same_class_other_masks:
            same_union |= m
        same_class_other_pixels = int(same_union.sum())
    else:
        same_class_other_pixels = 0
    same_class_other_fraction = same_class_other_pixels / crop_area

    # Different-class area (union)
    if different_class_masks:
        diff_union = np.zeros_like(target_mask)
        for m in different_class_masks:
            diff_union |= m
        different_class_pixels = int(diff_union.sum())
    else:
        different_class_pixels = 0
    different_class_fraction = different_class_pixels / crop_area

    # Total annotated roof fraction (target + all others, unioned)
    all_union = target_mask.copy()
    for m in same_class_other_masks:
        all_union |= m
    for m in different_class_masks:
        all_union |= m
    annotated_roof_pixels = int(all_union.sum())
    annotated_roof_fraction = annotated_roof_pixels / crop_area

    # Multi-label overlap: does any different-class polygon overlap the target?
    multi_label_overlap = False
    if different_class_masks:
        for m in different_class_masks:
            if (m & target_mask).any():
                multi_label_overlap = True
                break

    # Nodata
    if nodata_mask is not None:
        nodata_pixels = int(nodata_mask.sum())
        nodata_fraction = nodata_pixels / crop_area
    else:
        nodata_fraction = 0.0
    contains_nodata = nodata_fraction > 0.0

    # Flags
    low_target_occupancy = target_fraction < LOW_TARGET_OCCUPANCY_THRESHOLD
    other_class_in_context = len(different_class_ann_ids) > 0
    same_class_neighbor = len(same_class_ann_ids) > 0

    return {
        "target_pixels": target_pixels,
        "target_crop_fraction": round(target_fraction, 6),
        "annotated_roof_crop_fraction": round(annotated_roof_fraction, 6),
        "same_class_other_pixels": same_class_other_pixels,
        "same_class_other_fraction": round(same_class_other_fraction, 6),
        "different_class_pixels": different_class_pixels,
        "different_class_fraction": round(different_class_fraction, 6),
        "same_class_annotation_count": len(same_class_ann_ids),
        "different_class_annotation_count": len(different_class_ann_ids),
        "multi_label_overlap": multi_label_overlap,
        "other_class_in_context": other_class_in_context,
        "same_class_neighbor": same_class_neighbor,
        "low_target_occupancy": low_target_occupancy,
        "contains_nodata": contains_nodata,
        "nodata_fraction": round(nodata_fraction, 6),
    }


def compute_mask_agreement_flag(
    agreement_fraction: float,
) -> tuple[bool, bool]:
    """Return (coco_mask_agreement, coco_mask_disagreement) from fraction."""
    if agreement_fraction == 0.0:
        # No valid pixels to compare
        return False, False
    agrees = agreement_fraction >= MASK_DISAGREEMENT_THRESHOLD
    disagrees = not agrees
    return agrees, disagrees


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------


JPEG_QUALITY = 95


def build_contact_sheet_clean(
    image_dir: Path,
    manifest_rows: list[dict[str, str]],
    output_path: Path,
    per_class: int = 4,
) -> Path:
    """Build a clean contact sheet: class-stratified sample of unflagged JPEGs.

    Deterministic: sorted by sample_id within each class. Labels show
    sample ID, RoofSense class, split, and target occupancy.
    """
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        if row.get("status") != "prepared" or _is_flagged(row):
            continue
        cls = row.get("roofsense_class", "unknown")
        by_class.setdefault(cls, []).append(row)

    selected = []
    for cls in sorted(by_class):
        rows = sorted(by_class[cls], key=lambda r: r["sample_id"])
        selected.extend(rows[:per_class])
    selected.sort(key=lambda r: r["sample_id"])

    return _render_contact_sheet(image_dir, selected, output_path)


def build_contact_sheet_diagnostics(
    image_dir: Path,
    source_dir: Path,
    coco_data,
    manifest_rows: list[dict[str, str]],
    output_path: Path,
    per_class: int = 4,
) -> Path:
    """Build a diagnostics contact sheet emphasizing flagged and boundary cases.

    Each cell shows: the unmarked JPEG, annotation ID, native class,
    mapped class, target fraction, competing fraction, and flags.
    """
    flagged = [r for r in manifest_rows if r.get("status") == "prepared" and _is_flagged(r)]

    by_class: dict[str, list[dict[str, str]]] = {}
    for row in flagged:
        cls = row.get("roofsense_class", "unknown")
        by_class.setdefault(cls, []).append(row)

    selected = []
    for cls in sorted(by_class):
        rows = sorted(by_class[cls], key=lambda r: r["sample_id"])
        selected.extend(rows[:per_class])
    selected.sort(key=lambda r: r["sample_id"])

    return _render_diagnostic_comparison(
        image_dir, source_dir, coco_data, selected, output_path
    )


def _render_diagnostic_comparison(
    image_dir: Path,
    source_dir: Path,
    coco_data,
    rows: list[dict[str, str]],
    output_path: Path,
    thumb_size: int = 224,
) -> Path:
    """Render tile/seed, selected footprint, crop bounds, and model input."""
    if not rows:
        image = Image.new("RGB", (400, 40), "white")
        image.save(output_path, "JPEG", quality=JPEG_QUALITY)
        return output_path

    from coco_crop_utils import read_rgb_bands

    sheet = Image.new("RGB", (4 * thumb_size, len(rows) * thumb_size), (200, 200, 200))
    font = ImageFont.load_default()
    panel_names = ("seed annotation", "selected footprint", "final crop bounds", "model input")

    for row_index, row in enumerate(rows):
        source_path = source_dir / row["source_relpath"]
        tile = Image.fromarray(read_rgb_bands(source_path), "RGB")
        annotation = coco_data.annotations[int(row["coco_annotation_id"])]
        points = list(zip(annotation["segmentation"][0][0::2], annotation["segmentation"][0][1::2]))

        seed_panel = tile.copy()
        ImageDraw.Draw(seed_panel).line(points + [points[0]], fill="red", width=5)

        footprint_panel = tile.copy()
        fp_draw = ImageDraw.Draw(footprint_panel, "RGBA")
        footprint = tuple(int(row[key]) for key in (
            "footprint_left", "footprint_top", "footprint_right", "footprint_bottom"
        ))
        fp_draw.rectangle(footprint, outline=(0, 255, 0, 255), width=5)
        fp_draw.rectangle(footprint, fill=(0, 255, 0, 45))

        crop_panel = tile.copy()
        crop_box = tuple(int(row[key]) for key in (
            "crop_left", "crop_top", "crop_right", "crop_bottom"
        ))
        ImageDraw.Draw(crop_panel).rectangle(crop_box, outline="cyan", width=5)

        model_path = image_dir / row["prepared_filename"]
        model_panel = Image.open(model_path).convert("RGB")
        panels = (seed_panel, footprint_panel, crop_panel, model_panel)

        for panel_index, (panel, name) in enumerate(zip(panels, panel_names)):
            thumb = panel.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            x = panel_index * thumb_size
            y = row_index * thumb_size
            sheet.paste(thumb, (x, y))
            draw = ImageDraw.Draw(sheet)
            label = name if panel_index else (
                f"{name}\n{row['sample_id'][:24]}\n{row['roofsense_class']}"
            )
            box = draw.textbbox((x + 4, y + 4), label, font=font)
            draw.rectangle(box, fill="black")
            draw.text((x + 4, y + 4), label, fill="yellow", font=font)

    sheet.save(output_path, "JPEG", quality=JPEG_QUALITY)
    return output_path


def _is_flagged(row: dict[str, str]) -> bool:
    """Check if a manifest row has any diagnostic flags set."""
    flags = [
        "multi_label_overlap",
        "other_class_in_context",
        "same_class_neighbor",
        "low_target_occupancy",
        "coco_mask_disagreement",
        "contains_nodata",
        "multi_material_component",
        "potential_building_merge",
    ]
    for f in flags:
        val = row.get(f, False)
        if isinstance(val, bool):
            if val:
                return True
        elif str(val).lower() in ("true", "1"):
            return True
    return False


def _render_contact_sheet(
    image_dir: Path,
    rows: list[dict[str, str]],
    output_path: Path,
    show_flags: bool = False,
    thumb_size: int = 224,
) -> Path:
    """Render a contact sheet grid with labeled thumbnails."""
    if not rows:
        img = Image.new("RGB", (400, 40), (255, 255, 255))
        img.save(str(output_path), "JPEG", quality=JPEG_QUALITY)
        return output_path

    n = len(rows)
    cols = min(8, int(np.ceil(np.sqrt(n))))
    rows_count = int(np.ceil(n / cols))

    sheet = Image.new("RGB", (cols * thumb_size, rows_count * thumb_size), (200, 200, 200))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, row in enumerate(rows):
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
        label_lines = [
            row.get("sample_id", "")[:24],
            row.get("roofsense_class", ""),
            row.get("split", ""),
            f"tgt={row.get('target_crop_fraction', '')}",
        ]
        if show_flags:
            active_flags = []
            for f in ("multi_label_overlap", "low_target_occupancy", "coco_mask_disagreement",
                       "contains_nodata", "multi_material_component", "potential_building_merge"):
                val = row.get(f, False)
                is_flagged = val if isinstance(val, bool) else str(val).lower() in ("true", "1")
                if is_flagged:
                    active_flags.append(f[:16])
            if active_flags:
                label_lines.append("flags:" + ",".join(active_flags))

        label = "\n".join(label_lines)
        # Background box for readability
        bbox = draw.textbbox((x0 + 4, y0 + 4), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 0), font=font)

    sheet.save(str(output_path), "JPEG", quality=JPEG_QUALITY)
    return output_path
