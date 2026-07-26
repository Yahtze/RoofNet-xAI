"""
coco_crop_utils.py
==================
Reusable utilities for COCO-crop preparation pipelines.

Provides:
- Collision-safe sample ID generation from COCO identity tuples
- Square expansion in source coordinates with constant boundary padding
- JPEG writing with content-match guard
- Atomic CSV manifest writing
- File and metadata fingerprinting
"""

from __future__ import annotations

import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def make_sample_id(
    source_relpath: str,
    coco_image_id: int,
    coco_annotation_id: int,
    roofsense_class_id: int,
) -> str:
    """Generate a collision-safe sample ID from COCO identity components.

    Format: ``{source_stem}__ann{ann_id}--{sha256_prefix}``

    The SHA-256 prefix covers the full identity tuple so distinct annotations
    on the same tile always get distinct IDs, even if annotation IDs collide
    across images.
    """
    identity_key = (
        f"{source_relpath}|img{coco_image_id}|ann{coco_annotation_id}|cls{roofsense_class_id}"
    )
    digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:16]
    source_stem = Path(source_relpath).stem
    return f"{source_stem}__ann{coco_annotation_id}--{digest}"


def assign_probable_building_groups(
    rows: list[dict],
    iou_threshold: float = 0.8,
) -> None:
    """Group highly overlapping prepared crops from the same source tile."""
    prepared = [row for row in rows if row.get("status") == "prepared"]
    by_source: dict[str, list[dict]] = {}
    for row in prepared:
        by_source.setdefault(str(row["source_relpath"]), []).append(row)

    def crop_iou(first: dict, second: dict) -> float:
        left = max(int(first["crop_left"]), int(second["crop_left"]))
        top = max(int(first["crop_top"]), int(second["crop_top"]))
        right = min(int(first["crop_right"]), int(second["crop_right"]))
        bottom = min(int(first["crop_bottom"]), int(second["crop_bottom"]))
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = (
            (int(first["crop_right"]) - int(first["crop_left"]))
            * (int(first["crop_bottom"]) - int(first["crop_top"]))
        )
        second_area = (
            (int(second["crop_right"]) - int(second["crop_left"]))
            * (int(second["crop_bottom"]) - int(second["crop_top"]))
        )
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    for source_relpath, source_rows in by_source.items():
        parents = list(range(len(source_rows)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        for first in range(len(source_rows)):
            for second in range(first + 1, len(source_rows)):
                if crop_iou(source_rows[first], source_rows[second]) >= iou_threshold:
                    union(first, second)

        groups: dict[int, list[dict]] = {}
        for index, row in enumerate(source_rows):
            groups.setdefault(find(index), []).append(row)

        for group_rows in groups.values():
            annotation_ids = sorted(int(row["coco_annotation_id"]) for row in group_rows)
            identity = f"{source_relpath}|{','.join(map(str, annotation_ids))}"
            group_id = "pbg--" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            labels = {str(row["roofsense_class"]) for row in group_rows}
            representative = max(
                group_rows,
                key=lambda row: (
                    int(row.get("target_pixels", 0)),
                    -int(row["coco_annotation_id"]),
                ),
            )
            for row in group_rows:
                row["probable_building_group_id"] = group_id
                row["probable_building_group_size"] = len(group_rows)
                row["probable_building_group_label_count"] = len(labels)
                row["probable_building_group_ambiguous"] = len(labels) > 1
                row["probable_building_representative"] = row is representative


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def polygon_bbox(segmentation: list[float]) -> tuple[int, int, int, int]:
    """Compute the smallest integer pixel rectangle containing a COCO polygon.

    The polygon is a flat list [x1, y1, x2, y2, ...]. Returns (left, top,
    right, bottom) with inclusive-exclusive semantics (right/bottom are one
    past the last pixel).

    Raises ValueError for empty or degenerate polygons.
    """
    if len(segmentation) < 6:
        raise ValueError(f"Polygon has fewer than 3 points ({len(segmentation) // 2})")
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    left = int(np.floor(min(xs)))
    top = int(np.floor(min(ys)))
    right = int(np.ceil(max(xs))) + 1
    bottom = int(np.ceil(max(ys))) + 1
    if right <= left or bottom <= top:
        raise ValueError(f"Degenerate polygon bounds: ({left},{top})-({right},{bottom})")
    return left, top, right, bottom


def clip_expand_rect(
    left: int,
    top: int,
    right: int,
    bottom: int,
    padding: int,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """Expand a rectangle by *padding* on every side, clipping to image bounds."""
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(img_width, right + padding),
        min(img_height, bottom + padding),
    )


def expand_rect_to_square(
    left: int,
    top: int,
    right: int,
    bottom: int,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """Expand a rectangle into a centered square in source coordinates.

    The shorter dimension is expanded symmetrically (extra pixel on the
    right/bottom if odd difference). If expansion clips at a tile boundary,
    the other side extends further to preserve the square shape. If the
    image is too small for the full square, both sides are clipped and the
    result is non-square.

    Returns (left, top, right, bottom) in inclusive-exclusive semantics.
    """
    w = right - left
    h = bottom - top
    if w == h:
        return left, top, right, bottom

    target = max(w, h)

    if w < h:
        diff = target - w
        left -= diff // 2
        right += diff - diff // 2
    else:
        diff = target - h
        top -= diff // 2
        bottom += diff - diff // 2

    # Clip: if one edge goes past the boundary, shift the other to preserve
    # the square shape. If the image is too small, both clip.
    if left < 0:
        shift = -left
        left = 0
        right = min(img_width, right + shift)
    if right > img_width:
        right = img_width
        left = max(0, right - target)
    if top < 0:
        shift = -top
        top = 0
        bottom = min(img_height, bottom + shift)
    if bottom > img_height:
        bottom = img_height
        top = max(0, bottom - target)

    return left, top, right, bottom


def crop_with_constant_boundary_padding(
    rgb: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    target_size: int,
) -> np.ndarray:
    """Crop *rgb* at the given rectangle, pad with zeros at tile boundaries.

    The rectangle is in inclusive-exclusive pixel coordinates. If the rectangle
    extends beyond the image, those regions are filled with zero (black).
    The result is always exactly (target_size, target_size, 3).

    Used when the square expansion from ``expand_rect_to_square`` clips at
    a tile edge, producing a crop smaller than *target_size*.
    """
    img_h, img_w = rgb.shape[:2]

    # Where the crop rectangle sits relative to the image
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(img_w, right)
    src_bottom = min(img_h, bottom)

    # Offsets into the output array
    pad_left = src_left - left
    pad_top = src_top - top

    crop_h = src_bottom - src_top
    crop_w = src_right - src_left

    out = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    if crop_h > 0 and crop_w > 0:
        out[pad_top:pad_top + crop_h, pad_left:pad_left + crop_w] = \
            rgb[src_top:src_bottom, src_left:src_right]

    return out


# ---------------------------------------------------------------------------
# Image operations
# ---------------------------------------------------------------------------


def read_rgb_bands(tiff_path: Path) -> np.ndarray:
    """Read bands 1-3 of a multi-band TIFF as an (H, W, 3) uint8 RGB array.

    Non-finite pixels are replaced with zero. Values are clipped to [0, 255]
    and cast to uint8.
    """
    import tifffile

    data = tifffile.imread(str(tiff_path))
    # Shape: (bands, H, W) or (H, W, bands)
    if data.ndim == 3 and data.shape[0] in (3, 4, 7):
        rgb = data[:3].transpose(1, 2, 0).astype(np.float64)
    elif data.ndim == 3 and data.shape[2] in (3, 4, 7):
        rgb = data[:, :, :3].astype(np.float64)
    else:
        raise ValueError(f"Unexpected TIFF shape {data.shape} for {tiff_path}")

    # Replace non-finite with zero, clip to [0, 255], cast to uint8
    rgb[~np.isfinite(rgb)] = 0.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def square_pad_rgb(image: Image.Image) -> Image.Image:
    """Pad an RGB image to a square using numpy edge-mode replication.

    The shorter dimension is expanded so width == height. Padding is applied
    symmetrically (extra pixel on the right/bottom if odd difference).
    Matches the Kakuma preparation behavior exactly.
    """
    w, h = image.size
    if w == h:
        return image.copy()

    arr = np.asarray(image)
    diff = abs(w - h)
    pad_before = diff // 2
    pad_after = diff - pad_before

    if w < h:
        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
    else:
        pad_width = ((pad_before, pad_after), (0, 0), (0, 0))

    padded = np.pad(arr, pad_width, mode="edge")
    return Image.fromarray(padded, "RGB")


def write_prepared_jpeg(
    image: Image.Image,
    output_dir: Path,
    sample_id: str,
    quality: int = 95,
) -> Path:
    """Write an RGB JPEG to *output_dir*/<sample_id>.jpg.

    Refuses to overwrite an existing file whose image bytes do not match
    the requested output. Returns the Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sample_id}.jpg"
    output_path = output_dir / filename

    encoded = BytesIO()
    image.convert("RGB").save(encoded, "JPEG", quality=quality)
    requested_bytes = encoded.getvalue()

    if output_path.exists():
        if output_path.read_bytes() == requested_bytes:
            return output_path
        raise FileExistsError(
            f"Refusing to overwrite {output_path}: existing image bytes differ"
        )

    output_path.write_bytes(requested_bytes)
    return output_path


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


# Full manifest columns per the RoofSense spec
MANIFEST_COLUMNS = [
    "sample_id",
    "source_relpath",
    "source_stem",
    "coco_image_id",
    "coco_annotation_id",
    "roofsense_class_id",
    "roofsense_class",
    "semantic_mask_class_id",
    "mapped_remoteclip_class",
    "mapping_version",
    "split",
    "prepared_filename",
    "crop_method",
    "polygon_bbox_x",
    "polygon_bbox_y",
    "polygon_bbox_width",
    "polygon_bbox_height",
    "footprint_left",
    "footprint_top",
    "footprint_right",
    "footprint_bottom",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "padding_px",
    "square_size",
    "component_pixel_count",
    "target_pixels",
    "target_crop_fraction",
    "annotated_roof_crop_fraction",
    "same_class_other_pixels",
    "same_class_other_fraction",
    "different_class_pixels",
    "different_class_fraction",
    "same_class_annotation_count",
    "different_class_annotation_count",
    "shared_annotation_count",
    "multi_label_overlap",
    "other_class_in_context",
    "same_class_neighbor",
    "low_target_occupancy",
    "multi_material_component",
    "potential_building_merge",
    "probable_building_group_id",
    "probable_building_group_size",
    "probable_building_group_label_count",
    "probable_building_group_ambiguous",
    "probable_building_representative",
    "coco_mask_agreement",
    "coco_mask_disagreement",
    "nodata_fraction",
    "contains_nodata",
    "status",
    "error",
]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a CSV manifest atomically with all required columns.

    Uses a temporary file followed by an atomic rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [",".join(MANIFEST_COLUMNS)]
    for row in rows:
        values = []
        for col in MANIFEST_COLUMNS:
            val = str(row.get(col, ""))
            if "," in val or '"' in val or "\n" in val:
                val = '"' + val.replace('"', '""') + '"'
            values.append(val)
        lines.append(",".join(values))

    content = "\n".join(lines) + "\n"

    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".csv"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_json_metadata(path: Path, metadata: dict) -> None:
    """Write JSON metadata atomically via temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".json"
    ) as tmp:
        json.dump(metadata, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def fingerprint_file(path: Path) -> str:
    """SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_strings(items: list[str]) -> str:
    """Deterministic SHA-256 fingerprint of a list of strings."""
    h = hashlib.sha256()
    for item in sorted(items):
        h.update(item.encode("utf-8"))
    return h.hexdigest()
