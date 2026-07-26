"""
coco_parser.py
==============
COCO annotation parsing, polygon rasterization, split resolution, and
semantic-mask comparison for the RoofSense dataset.

RoofSense-specific logic that does not belong in the shared Kakuma utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# COCO category mapping
# ---------------------------------------------------------------------------

# COCO ID -> (mask_id, roofsense_label, directory_name)
# Category 0 is a container, category 4 is Invalid.
COCO_CATEGORY_MAP: dict[int, tuple[int | None, str, str | None]] = {
    0: (None, "Container", None),
    1: (1, "Ceramic Tile", "ceramic_tile"),
    2: (2, "Dark-coloured Membrane", "dark_membrane"),
    3: (3, "Gravel", "gravel"),
    4: (None, "Invalid", None),  # excluded
    5: (4, "Light-coloured Membrane", "light_membrane"),
    6: (5, "Light-permitting Surface", "light_permitting"),
    7: (6, "Metal", "metal"),
    8: (7, "Solar Panel", "solar_panel"),
    9: (8, "Vegetation", "vegetation"),
}

SUPPORTED_COCO_IDS = {k for k, v in COCO_CATEGORY_MAP.items() if v[2] is not None}
INVALID_COCO_ID = 4

# Mask ID -> RoofSense label (for semantic mask comparison)
MASK_ID_TO_LABEL: dict[int, str] = {
    0: "Background",
    1: "Ceramic Tile",
    2: "Dark-coloured Membrane",
    3: "Gravel",
    4: "Light-coloured Membrane",
    5: "Light-permitting Surface",
    6: "Metal",
    7: "Solar Panel",
    8: "Vegetation",
}

# RemoteCLIP mapping: RoofSense label -> RemoteCLIP target
REMOTECLIP_MAPPING: dict[str, str] = {
    "Ceramic Tile": "ClayTiles",
    "Dark-coloured Membrane": "AmorphousAsphalt",
    "Gravel": "Unknown",
    "Light-coloured Membrane": "AmorphousMembrane",
    "Light-permitting Surface": "GlassSheetMaterials",
    "Metal": "MetalSheetMaterials",
    "Solar Panel": "Unknown",
    "Vegetation": "GreenVegetative",
}

MAPPING_VERSION = "1.0"


# ---------------------------------------------------------------------------
# COCO loading
# ---------------------------------------------------------------------------


class CocoData:
    """Parsed COCO annotation data with indexed lookups."""

    def __init__(self, annotations_path: Path):
        raw = json.loads(Path(annotations_path).read_text())

        self.info = raw.get("info", {})
        self.licenses = raw.get("licenses", [])

        # Categories indexed by ID
        self.categories: dict[int, dict] = {c["id"]: c for c in raw["categories"]}

        # Images indexed by ID
        self.images: dict[int, dict] = {img["id"]: img for img in raw["images"]}

        # Annotations indexed by ID
        self.annotations: dict[int, dict] = {ann["id"]: ann for ann in raw["annotations"]}

        # Build image_id -> annotations index
        self.image_annotations: dict[int, list[dict]] = {}
        for ann in raw["annotations"]:
            img_id = ann["image_id"]
            self.image_annotations.setdefault(img_id, []).append(ann)

    def validate(self) -> list[str]:
        """Validate COCO data integrity. Returns list of errors (empty = OK)."""
        errors = []

        # Check category definitions match expectations
        expected_cats = {
            0: "roofing-materials-zsJ0",
            1: "Ceramic Tile",
            2: "Dark-coloured Membrane",
            3: "Gravel",
            4: "Invalid",
            5: "Light-coloured Membrane",
            6: "Light-permitting Surface",
            7: "Metal",
            8: "Solar Panel",
            9: "Vegetation",
        }
        for cat_id, expected_name in expected_cats.items():
            if cat_id not in self.categories:
                errors.append(f"Missing expected category ID {cat_id}")
            elif self.categories[cat_id]["name"] != expected_name:
                errors.append(
                    f"Category {cat_id}: expected '{expected_name}', "
                    f"got '{self.categories[cat_id]['name']}'"
                )

        return errors

    def resolve_tiff_path(self, image_record: dict, images_dir: Path) -> Path:
        """Resolve a COCO image record to its source .tif file path.

        COCO filenames use .png extension; actual files are .tif.
        """
        coco_fname = image_record["file_name"]
        tif_fname = coco_fname.replace(".png", ".tif")
        return images_dir / tif_fname

    def resolve_split(
        self, image_record: dict, splits: dict[str, list[str]]
    ) -> str | None:
        """Resolve the official split for a COCO image record.

        Returns 'training', 'validation', 'test', or None if not found.
        """
        tif_fname = image_record["file_name"].replace(".png", ".tif")
        for split_name, files in splits.items():
            if tif_fname in files:
                return split_name
        return None


def load_splits(splits_path: Path) -> dict[str, list[str]]:
    """Load splits.json."""
    return json.loads(Path(splits_path).read_text())


def load_names(names_path: Path) -> dict[str, str]:
    """Load names.json (mask ID string -> class name)."""
    raw = json.loads(Path(names_path).read_text())
    return {k: v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Polygon rasterization
# ---------------------------------------------------------------------------


def rasterize_polygon_mask(
    segmentation: list[float],
    width: int,
    height: int,
    left: int = 0,
    top: int = 0,
) -> np.ndarray:
    """Rasterize a COCO polygon into a boolean mask.

    The segmentation is a flat list [x1, y1, x2, y2, ...]. Uses PIL
    ImageDraw.polygon for rasterization with consistent pixel-inclusion
    convention.

    *left* and *top* offset the polygon coordinates for crop-relative
    rasterization.

    Returns an (H, W) boolean array where True = inside polygon.
    """
    from PIL import ImageDraw

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Convert flat list to (x, y) pairs, offset by (left, top)
    points = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] - left
        y = segmentation[i + 1] - top
        points.append((x, y))

    draw.polygon(points, fill=1)
    return np.array(mask, dtype=bool)


def compute_union_mask(
    segmentations: list[list[float]],
    width: int,
    height: int,
    left: int = 0,
    top: int = 0,
) -> np.ndarray:
    """Compute a union mask of multiple polygons (no double-counting)."""
    union = np.zeros((height, width), dtype=bool)
    for seg in segmentations:
        union |= rasterize_polygon_mask(seg, width, height, left, top)
    return union


# ---------------------------------------------------------------------------
# Semantic mask comparison
# ---------------------------------------------------------------------------


def read_semantic_mask(mask_path: Path) -> np.ndarray:
    """Read a class-index TIFF mask as an integer array."""
    import tifffile

    data = tifffile.imread(str(mask_path))
    return data.astype(np.int32)


def compute_mask_agreement(
    semantic_mask: np.ndarray,
    polygon_mask: np.ndarray,
    coco_category_id: int,
    crop_left: int = 0,
    crop_top: int = 0,
    crop_width: int | None = None,
    crop_height: int | None = None,
) -> tuple[int, int, float]:
    """Compute agreement between COCO category and semantic mask within polygon.

    *semantic_mask* is the full-tile semantic mask. *polygon_mask* is a
    crop-sized boolean mask. When crop offsets are given, the semantic mask
    is sliced to the corresponding crop region before comparison.

    Returns (total_valid_pixels, agreeing_pixels, agreement_fraction).
    """
    # Get the mask ID for this COCO category
    cat_info = COCO_CATEGORY_MAP.get(coco_category_id)
    if cat_info is None or cat_info[0] is None:
        return 0, 0, 0.0

    mask_id = cat_info[0]

    # Crop the semantic mask to match the polygon mask dimensions
    if crop_width is not None and crop_height is not None:
        # Clamp slice bounds to [0, size] to avoid numpy wraparound on negatives
        sl = max(0, crop_left)
        st = max(0, crop_top)
        cropped_sem = semantic_mask[st:crop_top + crop_height, sl:crop_left + crop_width]
    else:
        cropped_sem = semantic_mask

    # Pixels inside the polygon
    inside = cropped_sem[polygon_mask]
    valid = inside > 0  # exclude background
    total_valid = int(valid.sum())
    if total_valid == 0:
        return 0, 0, 0.0
    agreeing = int((inside[valid] == mask_id).sum())
    fraction = agreeing / total_valid
    return total_valid, agreeing, fraction
