"""
building_footprint.py
=====================
Estimate building footprints from annotation seeds using connected-component
analysis on nonblack RGB pixels.

The COCO polygon becomes a *seed* identifying the target material. The
building footprint is the connected nonblack RGB region containing that seed.
The crop is the bounding box of the full footprint, expanded to a square
with padding, and padded with constant black at tile boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from coco_parser import COCO_CATEGORY_MAP


@dataclass
class FootprintResult:
    """Result of footprint extraction for one annotation seed."""

    # Bounding box of the nonblack component (inclusive-exclusive, tile coords)
    footprint_left: int
    footprint_top: int
    footprint_right: int
    footprint_bottom: int

    # Component pixel count (nonblack pixels in the component)
    component_pixel_count: int

    # Number of distinct nonblack connected components on the entire tile
    total_components: int

    # Annotation IDs that share this component (including the seed)
    shared_annotation_ids: list[int]

    # Material class IDs present in the component (excluding the seed's class)
    other_material_class_ids: list[int]

    # Flags
    multi_material_component: bool
    potential_building_merge: bool


def find_footprint_and_crop_rect(
    rgb: np.ndarray,
    segmentation: list[float],
    seed_annotation_id: int,
    seed_category_id: int,
    image_annotations: list[dict],
    padding_px: int,
    tile_size: int,
) -> FootprintResult:
    """Estimate building footprint from the nonblack connected component
    containing the seed annotation.

    Parameters
    ----------
    rgb : np.ndarray
        Full-tile RGB image (tile_size, tile_size, 3), uint8.
    segmentation : list[float]
        Flat COCO polygon coordinates for the seed annotation.
    seed_annotation_id : int
        The annotation ID being processed.
    seed_category_id : int
        COCO category ID of the seed annotation.
    image_annotations : list[dict]
        All annotations on this tile (including the seed).
    padding_px : int
        Context padding in pixels.
    tile_size : int
        Tile dimension (512).

    Returns
    -------
    FootprintResult with footprint bounds, component info, and flags.
    """
    # 1. Binary mask of nonblack pixels
    nonblack = np.any(rgb > 0, axis=2)

    # 2. Connected components
    labeled, num_components = ndimage.label(nonblack)

    # 3. Find which component the seed polygon touches
    from coco_crop_utils import polygon_bbox
    from coco_parser import rasterize_polygon_mask

    try:
        poly_left, poly_top, poly_right, poly_bottom = polygon_bbox(segmentation)
    except ValueError:
        # Degenerate polygon: fall back to single-pixel seed
        poly_left = poly_top = 0
        poly_right = poly_bottom = 1

    # Clamp to tile bounds
    poly_left = max(0, poly_left)
    poly_top = max(0, poly_top)
    poly_right = min(tile_size, poly_right)
    poly_bottom = min(tile_size, poly_bottom)

    # Rasterize seed polygon on the full tile
    seed_mask_full = rasterize_polygon_mask(
        segmentation, tile_size, tile_size
    )

    # Find component IDs that the seed polygon overlaps
    seed_pixels = labeled[seed_mask_full]
    seed_components = set(seed_pixels[seed_pixels > 0])

    if not seed_components:
        # Seed polygon is entirely on black pixels — fall back to polygon bbox
        return FootprintResult(
            footprint_left=poly_left,
            footprint_top=poly_top,
            footprint_right=poly_right,
            footprint_bottom=poly_bottom,
            component_pixel_count=0,
            total_components=num_components,
            shared_annotation_ids=[seed_annotation_id],
            other_material_class_ids=[],
            multi_material_component=False,
            potential_building_merge=False,
        )

    # Use the largest overlapping component (by overlap pixel count)
    component_id = max(seed_components, key=lambda c: int((seed_pixels == c).sum()))

    # 4. Bounding box of the selected component
    component_mask = labeled == component_id
    ys, xs = np.where(component_mask)
    footprint_left = int(xs.min())
    footprint_top = int(ys.min())
    footprint_right = int(xs.max()) + 1
    footprint_bottom = int(ys.max()) + 1
    component_pixel_count = len(xs)

    # 5. Find all annotations whose polygons touch this component
    shared_ids: list[int] = []
    other_class_ids: list[int] = []

    for ann in image_annotations:
        ann_id = ann["id"]
        ann_cat = ann["category_id"]
        ann_seg = ann.get("segmentation", [])
        if not ann_seg or not isinstance(ann_seg[0], list):
            continue

        ann_mask = rasterize_polygon_mask(ann_seg[0], tile_size, tile_size)
        overlap = component_mask & ann_mask
        if overlap.any():
            shared_ids.append(ann_id)
            if ann_id != seed_annotation_id and ann_cat != seed_category_id:
                other_class_ids.append(ann_cat)

    # 6. Flags
    multi_material = len(set(other_class_ids)) > 0
    # Heuristic: component is large relative to tile, or has many annotations
    # from different spatial clusters
    potential_merge = component_pixel_count > (tile_size * tile_size * 0.25) and len(shared_ids) > 3

    return FootprintResult(
        footprint_left=footprint_left,
        footprint_top=footprint_top,
        footprint_right=footprint_right,
        footprint_bottom=footprint_bottom,
        component_pixel_count=component_pixel_count,
        total_components=num_components,
        shared_annotation_ids=shared_ids,
        other_material_class_ids=list(set(other_class_ids)),
        multi_material_component=multi_material,
        potential_building_merge=potential_merge,
    )


def compute_component_overlay(
    rgb: np.ndarray,
    component_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> np.ndarray:
    """Create a diagnostic overlay showing the component and seed on the tile.

    Returns an RGB uint8 array with:
    - Component region: green tint (50% blend)
    - Seed polygon: red outline
    """
    overlay = rgb.copy().astype(np.float32)

    # Green tint on component
    green = overlay.copy()
    green[component_mask, 0] *= 0.5
    green[component_mask, 1] = np.clip(
        green[component_mask, 1] * 0.5 + 128, 0, 255
    )
    green[component_mask, 2] *= 0.5
    overlay[component_mask] = green[component_mask] * 0.5 + overlay[component_mask] * 0.5

    # Red outline on seed polygon boundary
    from PIL import Image, ImageFilter

    seed_img = Image.fromarray((seed_mask * 255).astype(np.uint8))
    edges = np.array(seed_img.filter(ImageFilter.FIND_EDGES)) > 0
    overlay[edges, 0] = 255
    overlay[edges, 1] = 0
    overlay[edges, 2] = 0

    return np.clip(overlay, 0, 255).astype(np.uint8)
