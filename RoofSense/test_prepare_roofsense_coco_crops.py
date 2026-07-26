"""
test_prepare_roofsense_coco_crops.py
=====================================
Comprehensive tests for the RoofSense COCO-crop preparation pipeline.

Uses synthetic data (no real RoofSense tiles needed).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from coco_crop_utils import (
    MANIFEST_COLUMNS,
    clip_expand_rect,
    crop_with_constant_boundary_padding,
    assign_probable_building_groups,
    expand_rect_to_square,
    fingerprint_file,
    fingerprint_strings,
    make_sample_id,
    polygon_bbox,
    square_pad_rgb,
    write_json_metadata,
    write_manifest,
    write_prepared_jpeg,
)
from coco_parser import (
    COCO_CATEGORY_MAP,
    INVALID_COCO_ID,
    MAPPING_VERSION,
    REMOTECLIP_MAPPING,
    SUPPORTED_COCO_IDS,
    CocoData,
    compute_mask_agreement,
    compute_union_mask,
    rasterize_polygon_mask,
)
from building_footprint import find_footprint_and_crop_rect, FootprintResult
from diagnostics import (
    LOW_TARGET_OCCUPANCY_THRESHOLD,
    MASK_DISAGREEMENT_THRESHOLD,
    compute_mask_agreement_flag,
    measure_crop_composition,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def synthetic_coco_data():
    """Create synthetic COCO annotation data."""
    categories = [
        {"id": 0, "name": "roofing-materials-zsJ0", "supercategory": "none"},
        {"id": 1, "name": "Ceramic Tile", "supercategory": "Roofing Materials"},
        {"id": 2, "name": "Dark-coloured Membrane", "supercategory": "Roofing Materials"},
        {"id": 3, "name": "Gravel", "supercategory": "Roofing Materials"},
        {"id": 4, "name": "Invalid", "supercategory": "Roofing Materials"},
        {"id": 5, "name": "Light-coloured Membrane", "supercategory": "Roofing Materials"},
        {"id": 6, "name": "Light-permitting Surface", "supercategory": "Roofing Materials"},
        {"id": 7, "name": "Metal", "supercategory": "Roofing Materials"},
        {"id": 8, "name": "Solar Panel", "supercategory": "Roofing Materials"},
        {"id": 9, "name": "Vegetation", "supercategory": "Roofing Materials"},
    ]
    images = [
        {"id": 0, "file_name": "9-368-464_5_18.png", "height": 512, "width": 512},
    ]
    annotations = [
        # Valid Ceramic Tile (isolated)
        {
            "id": 0, "image_id": 0, "category_id": 1,
            "bbox": [100, 100, 50, 50], "area": 2500,
            "segmentation": [[100, 100, 150, 100, 150, 150, 100, 150]],
            "iscrowd": 0,
        },
        # Valid Dark Membrane (isolated)
        {
            "id": 1, "image_id": 0, "category_id": 2,
            "bbox": [200, 200, 50, 50], "area": 2500,
            "segmentation": [[200, 200, 250, 200, 250, 250, 200, 250]],
            "iscrowd": 0,
        },
        # Neighboring same class
        {
            "id": 2, "image_id": 0, "category_id": 1,
            "bbox": [140, 100, 50, 50], "area": 2500,
            "segmentation": [[140, 100, 190, 100, 190, 150, 140, 150]],
            "iscrowd": 0,
        },
        # Different-class overlap
        {
            "id": 3, "image_id": 0, "category_id": 2,
            "bbox": [120, 120, 50, 50], "area": 2500,
            "segmentation": [[120, 120, 170, 120, 170, 170, 120, 170]],
            "iscrowd": 0,
        },
        # Invalid annotation
        {
            "id": 4, "image_id": 0, "category_id": 4,
            "bbox": [300, 300, 50, 50], "area": 2500,
            "segmentation": [[300, 300, 350, 300, 350, 350, 300, 350]],
            "iscrowd": 0,
        },
        # Low occupancy
        {
            "id": 5, "image_id": 0, "category_id": 3,
            "bbox": [400, 400, 100, 100], "area": 100,
            "segmentation": [[440, 440, 450, 440, 450, 450, 440, 450]],
            "iscrowd": 0,
        },
        # Solar Panel (should be excluded)
        {
            "id": 6, "image_id": 0, "category_id": 8,
            "bbox": [50, 50, 30, 30], "area": 900,
            "segmentation": [[50, 50, 80, 50, 80, 80, 50, 80]],
            "iscrowd": 0,
        },
    ]
    return {"categories": categories, "images": images, "annotations": annotations}


@pytest.fixture
def full_synthetic_dataset(tmp_path, synthetic_coco_data):
    """Create a complete synthetic dataset with all required files."""
    import tifffile

    ds = tmp_path / "dataset"
    ds.mkdir()
    (ds / "images").mkdir()
    (ds / "masks").mkdir()
    (ds / "annotations").mkdir()

    tile_name = "9-368-464_5_18.tif"

    # Create image: nonblack regions at annotation locations
    rng = np.random.RandomState(42)
    bands = np.zeros((7, 512, 512), dtype=np.uint8)
    # Fill nonblack regions where annotations exist
    for y in range(80, 260):
        for x in range(80, 260):
            bands[0, y, x] = rng.randint(10, 255)
            bands[1, y, x] = rng.randint(10, 255)
            bands[2, y, x] = rng.randint(10, 255)
    # Small region for low-occupancy annotation
    for y in range(430, 460):
        for x in range(430, 460):
            bands[0, y, x] = rng.randint(10, 255)
            bands[1, y, x] = rng.randint(10, 255)
            bands[2, y, x] = rng.randint(10, 255)
    # Solar panel region
    for y in range(40, 90):
        for x in range(40, 90):
            bands[0, y, x] = rng.randint(10, 255)
            bands[1, y, x] = rng.randint(10, 255)
            bands[2, y, x] = rng.randint(10, 255)
    tifffile.imwrite(str(ds / "images" / tile_name), bands)

    # Create mask
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:150, 100:150] = 1
    mask[200:250, 200:250] = 2
    mask[120:170, 120:170] = 2
    tifffile.imwrite(str(ds / "masks" / tile_name), mask)

    (ds / "annotations" / "annotations.json").write_text(json.dumps(synthetic_coco_data))
    (ds / "splits.json").write_text(json.dumps({"training": [tile_name]}))
    (ds / "names.json").write_text(json.dumps({
        "0": "Background", "1": "Ceramic Tile", "2": "Dark-coloured Membrane",
        "3": "Gravel", "4": "Light-coloured Membrane", "5": "Light-permitting Surface",
        "6": "Metal", "7": "Solar Panel", "8": "Vegetation",
    }))
    return ds


# ===========================================================================
# Test: .tif discovery excluding .tif.aux.xml
# ===========================================================================


class TestTiffDiscovery:
    def test_discovers_tif_only(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for name in ["a.tif", "b.tif", "c.tif"]:
            (img_dir / name).touch()
        for name in ["a.tif.aux.xml", "b.tif.aux.xml"]:
            (img_dir / name).touch()
        (img_dir / "readme.txt").touch()

        tiff_files = sorted(img_dir.glob("*.tif"))
        tiff_files = [f for f in tiff_files if not f.name.endswith(".tif.aux.xml")]
        assert len(tiff_files) == 3

    def test_aux_xml_not_counted(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "test.tif").touch()
        (img_dir / "test.tif.aux.xml").touch()
        tiff_files = [f for f in img_dir.glob("*.tif") if f.suffix == ".tif"]
        assert len(tiff_files) == 1


# ===========================================================================
# Test: COCO resolution
# ===========================================================================


class TestCocoResolution:
    def test_resolve_tiff_path(self, tmp_path, synthetic_coco_data):
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(synthetic_coco_data))
        coco = CocoData(ann_path)
        tiff_path = coco.resolve_tiff_path(coco.images[0], tmp_path / "images")
        assert tiff_path.name == "9-368-464_5_18.tif"

    def test_resolve_split(self, tmp_path, synthetic_coco_data):
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(synthetic_coco_data))
        coco = CocoData(ann_path)
        splits = {"training": ["9-368-464_5_18.tif"], "test": []}
        assert coco.resolve_split(coco.images[0], splits) == "training"

    def test_resolve_split_missing(self, tmp_path, synthetic_coco_data):
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(synthetic_coco_data))
        coco = CocoData(ann_path)
        assert coco.resolve_split(coco.images[0], {"training": ["other.tif"]}) is None


# ===========================================================================
# Test: polygon bounds and geometry
# ===========================================================================


class TestPolygonBounds:
    def test_basic_bounds(self):
        seg = [10, 20, 30, 20, 30, 40, 10, 40]
        left, top, right, bottom = polygon_bbox(seg)
        assert (left, top, right, bottom) == (10, 20, 31, 41)

    def test_fractional_coords(self):
        seg = [10.2, 20.8, 30.1, 20.8, 30.1, 40.5, 10.2, 40.5]
        left, top, right, bottom = polygon_bbox(seg)
        assert left == 10
        assert top == 20
        assert right == 32
        assert bottom == 42

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="fewer than 3 points"):
            polygon_bbox([0, 0, 10, 10])

    def test_clip_expand_rect(self):
        assert clip_expand_rect(10, 10, 20, 20, 8, 512, 512) == (2, 2, 28, 28)

    def test_clip_expand_rect_at_boundary(self):
        assert clip_expand_rect(0, 0, 10, 10, 8, 512, 512) == (0, 0, 18, 18)


# ===========================================================================
# Test: square expansion in source coordinates
# ===========================================================================


class TestExpandToSquare:
    def test_already_square(self):
        assert expand_rect_to_square(10, 10, 30, 30, 512, 512) == (10, 10, 30, 30)

    def test_wider_expands_height(self):
        # 40x20 -> expand height to 40
        left, top, right, bottom = expand_rect_to_square(10, 20, 50, 40, 512, 512)
        assert right - left == 40
        assert bottom - top == 40
        assert top < 20  # expanded upward
        assert bottom > 40  # expanded downward

    def test_taller_expands_width(self):
        # 20x40 -> expand width to 40
        left, top, right, bottom = expand_rect_to_square(20, 10, 40, 50, 512, 512)
        assert right - left == 40
        assert bottom - top == 40
        assert left < 20
        assert right > 40

    def test_clips_at_boundary(self):
        # Near origin: expansion clips at 0, shifts right to preserve square
        left, top, right, bottom = expand_rect_to_square(0, 0, 10, 20, 512, 512)
        assert left == 0
        assert top == 0
        assert right == 20
        assert bottom == 20

    def test_clips_at_max_boundary(self):
        # Near edge: expansion clips at 512
        left, top, right, bottom = expand_rect_to_square(500, 500, 512, 512, 512, 512)
        assert right <= 512
        assert bottom <= 512

    def test_result_may_be_nonsquare_at_edge(self):
        # Large rect near boundary: result stays square if image is big enough
        left, top, right, bottom = expand_rect_to_square(0, 0, 400, 50, 512, 512)
        assert left == 0
        assert top == 0
        assert right == 400
        assert bottom == 400


# ===========================================================================
# Test: constant boundary padding
# ===========================================================================


class TestConstantBoundaryPadding:
    def test_fully_inside_tile(self):
        """Crop fully inside tile: no padding needed."""
        rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out = crop_with_constant_boundary_padding(rgb, 10, 10, 30, 30, 20)
        assert out.shape == (20, 20, 3)
        np.testing.assert_array_equal(out, rgb[10:30, 10:30])

    def test_extends_past_left_boundary(self):
        """Crop extends past left edge: black padding on left."""
        rgb = np.ones((50, 50, 3), dtype=np.uint8) * 100
        out = crop_with_constant_boundary_padding(rgb, -5, 10, 15, 30, 20)
        assert out.shape == (20, 20, 3)
        # Left 5 columns should be black
        assert np.all(out[:, :5] == 0)
        # Rest should be from the image
        assert np.all(out[:, 5:] == 100)

    def test_extends_past_top_boundary(self):
        """Crop extends past top edge: black padding on top."""
        rgb = np.ones((50, 50, 3), dtype=np.uint8) * 100
        out = crop_with_constant_boundary_padding(rgb, 10, -5, 30, 15, 20)
        assert out.shape == (20, 20, 3)
        assert np.all(out[:5, :] == 0)
        assert np.all(out[5:, :] == 100)

    def test_extends_past_both_boundaries(self):
        """Crop extends past left and top: black corner."""
        rgb = np.ones((50, 50, 3), dtype=np.uint8) * 100
        out = crop_with_constant_boundary_padding(rgb, -5, -5, 15, 15, 20)
        assert out.shape == (20, 20, 3)
        assert np.all(out[:5, :5] == 0)  # corner
        assert np.all(out[5:, 5:] == 100)

    def test_entirely_outside_tile(self):
        """Crop entirely outside tile: all black."""
        rgb = np.ones((50, 50, 3), dtype=np.uint8) * 100
        out = crop_with_constant_boundary_padding(rgb, 100, 100, 120, 120, 20)
        assert out.shape == (20, 20, 3)
        assert np.all(out == 0)


# ===========================================================================
# Test: RGB conversion
# ===========================================================================


class TestRgbConversion:
    def test_rgb_bands_selection(self, tmp_path):
        import tifffile
        data = np.zeros((7, 8, 8), dtype=np.uint8)
        data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 99
        tifffile.imwrite(str(tmp_path / "test.tif"), data)
        from coco_crop_utils import read_rgb_bands
        rgb = read_rgb_bands(tmp_path / "test.tif")
        assert rgb.shape == (8, 8, 3)
        assert rgb[0, 0, 0] == 10
        assert rgb[0, 0, 1] == 20
        assert rgb[0, 0, 2] == 30

    def test_nan_replaced_with_zero(self, tmp_path):
        import tifffile
        data = np.zeros((3, 4, 4), dtype=np.float32)
        data[0, 0, 0] = float("nan")
        tifffile.imwrite(str(tmp_path / "test.tif"), data)
        from coco_crop_utils import read_rgb_bands
        rgb = read_rgb_bands(tmp_path / "test.tif")
        assert rgb[0, 0, 0] == 0

    def test_negative_clipped_to_zero(self, tmp_path):
        import tifffile
        data = np.full((3, 4, 4), -10.0, dtype=np.float32)
        tifffile.imwrite(str(tmp_path / "test.tif"), data)
        from coco_crop_utils import read_rgb_bands
        assert np.all(read_rgb_bands(tmp_path / "test.tif") == 0)

    def test_over_255_clipped(self, tmp_path):
        import tifffile
        data = np.full((3, 4, 4), 300.0, dtype=np.float32)
        tifffile.imwrite(str(tmp_path / "test.tif"), data)
        from coco_crop_utils import read_rgb_bands
        assert np.all(read_rgb_bands(tmp_path / "test.tif") == 255)


# ===========================================================================
# Test: JPEG output
# ===========================================================================


class TestJpegWrite:
    def test_writes_jpeg(self, tmp_path):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        path = write_prepared_jpeg(img, tmp_path, "test_id", quality=90)
        assert path.exists()

    def test_refuses_overwrite_different(self, tmp_path):
        img1 = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8) * 255)
        write_prepared_jpeg(img1, tmp_path, "test_id")
        with pytest.raises(FileExistsError):
            write_prepared_jpeg(img2, tmp_path, "test_id")


# ===========================================================================
# Test: stable collision-safe IDs
# ===========================================================================


class TestSampleIds:
    def test_deterministic(self):
        assert make_sample_id("a.tif", 0, 0, 1) == make_sample_id("a.tif", 0, 0, 1)

    def test_different_annotations_different_ids(self):
        assert make_sample_id("a.tif", 0, 0, 1) != make_sample_id("a.tif", 0, 1, 1)

    def test_different_classes_different_ids(self):
        assert make_sample_id("a.tif", 0, 0, 1) != make_sample_id("a.tif", 0, 0, 2)

    def test_format_contains_ann_id(self):
        assert "__ann42--" in make_sample_id("a.tif", 0, 42, 1)

    def test_format_contains_source_stem(self):
        assert make_sample_id("images/test.tif", 0, 0, 1).startswith("test__ann0--")


class TestProbableBuildingGroups:
    def test_groups_highly_overlapping_crops_and_marks_conflicting_labels(self):
        rows = [
            {
                "status": "prepared", "source_relpath": "images/a.tif",
                "coco_annotation_id": 10, "roofsense_class": "Metal",
                "crop_left": 0, "crop_top": 0, "crop_right": 100, "crop_bottom": 100,
                "target_pixels": 400,
            },
            {
                "status": "prepared", "source_relpath": "images/a.tif",
                "coco_annotation_id": 11, "roofsense_class": "Gravel",
                "crop_left": 2, "crop_top": 2, "crop_right": 102, "crop_bottom": 102,
                "target_pixels": 300,
            },
            {
                "status": "prepared", "source_relpath": "images/a.tif",
                "coco_annotation_id": 12, "roofsense_class": "Metal",
                "crop_left": 200, "crop_top": 200, "crop_right": 250, "crop_bottom": 250,
                "target_pixels": 500,
            },
        ]

        assign_probable_building_groups(rows, iou_threshold=0.8)

        assert rows[0]["probable_building_group_id"] == rows[1]["probable_building_group_id"]
        assert rows[0]["probable_building_group_size"] == 2
        assert rows[0]["probable_building_group_ambiguous"] is True
        assert rows[0]["probable_building_representative"] is True
        assert rows[1]["probable_building_representative"] is False
        assert rows[2]["probable_building_group_size"] == 1
        assert rows[2]["probable_building_group_ambiguous"] is False


class TestRoofScaleEligibility:
    def test_rejects_small_annotation(self):
        from prepare_roofsense_coco_crops import roof_scale_rejection_reason

        reason = roof_scale_rejection_reason(
            (10, 10, 25, 150),
            {"id": 1, "category_id": 1},
            [],
        )
        assert reason == "bbox_short_side_lt_16"

    def test_rejects_annotation_contained_in_much_larger_material_region(self):
        from prepare_roofsense_coco_crops import roof_scale_rejection_reason

        seed = {"id": 1, "category_id": 6}
        annotations = [{
            "id": 2, "category_id": 2,
            "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]],
        }]
        reason = roof_scale_rejection_reason((25, 25, 75, 75), seed, annotations)
        assert reason == "contained_in_larger_material_region"

    def test_accepts_roof_scale_annotation(self):
        from prepare_roofsense_coco_crops import roof_scale_rejection_reason

        reason = roof_scale_rejection_reason(
            (10, 10, 60, 60),
            {"id": 1, "category_id": 1},
            [],
        )
        assert reason is None


# ===========================================================================
# Test: building footprint extraction
# ===========================================================================


class TestBuildingFootprint:
    def test_finds_nonblack_component(self):
        """Nonblack region containing the seed is found."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[10:30, 10:30] = 100  # nonblack block

        seg = [12, 12, 20, 12, 20, 20, 12, 20]  # inside the block
        anns = [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [seg]}]

        result = find_footprint_and_crop_rect(rgb, seg, 0, 1, anns, 8, 64)
        assert result.component_pixel_count > 0
        assert result.footprint_left <= 12
        assert result.footprint_right >= 20

    def test_seed_entirely_on_black(self):
        """Seed on black pixels falls back to polygon bbox."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        seg = [10, 10, 20, 10, 20, 20, 10, 20]
        anns = [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [seg]}]

        result = find_footprint_and_crop_rect(rgb, seg, 0, 1, anns, 8, 64)
        assert result.component_pixel_count == 0
        assert result.footprint_left == 10

    def test_multi_material_detection(self):
        """Component with annotations of different classes is flagged."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[10:40, 10:40] = 100  # one big nonblack region

        seg1 = [12, 12, 20, 12, 20, 20, 12, 20]  # class 1
        seg2 = [25, 25, 35, 25, 35, 35, 25, 35]  # class 2, same region
        anns = [
            {"id": 0, "image_id": 0, "category_id": 1, "segmentation": [seg1]},
            {"id": 1, "image_id": 0, "category_id": 2, "segmentation": [seg2]},
        ]

        result = find_footprint_and_crop_rect(rgb, seg1, 0, 1, anns, 8, 64)
        assert result.multi_material_component is True

    def test_single_material_no_flag(self):
        """Component with only same-class annotations is not flagged."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[10:40, 10:40] = 100

        seg1 = [12, 12, 20, 12, 20, 20, 12, 20]
        seg2 = [25, 25, 35, 25, 35, 35, 25, 35]
        anns = [
            {"id": 0, "image_id": 0, "category_id": 1, "segmentation": [seg1]},
            {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [seg2]},
        ]

        result = find_footprint_and_crop_rect(rgb, seg1, 0, 1, anns, 8, 64)
        assert result.multi_material_component is False

    def test_component_spans_entire_tile(self):
        """Large component triggers potential_building_merge with enough annotations."""
        rgb = np.ones((64, 64, 3), dtype=np.uint8) * 50  # all nonblack

        anns = []
        for i in range(5):
            seg = [i * 10, i * 10, i * 10 + 8, i * 10, i * 10 + 8, i * 10 + 8, i * 10, i * 10 + 8]
            anns.append({"id": i, "image_id": 0, "category_id": 1, "segmentation": [seg]})

        result = find_footprint_and_crop_rect(rgb, anns[0]["segmentation"][0], 0, 1, anns, 8, 64)
        assert result.potential_building_merge is True


# ===========================================================================
# Test: polygon rasterization
# ===========================================================================


class TestPolygonRasterization:
    def test_square_polygon(self):
        seg = [2, 2, 7, 2, 7, 7, 2, 7]
        mask = rasterize_polygon_mask(seg, 10, 10)
        assert mask[2, 2] == True
        assert mask[6, 6] == True
        assert mask[1, 1] == False

    def test_offset_rasterization(self):
        seg = [10, 10, 20, 10, 20, 20, 10, 20]
        mask = rasterize_polygon_mask(seg, 15, 15, left=8, top=8)
        assert mask[2, 2] == True


# ===========================================================================
# Test: diagnostic flags
# ===========================================================================


class TestDiagnosticFlags:
    def test_isolated_roof_no_flags(self):
        crop_size = 100
        target = np.zeros((crop_size, crop_size), dtype=bool)
        target[10:60, 10:60] = True  # 25%
        other_masks = {0: (target, 1)}
        result = measure_crop_composition(crop_size, crop_size, target, other_masks, 0, 1)
        assert not result["multi_label_overlap"]
        assert not result["other_class_in_context"]
        assert not result["same_class_neighbor"]
        assert not result["low_target_occupancy"]

    def test_same_class_neighbor_flag(self):
        crop_size = 100
        target = np.zeros((crop_size, crop_size), dtype=bool)
        target[10:30, 10:30] = True
        other_same = np.zeros((crop_size, crop_size), dtype=bool)
        other_same[40:60, 40:60] = True
        other_masks = {0: (target, 1), 1: (other_same, 1)}
        result = measure_crop_composition(crop_size, crop_size, target, other_masks, 0, 1)
        assert result["same_class_neighbor"]

    def test_multi_label_overlap(self):
        crop_size = 100
        target = np.zeros((crop_size, crop_size), dtype=bool)
        target[10:50, 10:50] = True
        diff_overlap = np.zeros((crop_size, crop_size), dtype=bool)
        diff_overlap[30:70, 30:70] = True
        other_masks = {0: (target, 1), 1: (diff_overlap, 2)}
        result = measure_crop_composition(crop_size, crop_size, target, other_masks, 0, 1)
        assert result["multi_label_overlap"]

    def test_low_target_occupancy(self):
        crop_size = 100
        target = np.zeros((crop_size, crop_size), dtype=bool)
        target[45:55, 45:55] = True  # 1%
        other_masks = {0: (target, 1)}
        result = measure_crop_composition(crop_size, crop_size, target, other_masks, 0, 1)
        assert result["low_target_occupancy"]

    def test_mask_agreement_flag(self):
        assert compute_mask_agreement_flag(0.95) == (True, False)
        assert compute_mask_agreement_flag(0.80) == (False, True)
        assert compute_mask_agreement_flag(0.0) == (False, False)


# ===========================================================================
# Test: mask agreement calculation
# ===========================================================================


class TestMaskAgreement:
    def test_perfect_agreement(self):
        sem_mask = np.ones((10, 10), dtype=np.int32)
        poly_mask = np.zeros((10, 10), dtype=bool)
        poly_mask[2:8, 2:8] = True
        total, agreeing, fraction = compute_mask_agreement(sem_mask, poly_mask, 1)
        assert fraction == pytest.approx(1.0)

    def test_no_agreement(self):
        sem_mask = np.full((10, 10), 2, dtype=np.int32)
        poly_mask = np.zeros((10, 10), dtype=bool)
        poly_mask[2:8, 2:8] = True
        total, agreeing, fraction = compute_mask_agreement(sem_mask, poly_mask, 1)
        assert fraction == pytest.approx(0.0)

    def test_partial_agreement(self):
        sem_mask = np.zeros((10, 10), dtype=np.int32)
        sem_mask[2:8, 2:5] = 1
        sem_mask[2:8, 5:8] = 2
        poly_mask = np.zeros((10, 10), dtype=bool)
        poly_mask[2:8, 2:8] = True
        total, agreeing, fraction = compute_mask_agreement(sem_mask, poly_mask, 1)
        assert total == 36
        assert agreeing == 18
        assert fraction == pytest.approx(0.5)

    def test_background_excluded(self):
        sem_mask = np.zeros((10, 10), dtype=np.int32)
        poly_mask = np.zeros((10, 10), dtype=bool)
        poly_mask[2:8, 2:8] = True
        total, agreeing, fraction = compute_mask_agreement(sem_mask, poly_mask, 1)
        assert total == 0


# ===========================================================================
# Test: manifest
# ===========================================================================


class TestManifest:
    def test_write_and_read(self, tmp_path):
        rows = [{col: f"val_{col}" for col in MANIFEST_COLUMNS}]
        path = tmp_path / "manifest.csv"
        write_manifest(path, rows)
        with open(path) as f:
            loaded = list(csv.DictReader(f))
        assert len(loaded) == 1
        for col in MANIFEST_COLUMNS:
            assert loaded[0][col] == f"val_{col}"

    def test_all_columns_present(self, tmp_path):
        rows = [{col: "" for col in MANIFEST_COLUMNS}]
        path = tmp_path / "manifest.csv"
        write_manifest(path, rows)
        with open(path) as f:
            fieldnames = csv.DictReader(f).fieldnames
        for col in MANIFEST_COLUMNS:
            assert col in fieldnames


# ===========================================================================
# Test: atomic writes
# ===========================================================================


class TestAtomicWrites:
    def test_manifest_atomic(self, tmp_path):
        rows = [{col: "test" for col in MANIFEST_COLUMNS}]
        path = tmp_path / "manifest.csv"
        write_manifest(path, rows)
        assert path.exists()
        assert len(list(tmp_path.glob("*.csv"))) == 1

    def test_json_atomic(self, tmp_path):
        path = tmp_path / "metadata.json"
        write_json_metadata(path, {"key": "value"})
        assert json.loads(path.read_text())["key"] == "value"


# ===========================================================================
# Test: contact sheet determinism
# ===========================================================================


class TestContactSheet:
    def test_deterministic(self, tmp_path):
        from diagnostics import build_contact_sheet_clean
        img_dir = tmp_path / "images" / "cls"
        img_dir.mkdir(parents=True)
        rows = []
        for i in range(5):
            img = Image.fromarray(np.full((32, 32, 3), i * 50, dtype=np.uint8))
            img.save(str(img_dir / f"img_{i}.jpg"))
            rows.append({
                "sample_id": f"img_{i}", "roofsense_class": "Metal",
                "split": "training", "target_crop_fraction": "0.5",
                "status": "prepared", "prepared_filename": f"cls/img_{i}.jpg",
            })
        p1 = tmp_path / "s1.jpg"
        p2 = tmp_path / "s2.jpg"
        build_contact_sheet_clean(tmp_path / "images", rows, p1, per_class=2)
        build_contact_sheet_clean(tmp_path / "images", rows, p2, per_class=2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_clean_sheet_excludes_flagged_rows(self, tmp_path):
        from diagnostics import build_contact_sheet_clean

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (32, 32), "white").save(img_dir / "flagged.jpg")
        rows = [{
            "sample_id": "flagged", "roofsense_class": "Metal",
            "split": "training", "target_crop_fraction": "0.5",
            "status": "prepared", "prepared_filename": "flagged.jpg",
            "low_target_occupancy": True,
        }]

        output = tmp_path / "clean.jpg"
        build_contact_sheet_clean(img_dir, rows, output)
        assert Image.open(output).size == (400, 40)


# ===========================================================================
# Test: pixel invariance
# ===========================================================================


class TestPixelInvariance:
    def test_jpeg_unchanged_by_diagnostics(self, tmp_path):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        path = write_prepared_jpeg(img, tmp_path, "test_id", quality=95)
        b1 = path.read_bytes()
        from diagnostics import build_contact_sheet_clean
        rows = [{
            "sample_id": "test_id", "roofsense_class": "Metal",
            "split": "test", "target_crop_fraction": "0.5",
            "status": "prepared", "prepared_filename": "test_id.jpg",
        }]
        build_contact_sheet_clean(tmp_path, rows, tmp_path / "sheet.jpg")
        assert path.read_bytes() == b1


# ===========================================================================
# Test: mapping version
# ===========================================================================


class TestMappingVersion:
    def test_mapping_coverage(self):
        expected = {"Ceramic Tile", "Dark-coloured Membrane", "Gravel",
                    "Light-coloured Membrane", "Light-permitting Surface",
                    "Metal", "Solar Panel", "Vegetation"}
        assert set(REMOTECLIP_MAPPING.keys()) == expected

    def test_mapping_version(self):
        assert MAPPING_VERSION == "1.0"

    def test_supported_excludes_solar(self):
        """Solar Panel (8) is in SUPPORTED_COCO_IDS but excluded at preparation time."""
        assert 8 in SUPPORTED_COCO_IDS
        # The script excludes it via SOLAR_PANEL_COCO_ID constant

    def test_invalid_excluded(self):
        assert 4 not in SUPPORTED_COCO_IDS
        assert 0 not in SUPPORTED_COCO_IDS


# ===========================================================================
# Test: fingerprints
# ===========================================================================


class TestFingerprints:
    def test_file_deterministic(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("hello")
        assert fingerprint_file(f) == fingerprint_file(f)

    def test_file_content_sensitive(self, tmp_path):
        f1 = tmp_path / "a.txt"; f1.write_text("hello")
        f2 = tmp_path / "b.txt"; f2.write_text("world")
        assert fingerprint_file(f1) != fingerprint_file(f2)

    def test_strings_order_independent(self):
        assert fingerprint_strings(["a", "b"]) == fingerprint_strings(["b", "a"])


# ===========================================================================
# Test: new manifest columns for building footprint
# ===========================================================================


class TestManifestColumns:
    def test_has_footprint_columns(self):
        """Manifest schema includes building footprint fields."""
        assert "footprint_left" in MANIFEST_COLUMNS
        assert "footprint_top" in MANIFEST_COLUMNS
        assert "footprint_right" in MANIFEST_COLUMNS
        assert "footprint_bottom" in MANIFEST_COLUMNS
        assert "component_pixel_count" in MANIFEST_COLUMNS
        assert "shared_annotation_count" in MANIFEST_COLUMNS
        assert "multi_material_component" in MANIFEST_COLUMNS
        assert "potential_building_merge" in MANIFEST_COLUMNS


# ===========================================================================
# Integration test: synthetic tile
# ===========================================================================


class TestIntegration:
    def test_full_pipeline(self, full_synthetic_dataset):
        from prepare_roofsense_coco_crops import run_preparation

        ds = full_synthetic_dataset
        output_dir = ds / "prepared_output"

        result = run_preparation(
            dataset_root=ds, output_dir=output_dir,
            padding_px=8, jpeg_quality=95, contact_sheet_per_class=2, reset=True,
        )

        # 7 annotations: Invalid + Solar + one tiny Gravel region excluded
        assert result["excluded"] == 3
        assert result["prepared"] + result["failed"] == 4

        # Verify output structure
        assert (output_dir / "manifest.csv").exists()
        assert (output_dir / "preparation_metadata.json").exists()
        assert (output_dir / "contact_sheet.jpg").exists()
        assert (output_dir / "contact_sheet_diagnostics.jpg").exists()

        # Verify manifest
        with open(output_dir / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 7  # all annotations attempted

        # Verify exclusions
        excluded = [r for r in rows if r["status"].startswith("excluded")]
        assert len(excluded) == 3
        excluded_classes = {r["roofsense_class"] for r in excluded}
        assert "Invalid" in excluded_classes
        assert "Solar Panel" in excluded_classes
        assert any(r["status"] == "excluded_non_roof_scale" for r in excluded)
        assert all(r["sample_id"] for r in excluded)
        assert all(r["source_relpath"] == "images/9-368-464_5_18.tif" for r in excluded)
        assert all(r["source_stem"] == "9-368-464_5_18" for r in excluded)
        assert all(r["split"] == "training" for r in excluded)

        # Verify prepared rows have JPEGs
        for row in rows:
            if row["status"] == "prepared" and row["prepared_filename"]:
                jpeg_path = output_dir / "images" / row["prepared_filename"]
                assert jpeg_path.exists()
                assert row["crop_method"] == "annotation_bbox"
                assert row["footprint_left"] == row["polygon_bbox_x"]
                assert row["footprint_top"] == row["polygon_bbox_y"]
                assert row["probable_building_group_id"]

        # Verify metadata
        metadata = json.loads((output_dir / "preparation_metadata.json").read_text())
        assert metadata["counts"]["solar_panel_excluded"] == 1
        assert metadata["counts"]["invalid_excluded"] == 1
        assert metadata["counts"]["non_roof_scale_excluded"] == 1
        assert metadata["settings"]["crop_method"] == "annotation_bbox"
        assert Image.open(output_dir / "contact_sheet_diagnostics.jpg").width == 4 * 224

    def test_deterministic(self, full_synthetic_dataset):
        from prepare_roofsense_coco_crops import run_preparation

        ds = full_synthetic_dataset
        out1 = ds / "out1"
        out2 = ds / "out2"
        run_preparation(ds, out1, padding_px=8, jpeg_quality=95, reset=True)
        run_preparation(ds, out2, padding_px=8, jpeg_quality=95, reset=True)

        with open(out1 / "manifest.csv") as f:
            rows1 = list(csv.DictReader(f))
        with open(out2 / "manifest.csv") as f:
            rows2 = list(csv.DictReader(f))

        assert len(rows1) == len(rows2)
        for r1, r2 in zip(rows1, rows2):
            assert r1["sample_id"] == r2["sample_id"]
            assert r1["status"] == r2["status"]
            if r1["status"] == "prepared":
                j1 = out1 / "images" / r1["prepared_filename"]
                j2 = out2 / "images" / r2["prepared_filename"]
                assert j1.read_bytes() == j2.read_bytes()

    def test_incompatible_resume_rejected(self, full_synthetic_dataset):
        from prepare_roofsense_coco_crops import run_preparation

        ds = full_synthetic_dataset
        out = ds / "out"
        run_preparation(ds, out, padding_px=8, reset=True)
        with pytest.raises(SystemExit):
            run_preparation(ds, out, padding_px=12, reset=False)

    def test_resume_rejects_changed_source_fingerprint(self, full_synthetic_dataset):
        from prepare_roofsense_coco_crops import run_preparation

        ds = full_synthetic_dataset
        out = ds / "out"
        run_preparation(ds, out, reset=True)
        annotations = ds / "annotations" / "annotations.json"
        annotations.write_text(annotations.read_text() + "\n")
        with pytest.raises(SystemExit):
            run_preparation(ds, out, reset=False)

    def test_resume_rejects_changed_script_version(self, full_synthetic_dataset):
        from prepare_roofsense_coco_crops import run_preparation

        ds = full_synthetic_dataset
        out = ds / "out"
        run_preparation(ds, out, reset=True)
        metadata_path = out / "preparation_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["script_version"] = "stale"
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(SystemExit):
            run_preparation(ds, out, reset=False)


class TestSafeReset:
    def test_refuses_unmarked_directory_with_images(self, tmp_path):
        from prepare_roofsense_coco_crops import _safe_reset

        output = tmp_path / "not_preparation_output"
        images = output / "images"
        images.mkdir(parents=True)
        sentinel = images / "keep.txt"
        sentinel.write_text("user data")

        with pytest.raises(RuntimeError, match="recognized preparation output"):
            _safe_reset(output)
        assert sentinel.read_text() == "user data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
