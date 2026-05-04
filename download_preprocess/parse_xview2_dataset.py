"""
parse_xview2_dataset.py
====================
Usage: parsing individual building polygon information from xBD files.

Call via:
    python parse_xview2_dataset.py
"""

import os
import json
import rasterio
from rasterio import features
from shapely import shapely_transform, wkt
from pyproj import Transformer
import numpy as np
import glob
from PIL import Image


# --- Directory Configuration ---
ROOT = './'  # Root directory path for data processing
LABELS_DIR = os.path.join(ROOT, "labels")
IMAGES_DIR = os.path.join(ROOT, "images")      # Source directory for xBD .tif images
IMAGES_PNG_DIR = os.path.join(ROOT, "images_png")
CROPS_DIR = os.path.join(ROOT, "crops")
MASKS_DIR = os.path.join(ROOT, "masks")

# Initialize output directory structure
os.makedirs(IMAGES_PNG_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

def extract_crops_and_masks(label_file):
    """
    Parses a single xBD JSON label file, reprojects building geometries,
    and saves individual building crops and binary masks.
    """
    with open(label_file, 'r') as f:
        data = json.load(f)

    # Derive base filename and verify corresponding imagery exists
    base_name = os.path.splitext(os.path.basename(label_file))[0]
    tif_path = os.path.join(IMAGES_DIR, base_name + ".tif")
    if not os.path.exists(tif_path):
        print(f"Missing .tif file for {base_name}")
        return

    with rasterio.open(tif_path) as src:
        # Initialize coordinate transformer (WGS84 to Image CRS)
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        def reproject_geom(geom):
            """Helper function to reproject shapely geometries."""
            return shapely_transform(transformer.transform, geom)

        # Read RGB bands and handle normalization for 8-bit PNG conversion
        image = src.read([1, 2, 3])  
        transform = src.transform
        full_img = np.moveaxis(image, 0, -1)  # Reorder from CHW to HWC
        
        if full_img.dtype != np.uint8:
            full_img = ((full_img - full_img.min()) / 
                        (full_img.max() - full_img.min()) * 255).astype(np.uint8)
        
        # Save the full image as a standard PNG
        Image.fromarray(full_img).save(os.path.join(IMAGES_PNG_DIR, f"{base_name}.png"))

        height, width = image.shape[1], image.shape[2]

        # Iterate through features in the JSON file
        for feature in data["features"]["lng_lat"]:
            # Filter for building types only
            if feature["properties"]["feature_type"] != "building":
                print(f"Skipping non-building feature type: {feature['properties']['feature_type']}")
                continue

            uid = feature["properties"]["uid"]
            geom = reproject_geom(wkt.loads(feature["wkt"]))

            # --- Mask Generation ---
            # Rasterize the building geometry into a binary mask
            mask = features.rasterize([(geom, 1)], out_shape=(height, width), 
                                      transform=transform, fill=0, dtype=np.uint8)
            mask_path = os.path.join(MASKS_DIR, f"{base_name}_{uid}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

            # --- Crop Generation ---
            # Convert spatial bounds to pixel coordinates using the inverse transform
            bounds = geom.bounds
            col_min, row_min = ~transform * (bounds[0], bounds[3])
            col_max, row_max = ~transform * (bounds[2], bounds[1])
            
            # Ensure integer indices and handle image boundary constraints
            row_min, row_max = sorted([int(np.floor(row_min)), int(np.ceil(row_max))])
            col_min, col_max = sorted([int(np.floor(col_min)), int(np.ceil(col_max))])
            
            row_min, row_max = max(0, row_min), min(height, row_max)
            col_min, col_max = max(0, col_min), min(width, col_max)

            # Extract the subset from the full image array
            crop = full_img[row_min:row_max, col_min:col_max]
            
            # Validate and save building crop
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"Skipping empty crop for {base_name}_{uid}")
            else:
                print(f"UID: {uid}, Pixel bounds: rows {row_min}-{row_max}, cols {col_min}-{col_max}")
                crop_path = os.path.join(CROPS_DIR, f"{base_name}_{uid}.png")
                Image.fromarray(crop).save(crop_path)

# --- Execution ---
# Batch process all pre-disaster label files found in the labels directory
label_files = glob(os.path.join(LABELS_DIR, "**_pre_disaster.json"), recursive=True)
for file in label_files:
    extract_crops_and_masks(file)
