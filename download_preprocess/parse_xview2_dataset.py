"""
parse_xview2_dataset.py
====================
Usage: parsing individual building polygon information from xBD files.

Call via:
    python parse_xview2_dataset.py --xbd_dir <path_to_xbd_dataset>
"""

import os
import json
import rasterio
from rasterio import features
from shapely import wkt
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import numpy as np
from glob import glob
from PIL import Image
import argparse

def extract_crops_and_masks(label_file, images_dir, images_png_dir, crops_dir, masks_dir):
    """
    Parses a single xBD JSON label file, reprojects building geometries,
    and saves individual building crops and binary masks.
    """
    with open(label_file, 'r') as f:
        data = json.load(f)

    base_name = os.path.splitext(os.path.basename(label_file))[0]
    tif_path = os.path.join(images_dir, base_name + ".tif")
    
    if not os.path.exists(tif_path):
        print(f"Missing .tif file for {base_name}")
        return

    with rasterio.open(tif_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        def reproject_geom(geom):
            return shapely_transform(transformer.transform, geom)

        image = src.read([1, 2, 3])  
        transform = src.transform
        full_img = np.moveaxis(image, 0, -1)  
        
        if full_img.dtype != np.uint8:
            full_img = ((full_img - full_img.min()) / 
                        (full_img.max() - full_img.min()) * 255).astype(np.uint8)
        
        Image.fromarray(full_img).save(os.path.join(images_png_dir, f"{base_name}.png"))

        height, width = image.shape[1], image.shape[2]

        for feature in data["features"]["lng_lat"]:
            if feature["properties"]["feature_type"] != "building":
                print(f"Skipping non-building feature type: {feature['properties']['feature_type']}")
                continue

            uid = feature["properties"]["uid"]
            geom = reproject_geom(wkt.loads(feature["wkt"]))

            # --- Mask Generation ---
            mask = features.rasterize([(geom, 1)], out_shape=(height, width), 
                                      transform=transform, fill=0, dtype=np.uint8)
            mask_path = os.path.join(masks_dir, f"{base_name}_{uid}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

            # --- Crop Generation ---
            bounds = geom.bounds
            col_min, row_min = ~transform * (bounds[0], bounds[3])
            col_max, row_max = ~transform * (bounds[2], bounds[1])
            
            row_min, row_max = sorted([int(np.floor(row_min)), int(np.ceil(row_max))])
            col_min, col_max = sorted([int(np.floor(col_min)), int(np.ceil(col_max))])
            
            row_min, row_max = max(0, row_min), min(height, row_max)
            col_min, col_max = max(0, col_min), min(width, col_max)

            crop = full_img[row_min:row_max, col_min:col_max]
            
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"Skipping empty crop for {base_name}_{uid}")
            else:
                print(f"UID: {uid}, Pixel bounds: rows {row_min}-{row_max}, cols {col_min}-{col_max}")
                crop_path = os.path.join(crops_dir, f"{base_name}_{uid}.png")
                Image.fromarray(crop).save(crop_path)

def main(xbd_dir):
    # --- Directory Configuration ---
    labels_dir = os.path.join(xbd_dir, "labels")
    images_dir = os.path.join(xbd_dir, "images")      
    images_png_dir = os.path.join(xbd_dir, "images_png")
    crops_dir = os.path.join(xbd_dir, "crops")
    masks_dir = os.path.join(xbd_dir, "masks")

    os.makedirs(images_png_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Batch process
    label_files = glob(os.path.join(labels_dir, "**_pre_disaster.json"), recursive=True)
    for file in label_files:
        extract_crops_and_masks(file, images_dir, images_png_dir, crops_dir, masks_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse xView2 dataset geometries.")
    parser.add_argument('--xbd_dir', type=str, required=True, help="Root directory path to the downloaded xBD dataset")
    args = parser.parse_args()

    main(args.xbd_dir)