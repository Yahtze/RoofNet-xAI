import os
import json
import rasterio
from rasterio import features
from shapely import wkt
from shapely.ops import transform as shapely_transform
from PIL import Image
import numpy as np
from glob import glob
from pyproj import Transformer

ROOT = './' # Insert path to the root directory containing labels and image files

LABELS_DIR = os.path.join(ROOT, "labels")
IMAGES_DIR = os.path.join(ROOT, "images") # Insert path to the xBD images directory
IMAGES_PNG_DIR = os.path.join(ROOT, "images_png")
CROPS_DIR = os.path.join(ROOT, "crops")
MASKS_DIR = os.path.join(ROOT, "masks")

os.makedirs(IMAGES_PNG_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

def extract_crops_and_masks(label_file):
    with open(label_file, 'r') as f:
        data = json.load(f)

    base_name = os.path.splitext(os.path.basename(label_file))[0]
    tif_path = os.path.join(IMAGES_DIR, base_name + ".tif")
    if not os.path.exists(tif_path):
        print(f"Missing .tif file for {base_name}")
        return

    with rasterio.open(tif_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        def reproject_geom(geom):
            return shapely_transform(transformer.transform, geom)

        image = src.read([1, 2, 3])  # RGB
        transform = src.transform
        full_img = np.moveaxis(image, 0, -1)  # CHW to HWC
        if full_img.dtype != np.uint8:
            full_img = ((full_img - full_img.min()) / (full_img.max() - full_img.min()) * 255).astype(np.uint8)
        Image.fromarray(full_img).save(os.path.join(IMAGES_PNG_DIR, f"{base_name}.png"))

        height, width = image.shape[1], image.shape[2]

        for feature in data["features"]["lng_lat"]:
            if feature["properties"]["feature_type"] != "building":
                print(f"out of curiosity this is the feature type {feature["properties"]["feature_type"]}")
                continue

            uid = feature["properties"]["uid"]
            geom = reproject_geom(wkt.loads(feature["wkt"]))

            # Full-size mask with just this building
            mask = features.rasterize([(geom, 1)], out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
            mask_path = os.path.join(MASKS_DIR, f"{base_name}_{uid}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

            # Crop just the building area for image crop
            bounds = geom.bounds
            col_min, row_min = ~transform * (bounds[0], bounds[3])
            col_max, row_max = ~transform * (bounds[2], bounds[1])
            row_min, row_max = int(np.floor(row_min)), int(np.ceil(row_max))
            col_min, col_max = int(np.floor(col_min)), int(np.ceil(col_max))
            row_min, row_max = max(0, row_min), min(height, row_max)
            col_min, col_max = max(0, col_min), min(width, col_max)
            row_min, row_max = sorted([int(np.floor(row_min)), int(np.ceil(row_max))])
            col_min, col_max = sorted([int(np.floor(col_min)), int(np.ceil(col_max))])

            crop = full_img[row_min:row_max, col_min:col_max]
            print(f"UID: {uid}, Bounds: {bounds}, Pixel bounds: rows {row_min}-{row_max}, cols {col_min}-{col_max}")
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"Skipping empty crop for {base_name}_{uid}")
            else:
                crop_path = os.path.join(CROPS_DIR, f"{base_name}_{uid}.png")
                Image.fromarray(crop).save(crop_path)

# Process all *_pre_disaster.json files
label_files = glob(os.path.join(LABELS_DIR, "**_pre_disaster.json", recursive=True))
for file in label_files:
    extract_crops_and_masks(file)
