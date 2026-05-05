"""
verify_material_classes.py
====================
Usage: Validates that the physical folder placement of an image matches the 
allowed roof materials for its respective city. If it does not match, the image 
is moved to a 'reassess' subfolder.

Call via:
    python verify_material_classes.py --dataset_dir <path> --city_materials_csv <path> --building_csv <path>
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import shutil
import ast

def main(dataset_dir, city_materials_csv, building_csv):
    # === LOAD CITY ALLOWED MATERIALS ===
    print(f"Loading allowed materials from {city_materials_csv}...")
    df_mats = pd.read_csv(city_materials_csv, encoding='latin-1')
    # Create a normalized city key (lowercase, no spaces, no commas)
    df_mats['city_key'] = df_mats['City'].str.lower().str.replace(" ", "_").str.replace(",", "")
    city_to_materials = {}

    # Parse the 'Roof Materials' JSON-like strings into Python lists
    for _, row in df_mats.iterrows():
        try:
            city = row['city_key']
            material_list = [m['class'] for m in ast.literal_eval(row['Roof Materials'])]
            city_to_materials[city] = material_list
        except Exception as e:
            print(f"Error parsing materials for city: {row['City']}: {e}")

    # === LOAD BUILDING METADATA ===
    print(f"Loading building mappings from {building_csv}...")
    df_buildings = pd.read_csv(building_csv)
    
    # Normalize column names to lowercase to ensure we find 'filename' and 'city'
    df_buildings.columns = [c.lower() for c in df_buildings.columns]
    
    if 'filename' not in df_buildings.columns or 'city' not in df_buildings.columns:
        raise ValueError(f"The building CSV must contain 'filename' and 'city' columns. Found: {df_buildings.columns}")

    # Normalize the city names in the building CSV to match our dictionary keys
    df_buildings['city_key'] = df_buildings['city'].astype(str).str.lower().str.replace(" ", "_").str.replace(",", "")
    
    # Create a rapid lookup dictionary mapping 'image_name.png' -> 'city_key'
    filename_to_city = dict(zip(df_buildings['filename'], df_buildings['city_key']))

    # === MAIN VALIDATION SCRIPT ===
    print(f"Validating images in {dataset_dir}...")
    moved_count = 0

    for material_folder in os.listdir(dataset_dir):
        material_path = os.path.join(dataset_dir, material_folder)
        if not os.path.isdir(material_path):
            continue

        reassess_dir = os.path.join(material_path, "reassess")

        for img_file in os.listdir(material_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                
                # Look up the city directly using the filename
                city_key = filename_to_city.get(img_file)
                
                if not city_key or city_key == "nan":
                    print(f"⚠️ City mapping not found in CSV for file: {img_file}")
                    continue

                allowed_materials = city_to_materials.get(city_key)
                if allowed_materials is None:
                    print(f"⚠️ No material list found in resource CSV for city: {city_key}")
                    continue

                # If the folder the image is currently in is NOT in the allowed list for that city
                if material_folder not in allowed_materials:
                    os.makedirs(reassess_dir, exist_ok=True) # Create folder only if needed
                    src_path = os.path.join(material_path, img_file)
                    dst_path = os.path.join(reassess_dir, img_file)
                    
                    shutil.move(src_path, dst_path)
                    print(f"Moved {img_file} from {material_folder}/ to reassess/ (city: {city_key})")
                    moved_count += 1
                    
    print(f"\n✅ Validation complete. Moved {moved_count} images to reassess folders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify material classes using a building metadata CSV.")
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help="Path to the dataset base directory containing material subfolders")
    parser.add_argument('--city_materials_csv', type=str, default='resources/City_Roof_Materials_with_Continent_and_Country_Centroids.csv', 
                        help="Path to the reference CSV with allowed materials per city (e.g., City_Roof_Materials_with_Continent_and_Country_Centroids.csv)")
    parser.add_argument('--building_csv', type=str, required=True, 
                        help="Path to the CSV containing building metadata mapping filenames to cities")
    args = parser.parse_args()

    main(args.dataset_dir, args.city_materials_csv, args.building_csv)