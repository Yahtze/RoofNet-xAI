"""
GOAL: Generate metadata CSV for RoofNet images by merging physical image files 
with an existing CSV containing per-building latitude, longitude, and other metadata.

Call via: 
    python generate_roof_material_metadata.py --dataset_dir <RoofNet_dataset_dir> 
    --building_csv <building_coordinates_csv>
"""

import os
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import osmnx as ox
from tqdm import tqdm

# Configure OSMnx to prevent excessive logging and timeout gracefully
ox.settings.log_console = False
ox.settings.timeout = 15

# === OSM ATTRIBUTE EXTRACTION ===
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import numpy as np
import time

def batch_extract_osm_metadata(df_metadata):
    """
    Takes a dataframe with latitude/longitude and performs a local spatial join
    against bulk-downloaded OSM data to prevent API throttling.
    """
    print("Converting coordinates to spatial geometries...")
    # 1. Convert your Pandas DataFrame to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_metadata['longitude'], df_metadata['latitude'])]
    gdf_points = gpd.GeoDataFrame(df_metadata, geometry=geometry, crs="EPSG:4326")
    
    # Create empty columns for our results
    gdf_points['height'] = np.nan
    gdf_points['numstories'] = np.nan
    gdf_points['fpArea'] = np.nan
    gdf_points['roofshape'] = pd.Series(np.nan, index=gdf_points.index, dtype=object)

    
    # 2. Group your data geographically. 
    # If you have a 'City' column, group by that. If not, we can cluster or just 
    # use the bounding box of the whole dataset (if it's not a massive global spread).
    # Assuming 'City' is in your CSV from the download script:
    
    if 'city' not in gdf_points.columns:
        # Fallback: Treat all points as one region (Dangerous if points span the globe)
        groups = [("All Data", gdf_points)]
    else:
        groups = gdf_points.groupby('city')

    for group_name, group in groups:
        print(f"\nProcessing batch for group: {group_name} ({len(group)} buildings)")
        
        # total_bounds returns (minx, miny, maxx, maxy) -> (West, South, East, North)
        west, south, east, north = group.total_bounds
        
        # Buffer to catch edge buildings (~100m)
        buffer = 0.001 
        
        try:
            print("  -> Downloading bulk OSM buildings for bounding box...")
            
            # --- CORRECT OSMnx v2.0+ SYNTAX ---
            # Order MUST be: (Left, Bottom, Right, Top) -> (West, South, East, North)
            bbox = (west - buffer, south - buffer, east + buffer, north + buffer)
            
            buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
            
            # Filter for polygons
            buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            if buildings.empty:
                print("  -> No buildings found in OSM for this area.")
                continue
                
            # Project to UTM to calculate area in square meters
            buildings_proj = ox.projection.project_gdf(buildings)
            buildings['fpArea'] = buildings_proj.geometry.area
            
            print("  -> Performing local spatial join...")
            joined = gpd.sjoin(group, buildings, how="left", predicate="intersects")
                        
            # Map the data back to your original GeoDataFrame
            for idx, row in joined.iterrows():
                if pd.notna(row.get('index_right')): # If a building matched
                    
                    # Map Height
                    if 'height' in row and pd.notna(row['height']):
                        try:
                            h_str = str(row['height']).lower().replace('m', '').strip()
                            gdf_points.at[idx, 'height'] = float(h_str)
                        except ValueError:
                            pass
                            
                    # Map Stories
                    if 'building:levels' in row and pd.notna(row['building:levels']):
                        try:
                            gdf_points.at[idx, 'numstories'] = float(row['building:levels'])
                        except ValueError:
                            pass
                            
                    # Map Roofshape
                    if 'roof:shape' in row and pd.notna(row['roof:shape']):
                        gdf_points.at[idx, 'roofshape'] = str(row['roof:shape'])
                        
                    # Map Area
                    if 'fpArea' in row and pd.notna(row['fpArea']):
                        gdf_points.at[idx, 'fpArea'] = row['fpArea']
                        
        except Exception as e:
            print(f"  -> Skipped {group_name} due to API/Data error: {e}")
            
        # Small sleep between bulk city queries, NOT per building
        time.sleep(2) 

    # Drop the geometry column so it can be saved back to a standard CSV
    return pd.DataFrame(gdf_points.drop(columns=['geometry']))

def extract_structural_attributes(lat, lon, dist=25):
    """
    Query OSM around a specific lat/lon to extract structural parameters.
    Returns height, numstories, roofshape, and fpArea (in square meters).
    """
    height = numstories = roofshape = fpArea = np.nan
    
    if pd.isna(lat) or pd.isna(lon):
        return height, numstories, roofshape, fpArea
        
    try:
        # Query OSM for buildings within a small search radius of the point
        gdf = ox.features_from_point((lat, lon), tags={'building': True}, dist=dist)
        
        # Filter for valid geometries
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        if not gdf.empty:
            # Use the first matched building
            building = gdf.iloc[0]
            
            # 1. Footprint Area
            # Project geometry to local UTM to calculate accurate area in square meters
            gdf_proj = ox.projection.project_gdf(gdf)
            fpArea = gdf_proj.iloc[0].geometry.area
            
            # 2. Roof Shape
            if 'roof:shape' in building and pd.notna(building['roof:shape']):
                roofshape = str(building['roof:shape'])
                
            # 3. Number of Stories (OSM tag: 'building:levels')
            if 'building:levels' in building and pd.notna(building['building:levels']):
                try:
                    numstories = float(building['building:levels'])
                except ValueError:
                    pass
                    
            # 4. Height
            if 'height' in building and pd.notna(building['height']):
                try:
                    # Clean strings like "10.5 m" or "10.5m"
                    h_str = str(building['height']).lower().replace('m', '').strip()
                    height = float(h_str)
                except ValueError:
                    pass
            
            # Fallback Estimate: If height is missing but stories exist, estimate ~3m per story
            if pd.isna(height) and pd.notna(numstories):
                height = numstories * 3.0
                
    except Exception:
        # Fails silently for timeouts, no data found, or network errors
        pass
        
    return height, numstories, roofshape, fpArea

# === MAIN PROCESSING LOGIC ===

def main(base_dir, building_csv_path, out_dir):
    # 1. Load the per-building CSV
    print(f"Loading building metadata from {building_csv_path}...")
    df_metadata = pd.read_csv(building_csv_path)
    
    if 'filename' not in df_metadata.columns:
        raise ValueError("The provided building CSV must contain a 'filename' column.")
        
    # 2. Gather all images from the dataset directory recursively
    print(f"Scanning for images recursively in {base_dir}...")
    base_path = Path(base_dir)
    image_paths = [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in base_path.rglob(ext)]
    
    records = []
    for img_path in image_paths:
        split = "train" if "train" in img_path.parts else ("val" if "val" in img_path.parts else "unknown")
        records.append({
            "filename": img_path.name,
            "split": split,
        })
        
    df_images = pd.DataFrame(records)
    if df_images.empty:
        print("No images found in the specified directory. Exiting.")
        return

    # 3. Merge physical image files with coordinate data FIRST
    print("Merging image data with building coordinates...")
    output_df = pd.merge(df_images, df_metadata, on="filename", how="left")
    
    # 4. Bulk Query OSM using Spatial Joins
    print("Querying OSM via Spatial Join (Batch Mode)...")
    output_df = batch_extract_osm_metadata(output_df)
    
    # Initialize the columns
    output_df['height'] = np.nan
    output_df['numstories'] = np.nan
    output_df['fpArea'] = np.nan
    output_df['roofshape'] = pd.Series(np.nan, index=output_df.index, dtype=object)

    # Use tqdm to show a progress bar
    for idx, row in tqdm(output_df.iterrows(), total=len(output_df)):
        lat = row.get('latitude')
        lon = row.get('longitude')
        
        # We only query OSM if we have valid coordinates
        if pd.notna(lat) and pd.notna(lon):
            h, s, r, a = extract_structural_attributes(lat, lon)
            
            output_df.at[idx, 'height'] = h
            output_df.at[idx, 'numstories'] = s
            output_df.at[idx, 'roofshape'] = r
            output_df.at[idx, 'fpArea'] = a
            
            # CRITICAL: Sleep to prevent Overpass API from blocking your IP
            time.sleep(1.0) 

    # 5. Apply Semantic Rules (if material_class is present)
    if 'material_class' in output_df.columns:
        amorphous_materials = ["AmorphousMembrane", "AmorphousFabric", "AmorphousConcrete", "AmorphousAsphalt"]
        # If OSM didn't find a roofshape, but it's an amorphous material, force it to Flat
        output_df.loc[
            (output_df['material_class'].isin(amorphous_materials)) & (output_df['roofshape'].isna()), 
            'roofshape'
        ] = "Flat"
        
    # 6. Finalization and Export
    output_csv_path = os.path.join(out_dir, "roof_materials_augmented_osm.csv")
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✅ Combined metadata CSV saved to: {output_csv_path}")
    print(f"Total records processed: {len(output_df):,}")
    
    # Dataset completeness statistics
    print(f"\n📊 Dataset completeness statistics:\n")
    fields_to_check = ['latitude', 'longitude', 'fpArea', 'height', 'numstories', 'roofshape']
    for field in fields_to_check:
        if field in output_df.columns:
            available = output_df[field].notna().sum()
            percent = (available / len(output_df)) * 100
            print(f"{field:<15}: {available:,} entries ({percent:.2f}%)")

# === ENTRY POINT ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata CSV for RoofNet images using OSM.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset base directory")
    parser.add_argument('--building_csv', type=str, required=True, help="Path to CSV containing filename, lat, lon")
    parser.add_argument('--out_dir', type=str, required=True, help="Desired output path for augmented CSV")
    args = parser.parse_args()

    main(args.dataset_dir, args.building_csv, args.out_dir)