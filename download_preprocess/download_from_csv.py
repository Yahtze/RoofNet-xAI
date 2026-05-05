"""
Description:
This script automates the re-downloading of satellite imagery for the RoofNet dataset
using the Google Maps Static API. It processes a queue of coordinates from a CSV file,
handles HTTP request retries with exponential backoff, and logs failed downloads 
for later troubleshooting. If you have already downloaded images in tandem with
sampling polygons, as shown in sample_osm_polygons_gsat_imagery.ipynb, this script
will be largely redundant. It is meant for use in the case of having CSV information
only (i.e. no imagery).

Call via: 
    python download_from_csv.py --csv_file <path_to_csv> --keys_file <path_to_keys.json> [--out_dir <output_dir>]
"""

import os
import time
import json
import requests
import argparse
import pandas as pd

def main(csv_file, keys_file, out_dir, failed_csv):
    # === CONFIGURATION AND API SETUP ===
    
    # Extract the API key from the provided JSON file
    with open(keys_file, 'r') as f:
        keys_data = json.load(f)
        API_KEY = keys_data.get("google_static_maps_api_key")
        if not API_KEY:
            raise ValueError("Could not find 'google_static_maps_api_key' in the provided JSON file.")

    os.makedirs(out_dir, exist_ok=True) 

    df = pd.read_csv(csv_file) 
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    session = requests.Session()

    def get_image(lat, lon):
        url = (
            "https://maps.googleapis.com/maps/api/staticmap"
            f"?center={lat},{lon}"
            "&zoom=20"
            "&size=256x256"
            "&maptype=satellite"
            f"&key={API_KEY}"
        )
        return session.get(url, timeout=30)

    # === MAIN DOWNLOAD LOOP ===
    failed_rows = [] 

    for i, row in df.iterrows():
        filename = row["filename"]
        lat = row["latitude"]
        lon = row["longitude"]

        out_path = os.path.join(out_dir, filename) 

        if os.path.exists(out_path):
            continue

        if pd.isna(lat) or pd.isna(lon): 
            print(f"Bad coordinates: {filename} | lat={lat} lon={lon}")
            failed_rows.append({
                "filename": filename, "latitude": lat, "longitude": lon, "reason": "bad_coordinates"
            })
            continue

        success = False 
        for attempt in range(1, 6): 
            try:
                response = get_image(lat, lon)
                content_type = response.headers.get("Content-Type", "")
                body_preview = response.text[:200] if "text" in content_type.lower() else ""

                if response.status_code == 200 and "image" in content_type.lower(): 
                    with open(out_path, "wb") as f:
                        f.write(response.content)
                    success = True
                    break
                
                print(f"Attempt {attempt}/5 failed for {filename} | status={response.status_code}")

            except Exception as e:
                print(f"Attempt {attempt}/5 error for {filename}: {e}")

            time.sleep(2 * attempt) 

        if not success:
            failed_rows.append({
                "filename": filename, "latitude": lat, "longitude": lon, "reason": "request_failed"
            })

        if i % 10 == 0: 
            print(f"Processed {i}/{len(df)}")
        time.sleep(0.2)

    # === FINALIZATION ===
    if failed_rows: 
        pd.DataFrame(failed_rows).to_csv(failed_csv, index=False)
        print(f"Saved failed rows to {failed_csv}")
    else:
        print("No failed rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download satellite imagery from Google Static Maps API.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing building coordinates")
    parser.add_argument('--keys_file', type=str, default='../resources/keys.json', help="Path to the keys.json file containing the API key")
    parser.add_argument('--out_dir', type=str, default="roofnet_gsat_imagery", help="Output directory for satellite images")
    parser.add_argument('--failed_csv', type=str, default="failed_downloads.csv", help="Output CSV for failed downloads")
    
    args = parser.parse_args()
    main(args.csv_file, args.keys_file, args.out_dir, args.failed_csv)