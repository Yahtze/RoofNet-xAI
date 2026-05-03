# Description:
# This script automates the re-downloading of satellite imagery for the RoofNet dataset
# using the Google Maps Static API. It processes a queue of coordinates from a CSV file,
# handles HTTP request retries with exponential backoff, and logs failed downloads 
# for later troubleshooting.

import os
import csv
import time
import requests
import pandas as pd

# === CONFIGURATION AND API SETUP ===

# Load Google Maps Static API key from a local text file
API_KEY = "" # <-- Insert your API key generated from https://developers.google.com/maps/documentation/maps-static/get-api-key

# Input CSV containing the target filenames and their corresponding geographic coordinates
CSV_FILE = "" # <-- Insert your CSV file path containing building coordinates 

# Output directory where the downloaded satellite images will be stored
OUT_DIR = "roofnet_redownloaded" 

# CSV file to record metadata for any downloads that fail after multiple attempts
FAILED_CSV = "failed_downloads.csv" 

# Ensure the output directory exists on the filesystem
os.makedirs(OUT_DIR, exist_ok=True) 

# Load the download queue into a pandas DataFrame
df = pd.read_csv(CSV_FILE) 

# Ensure latitude and longitude columns are numeric; non-numeric values are set to NaN
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Initialize a persistent requests session to improve performance over multiple requests
session = requests.Session()

# === HELPER FUNCTIONS ===

def get_image(lat, lon):
    """
    Constructs and sends a request to the Google Static Maps API for a satellite 
    image centered at the provided coordinates.
    
    Parameters:
        lat, lon: Geographic coordinates for the center of the image.
    """
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}"
        "&zoom=20"            # Set zoom level to 20 for high-resolution building views
        "&size=256x256"       # Request 256x256 pixel tiles
        "&maptype=satellite"  # Use satellite imagery layer
        f"&key={API_KEY}"
    )
    return session.get(url, timeout=30)

# === MAIN DOWNLOAD LOOP ===

failed_rows = [] # List to store metadata for rows that failed validation or download

for i, row in df.iterrows():
    filename = row["filename"]
    lat = row["latitude"]
    lon = row["longitude"]

    # Define the full destination path for the current image
    out_path = os.path.join(OUT_DIR, filename) 

    # Skip download if the file already exists locally (resumable download logic)
    if os.path.exists(out_path):
        continue

    # Skip download if coordinates are missing or invalid
    if pd.isna(lat) or pd.isna(lon): 
        print(f"Bad coordinates: {filename} | lat={lat} lon={lon}")
        failed_rows.append({
            "filename": filename,
            "latitude": lat,
            "longitude": lon,
            "reason": "bad_coordinates"
        })
        continue

    success = False # Flag to track download status across retry attempts

    # Attempt to download the image up to 5 times in case of transient network errors
    for attempt in range(1, 6): 
        try:
            response = get_image(lat, lon)

            # Inspect headers to ensure the response contains actual image data
            content_type = response.headers.get("Content-Type", "")
            body_preview = response.text[:200] if "text" in content_type.lower() else ""

            # Check for HTTP 200 (OK) and a valid image MIME type
            if response.status_code == 200 and "image" in content_type.lower(): 
                with open(out_path, "wb") as f:
                    f.write(response.content)
                success = True
                break
            
            # Log failure details for the current attempt
            print(
                f"Attempt {attempt}/5 failed for {filename} | "
                f"status={response.status_code} content_type={content_type} "
                f"body={body_preview}"
            )

        except Exception as e:
            # Handle connection errors or timeouts
            print(f"Attempt {attempt}/5 error for {filename}: {e}")

        # Wait before retrying; sleep duration increases with each attempt (exponential backoff)
        time.sleep(2 * attempt) 

    # If all 5 attempts fail, log the record as a persistent failure
    if not success:
        failed_rows.append({
            "filename": filename,
            "latitude": lat,
            "longitude": lon,
            "reason": "request_failed"
        })

    # Display processing progress every 100 images
    if i % 100 == 0: 
        print(f"Processed {i}/{len(df)}")

    # Short delay to prevent overwhelming the API or exceeding rate limits
    time.sleep(0.2)

# === FINALIZATION ===

# Export all failed records to a CSV file for manual review or re-processing
if failed_rows: 
    pd.DataFrame(failed_rows).to_csv(FAILED_CSV, index=False)
    print(f"Saved failed rows to {FAILED_CSV}")
else:
    print("No failed rows.")
