"""
precompute_static.py
=======================
Run ONCE before the 400-run batch.

Covers every computation in build_exposure_model.py that is:
  • deterministic (no rand_generator calls), AND
  • identical across all 4 modes and all 100 seeds

Saves two artefacts:
  static_buildings_enriched.pkl — enriched building DataFrame with:
      lat, lon, area_m2, height_m, numstories_est, height_token,
      occupancy, height_conflict, copernicus_val

  usgs_snapped_cache.npz — pre-snapped USGS shakemap arrays ready for
      instant GMF sampling, keyed by site index

Usage:
    python precompute_static.py

Runtime: replaces ~80-90 % of the per-run wall time.
"""

import io
import os
import pickle
import time
import warnings
import json
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely import wkt
from shapely.ops import polylabel, transform
from shapely.geometry import Point, mapping
from shapely.strtree import STRtree
import pyproj
from scipy.spatial import cKDTree
from xml.etree import ElementTree as ET


try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib not found — polylabel will run single-threaded. "
                  "Install with: pip install joblib")

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths — mirror those in build_exposure_model.py ───────────────────────────

ROOFNET_CSV   = "../resources/roofnet_metadata.csv"
XBD_PATH = "Please download the full data with metadata from https://xview2.org/dataset and enter the path here"
GPKG_PATHS   = ["Please download Mexico_city_1.gpkg and Mexico_city_2.gpkg from https://zenodo.org/records/11156602 and enter the GPKG paths here"]
INDUSTRIAL_TIF = "./reference_data/Industrial_land_MEX_154_2017.tif" # from Yoo et al. (2025) 10-m Industrial lands dataset
GHSL_USE_TIF   = "Please download GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10_V1_0_R7_C9.tif from https://human-settlement.emergency.copernicus.eu/download.php?ds=builtC and enter the path here" 
GRID_XML       = "./reference_data/ground_motion_grid.xml" # derived from ShakeMap Atlas V4.0
UNCERTAINTY_XML= "./reference_data/uncertainty.xml" # derived from ShakeMap Atlas V4.0
JSON_PATTERN = f"{XBD_PATH}/*_pre_disaster.json"
CSV_PATH = "../resources/roofnet_metadata.csv"
IMPACT_CSV_PATH = "./reference_data/Impact_Buildings_Detailed.csv" # from the GEM Foundation's GEID

CACHE_DIR      = "./cache"
try:
	BUILDINGS_PKL  = os.path.join(CACHE_DIR, "static_buildings_enriched.pkl")
except FileNotFoundError:
	os.makedirs('./cache')
	BUILDINGS_PKL  = os.path.join(CACHE_DIR, "static_buildings_enriched.pkl")
	
USGS_NPZ       = os.path.join(CACHE_DIR, "usgs_snapped_cache.npz")

# Raster constants (keep in sync with build_exposure_model.py)
INDUSTRIAL_PIXEL_VALUE = 1
GHSL_RESIDENTIAL_VALUES = {1, 2, 3, 4, 11, 12, 13, 14, 15}
GHSL_HEIGHT_LIMITS = {
    11: (0, 3),  21: (0, 3),
    12: (3, 6),  22: (3, 6),
    13: (6, 15), 23: (6, 15),
    14: (15, 30), 24: (15, 30),
    15: (30, 500), 25: (30, 500),
}

os.makedirs(CACHE_DIR, exist_ok=True)

# ── CRS transformers ───────────────────────────────────────────────────────────

WGS84  = pyproj.CRS("EPSG:4326")
UTM14N = pyproj.CRS("EPSG:32614")
to_utm = pyproj.Transformer.from_crs(WGS84, UTM14N, always_xy=True).transform
to_wgs = pyproj.Transformer.from_crs(UTM14N, WGS84, always_xy=True).transform


# ─────────────────────────────────────────────────────────────────────────────
# Steps 0 - 2 — Load CSV + Merge CSV with xBD JSONs + Parallel Polylabel Centroids
# ─────────────────────────────────────────────────────────────────────────────

# --- Load all building polygons from JSONs ---
def load_building_polygons(json_glob_pattern):
    seen_uids = set()
    polygons = []
    metadata = []

    matched_files = glob.glob(json_glob_pattern)
    print(f"Files matched: {len(matched_files)}")

    for filepath in matched_files:
        with open(filepath, "r") as f:
            data = json.load(f)

        for feature in data["features"]["lng_lat"]:
            props = feature.get("properties", {})
            if props.get("feature_type") != "building":
                continue
            uid = props["uid"]
            if uid in seen_uids:
                continue
            seen_uids.add(uid)

            polygon = wkt.loads(feature["wkt"])
            polygons.append(polygon)
            metadata.append({
                "uid": uid,
                "subtype": props.get("subtype", "unknown"),
                "wkt": feature["wkt"],
                "source_file": filepath,
            })

    print(f"Polygons loaded: {len(polygons)}")
    return polygons, metadata


# --- Build STRtree spatial index ---

def build_spatial_index(polygons):
    from shapely.strtree import STRtree
    tree = STRtree(polygons)
    print("Spatial index built.")
    return tree


# --- Build merged dataframe ---

def build_merged_dataframe(
    csv_path,
    metadata,
    impact_csv_path="Impact_Buildings_Detailed.csv",
    city_col="city",
    city_filter="mexico_city_mexico",
    filename_col="filename",
    output_path="merged_roofnet_polygons.csv",
):
    # 1. Load RoofNet and Metadata
    df = pd.read_csv(csv_path)
    df = df[df[city_col] == city_filter].copy()
    df["uid"] = df[filename_col].str.replace(r"\.png$", "", regex=True).str.split("pre_disaster_").str[-1]

    meta_df = pd.DataFrame(metadata)[["uid", "subtype", "wkt"]]
    merged = df.merge(meta_df, on="uid", how="left")

    # 2. Convert to GeoDataFrames (Standard EPSG:4326 Lat/Lon)
    def parse_wkt(geom_str):
        try:
            return wkt.loads(geom_str)
        except Exception:
            return None

    merged['geometry'] = merged['wkt'].apply(parse_wkt)
    roofnet_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs="EPSG:4326")

    impact_df = pd.read_csv(impact_csv_path)
    impact_gdf = gpd.GeoDataFrame(
        impact_df, 
        geometry=gpd.points_from_xy(impact_df['LONGITUDE'], impact_df['LATITUDE']),
        crs="EPSG:4326"
    )

    # 3. Project to UTM Zone 14N (CDMX) to calculate distances in exact meters
    roofnet_utm = roofnet_gdf.dropna(subset=['geometry']).to_crs(epsg=32614).copy()
    impact_utm = impact_gdf.to_crs(epsg=32614).copy()

    # Save original point geometries, then buffer them by 10 meters
    impact_utm['point_geom'] = impact_utm.geometry
    impact_utm.geometry = impact_utm.geometry.buffer(10)
    impact_utm['pt_id'] = impact_utm.index 

    # 4. Find all intersections within the 10m buffer
    pairs = gpd.sjoin(impact_utm, roofnet_utm, how='inner', predicate='intersects')

    # 5. Bring in Polygon Geometries via Direct Lookup (Bypassing .merge())
    # pairs['index_right'] holds the exact index of the matched roofnet_utm row.
    # We use .loc to fetch the geometries and centroids directly, appending 
    # .values to strictly align them row-by-row without index mismatch issues.
    pairs['geometry_poly'] = roofnet_utm.geometry.loc[pairs['index_right']].values
    pairs['poly_centroid'] = roofnet_utm.geometry.centroid.loc[pairs['index_right']].values

    # 6. Calculate Exact Distances for ranking
    pt_gs = gpd.GeoSeries(pairs['point_geom'], crs=32614)
    poly_gs = gpd.GeoSeries(pairs['geometry_poly'], crs=32614)
    cent_gs = gpd.GeoSeries(pairs['poly_centroid'], crs=32614)

    pairs['dist_boundary'] = pt_gs.distance(poly_gs)
    pairs['dist_centroid'] = pt_gs.distance(cent_gs)
    
    # ... proceed directly into Step 6: Conflict Resolution Loop

    # 6. Conflict Resolution Loop (The "Stable Assignment" algorithm)
    final_matches = []
    
    print(f"Resolving conflicts for {len(pairs['pt_id'].unique())} candidate points within 10m...")
    
    while not pairs.empty:
        # Step A: Each point claims the polygon boundary it is closest to
        idx_min_boundary = pairs.groupby('pt_id')['dist_boundary'].idxmin()
        point_prefs = pairs.loc[idx_min_boundary]
        
        # Step B: If multiple points claimed the SAME polygon, the polygon picks 
        # the point closest to its centroid. The loser gets dropped from this round.
        idx_min_centroid = point_prefs.groupby('index_right')['dist_centroid'].idxmin()
        poly_prefs = point_prefs.loc[idx_min_centroid]
        
        # Save these finalized matches
        final_matches.append(poly_prefs)
        
        # Step C: Remove the matched points and matched polygons from the pool
        matched_pts = poly_prefs['pt_id'].unique()
        matched_polys = poly_prefs['index_right'].unique()
        
        # Any point that lost the centroid tie-breaker remains in `pairs`, 
        # but the polygon it wanted is gone. In the next loop iteration, 
        # it will automatically claim its NEXT closest building within 10m.
        pairs = pairs[~pairs['pt_id'].isin(matched_pts) & ~pairs['index_right'].isin(matched_polys)]

    # 7: Map damage states, merge, and discard unused columns ---
    final_df = merged.copy() # Create clean base without spatial geometries

    if final_matches:
        resolved_df = pd.concat(final_matches)
        
        # 1:1 mapping failsafe
        mapping = pd.DataFrame({
            'roofnet_idx': resolved_df['index_right'].values,
            'impact_idx': resolved_df['pt_id'].values
        }).drop_duplicates(subset=['impact_idx'])
        
        # Define the crosswalk from GEM Damage Levels to xBD subtypes
        # Adjust the values on the right to match your specific xBD conventions
        damage_mapping = {
            "Slight": "minor-damage",
            "Moderate": "minor-damage",  
            "Damaged": "major-damage",
            "Extensive": "major-damage",
            "Complete": "destroyed"
        }
        
        # Isolate ONLY the DAMAGE_LEVEL column so the rest of the Impact data disappears
        impact_minimal = impact_df[['DAMAGE_LEVEL']].copy()
        impact_minimal['impact_idx'] = impact_minimal.index
        
        # Translate the Impact damage strings to xBD damage strings
        impact_minimal['mapped_subtype'] = impact_minimal['DAMAGE_LEVEL'].map(damage_mapping)
        
        # Robustly merge the mapping onto our minimal impact dataframe
        impact_mapped = impact_minimal.merge(mapping, on='impact_idx', how='inner')
        impact_mapped.set_index('roofnet_idx', inplace=True)
        
        # Join ONLY the translated subtype back to the main RoofNet dataframe
        final_df = final_df.merge(
            impact_mapped[['mapped_subtype']], 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        
        # Overwrite the original xBD 'subtype' with the new Impact subtype where a match exists
        final_df['subtype'] = final_df['mapped_subtype'].fillna(final_df['subtype'])
        
        # Clean up temporary columns and unwanted geometry data
        final_df.drop(columns=['mapped_subtype', 'geometry'], inplace=True, errors='ignore')
        # Sort by damage severity , then drop duplicates keeping the first
        final_df = final_df.sort_values('subtype', ascending=True).drop_duplicates(subset=['uid'])

    # Quick metric to show how many subtypes were updated based on the spatial join
    if final_matches:
        updated_subtypes = len(impact_mapped.dropna(subset=['mapped_subtype']))
        print(f"Impact points successfully mapped and subtypes updated: {updated_subtypes}")
    else:
        print("No matches found to update subtypes.")

    # 8. Save
    final_df.to_csv(output_path, index=False)
    print(f"Merged dataframe saved to: {output_path}")

    return final_df

def _centroid_worker(wkt_str):
    """Compute pole-of-inaccessibility for a single building (joblib target)."""
    poly_wgs = wkt.loads(wkt_str)
    poly_utm = transform(to_utm, poly_wgs)
    pole_utm = polylabel(poly_utm, tolerance=0.1)
    pole_wgs = transform(to_wgs, pole_utm)
    return float(pole_wgs.y), float(pole_wgs.x), float(poly_utm.area)


def compute_centroids_parallel(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Replaces the serial polylabel loop in Step 2.
    Uses all available cores by default (n_jobs=-1).
    Falls back to serial if joblib is unavailable.
    """
    wkts = df["wkt"].tolist()
    print(f"  Computing polylabel centroids for {len(wkts)} buildings "
          f"({'parallel' if JOBLIB_AVAILABLE else 'serial'})...")
    t0 = time.perf_counter()

    if JOBLIB_AVAILABLE:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_centroid_worker)(w) for w in wkts
        )
    else:
        results = [_centroid_worker(w) for w in wkts]

    lats, lngs, areas = zip(*results)
    df = df.copy()
    df["lat"]     = list(lats)
    df["lon"]     = list(lngs)
    df["area_m2"] = list(areas)
    print(f"  Centroids done in {time.perf_counter() - t0:.1f}s")
    return df


def stories_to_height_token(n):
    return f"H{n}" if n <= 12 else "H12"


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — GeoPackage Height Enrichment (vectorized nearest-neighbour)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_height_from_gpkg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces the per-row lookup_height loop in Step 3.
    Builds the STRtree once, then uses a single bulk KD-tree query.
    """
    print("  Loading GeoPackage height data...")
    t0 = time.perf_counter()
    gpkg_frames = []
    for path in GPKG_PATHS:
        gdf = gpd.read_file(path)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        gpkg_frames.append(gdf[["geometry", "height"]].dropna(subset=["height"]).copy())

    gpkg_all     = pd.concat(gpkg_frames, ignore_index=True)
    gpkg_xy      = np.column_stack([
        gpkg_all.geometry.centroid.x.values,
        gpkg_all.geometry.centroid.y.values,
    ])
    gpkg_heights = gpkg_all["height"].values
    print(f"  GeoPackage loaded in {time.perf_counter() - t0:.1f}s "
          f"({len(gpkg_all)} points)")

    # Use cKDTree for O(n log n) bulk nearest-neighbour
    kd = cKDTree(gpkg_xy)
    building_xy = df[["lon", "lat"]].values          # (N, 2) array
    dists, idxs = kd.query(building_xy, workers=-1)  # fully parallel

    MATCH_RADIUS_DEG = 0.0001  # ~10 m at CDMX latitude

    df = df.copy()
    # Use existing height if available; otherwise take GeoPackage value
    # only when the nearest point is within the search radius
    existing = pd.to_numeric(df.get("height", pd.Series(dtype=float)), errors="coerce")

    resolved = np.where(
        (existing.notna()) & (existing > 0),
        existing.values,
        np.where(dists <= MATCH_RADIUS_DEG, gpkg_heights[idxs].astype(float), np.nan),
    )
    df["height_m"] = resolved

    df["numstories_est"] = df.apply(
        lambda r: int(r["numstories"]) if pd.notna(r.get("numstories")) and r.get("numstories", 0) > 0
                  else (max(1, round(r["height_m"] / 3.0)) if pd.notna(r["height_m"]) else 1),
        axis=1,
    )
    df["height_token"] = df["numstories_est"].apply(stories_to_height_token)
    print(f"  Height enrichment complete.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Vectorised Raster Occupancy Sampling
# ─────────────────────────────────────────────────────────────────────────────

def _load_raster(path: str):
    """Load a raster into memory and return (array, transform, crs, nodata)."""
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs, src.nodata


def _make_transformer(src_crs, dst_crs_epsg: int = None):
    """Return a pyproj Transformer or None if already in WGS-84."""
    if src_crs and src_crs.to_epsg() != 4326 and dst_crs_epsg:
        return pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    return None


def assign_occupancy_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized replacement for the per-row assign_occupancy_and_validate loop.

    For each raster we transform all building coordinates at once (one
    pyproj call), then index the pre-loaded numpy array with the resulting
    row/col integer arrays — no Python loop over buildings required.
    """
    print("  Loading rasters...")
    t0 = time.perf_counter()
    ind_arr, ind_tf, ind_crs, ind_nd   = _load_raster(INDUSTRIAL_TIF)
    ghs_arr, ghs_tf, ghs_crs, ghs_nd  = _load_raster(GHSL_USE_TIF)
    print(f"  Rasters loaded in {time.perf_counter() - t0:.1f}s")

    lons = df["lon"].values
    lats = df["lat"].values
    n    = len(df)

    # ── Industrial TIF ────────────────────────────────────────────────────────
    if ind_crs and ind_crs.to_epsg() != 4326:
        ind_tf_obj = pyproj.Transformer.from_crs("EPSG:4326", ind_crs, always_xy=True)
        xs_ind, ys_ind = ind_tf_obj.transform(lons, lats)
    else:
        xs_ind, ys_ind = lons, lats

    rows_ind, cols_ind = rowcol(ind_tf, xs_ind, ys_ind)
    rows_ind = np.asarray(rows_ind)
    cols_ind = np.asarray(cols_ind)

    # Clip to valid array bounds; mark out-of-bounds as -1
    valid_ind = (
        (rows_ind >= 0) & (rows_ind < ind_arr.shape[0]) &
        (cols_ind >= 0) & (cols_ind < ind_arr.shape[1])
    )
    ind_vals = np.full(n, -1, dtype=float)
    ind_vals[valid_ind] = ind_arr[rows_ind[valid_ind], cols_ind[valid_ind]].astype(float)
    if ind_nd is not None:
        ind_vals[ind_vals == float(ind_nd)] = -1.0
    is_industrial = ind_vals == float(INDUSTRIAL_PIXEL_VALUE)

    # ── GHSL TIF ──────────────────────────────────────────────────────────────
    if ghs_crs and ghs_crs.to_epsg() != 4326:
        ghs_tf_obj = pyproj.Transformer.from_crs("EPSG:4326", ghs_crs, always_xy=True)
        xs_ghs, ys_ghs = ghs_tf_obj.transform(lons, lats)
    else:
        xs_ghs, ys_ghs = lons, lats

    rows_ghs, cols_ghs = rowcol(ghs_tf, xs_ghs, ys_ghs)
    rows_ghs = np.asarray(rows_ghs)
    cols_ghs = np.asarray(cols_ghs)

    valid_ghs = (
        (rows_ghs >= 0) & (rows_ghs < ghs_arr.shape[0]) &
        (cols_ghs >= 0) & (cols_ghs < ghs_arr.shape[1])
    )
    ghs_vals = np.full(n, -1, dtype=int)
    ghs_vals[valid_ghs] = ghs_arr[rows_ghs[valid_ghs], cols_ghs[valid_ghs]].astype(int)
    if ghs_nd is not None:
        ghs_vals[ghs_vals == int(ghs_nd)] = -1

    # ── Occupancy logic ───────────────────────────────────────────────────────
    # Vectorised: residential check takes precedence over industrial
    is_residential = np.isin(ghs_vals, list(GHSL_RESIDENTIAL_VALUES))
    occupancy = np.where(
        is_residential, "Res",
        np.where(is_industrial, "Ind", "Com")
    )

    # ── Height validation ─────────────────────────────────────────────────────
    gpkg_h = df["height_m"].values
    has_height = ~np.isnan(gpkg_h.astype(float))

    h_mins = np.array([GHSL_HEIGHT_LIMITS.get(int(v), (0, 0))[0] for v in ghs_vals])
    h_maxs = np.array([GHSL_HEIGHT_LIMITS.get(int(v), (0, 0))[1] for v in ghs_vals])
    has_limits = np.isin(ghs_vals, list(GHSL_HEIGHT_LIMITS.keys()))
    conflict = has_height & has_limits & (
        (gpkg_h.astype(float) < h_mins) | (gpkg_h.astype(float) > h_maxs)
    )

    df = df.copy()
    df["occupancy"]       = occupancy
    df["height_conflict"] = conflict
    df["copernicus_val"]  = ghs_vals

    # Print summary to keep parity with original output
    occ_counts = pd.Series(occupancy).value_counts().to_dict()
    total_conf = int(conflict.sum())
    print(f"\n--- Height Validation Summary ---")
    print(f"Total Buildings Analyzed: {n}")
    print(f"GPKG vs. Copernicus Conflicts: {total_conf} ({total_conf/n*100:.2f}%)")
    print(f"Occupancy distribution: {occ_counts}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 (partial) — Pre-snap USGS Shakemap to building sites
# ─────────────────────────────────────────────────────────────────────────────

def _parse_shakemap(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {'ns': 'http://earthquake.usgs.gov/eqcenter/shakemap'}
    fields = {f.get('name'): int(f.get('index')) - 1
              for f in root.findall('ns:grid_field', ns)}
    data_text = root.find('ns:grid_data', ns).text.strip()
    import io as _io
    df = pd.read_csv(_io.StringIO(data_text), sep=r'\s+', header=None)
    return df, fields


def precompute_usgs_snapping(site_df: pd.DataFrame) -> dict:
    """
    Parse USGS mean + uncertainty XMLs once, snap all building sites to the
    nearest shakemap grid point, and pre-extract the per-site arrays needed
    for GMF sampling. This dict is passed directly into create_stochastic_gmf_csv_fast.
    """
    print("  Parsing USGS shakemap XMLs...")
    df_mean, map_mean = _parse_shakemap(GRID_XML)
    df_unc,  map_unc  = _parse_shakemap(UNCERTAINTY_XML)

    usgs_coords  = df_mean[[map_mean['LON'], map_mean['LAT']]].values
    site_coords  = site_df[['lon', 'lat']].values
    kd           = cKDTree(usgs_coords)
    _, indices   = kd.query(site_coords)

    # Pre-extract per-site mean and sigma arrays
    n_sites = len(site_df)
    m_pga = df_mean.iloc[indices, map_mean['PGA']].values   / 100.0
    m_03  = df_mean.iloc[indices, map_mean['PSA03']].values / 100.0
    m_10  = df_mean.iloc[indices, map_mean['PSA10']].values / 100.0

    s_pga = df_unc.iloc[indices, map_unc['STDPGA']].values
    s_03  = df_unc.iloc[indices, map_unc['STDPSA03']].values
    s_10  = df_unc.iloc[indices, map_unc['STDPSA10']].values

    # Interpolate SA(0.6) in log-space — vectorised over sites
    log_t  = np.log([0.3, 1.0])
    alpha  = (np.log(0.6) - log_t[0]) / (log_t[1] - log_t[0])   # scalar
    m_06   = np.exp((1.0 - alpha) * np.log(m_pga)  # intentionally using log(m_03) below
                    )  # placeholder — corrected:
    m_06   = np.exp((1.0 - alpha) * np.log(m_03) + alpha * np.log(m_10))
    s_06   = (1.0 - alpha) * s_03 + alpha * s_10

    return {
        "m_pga": m_pga, "m_03": m_03, "m_10": m_10, "m_06": m_06,
        "s_pga": s_pga, "s_03": s_03, "s_10": s_10, "s_06": s_06,
        "n_sites": n_sites,
        "indices": indices,    # kept for reference / debugging
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_total = time.perf_counter()

    # ── Step 0: Merge xBD + GEM + RoofNet ────────────────────────────────────────────────
    
    polygons, metadata = load_building_polygons(JSON_PATTERN)
    tree = build_spatial_index(polygons)

    merged_df = build_merged_dataframe(
        csv_path=CSV_PATH,
        metadata=metadata,
        impact_csv_path = IMPACT_CSV_PATH,
        city_filter="mexico_city_mexico",
        output_path="./reference_data/merged_roofnet_polygons.csv",
    )
    
    # ── Step 1: Load CSV ──────────────────────────────────────────────────────
    
    print("\n[Step 1] Loading roofnet_metadata.csv...")
    df = pd.read_csv(ROOFNET_CSV)
    df = df.dropna(subset=["wkt"]).copy()
    print(f"  {len(df)} buildings with matched polygons")

    # ── Step 2: Parallel polylabel centroids ─────────────────────────────────
    print("\n[Step 2] Computing polylabel centroids...")
    df = compute_centroids_parallel(df, n_jobs=-1)

    # ── Step 3: GeoPackage height enrichment ─────────────────────────────────
    print("\n[Step 3] Enriching heights from GeoPackage...")
    df = enrich_height_from_gpkg(df)

    # ── Step 4: Vectorised occupancy assignment ───────────────────────────────
    print("\n[Step 4] Assigning occupancy (vectorised raster sampling)...")
    df = assign_occupancy_vectorized(df)

    # ── Save building cache ───────────────────────────────────────────────────
    print(f"\nSaving building cache → {BUILDINGS_PKL}")
    with open(BUILDINGS_PKL, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {len(df)} buildings, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB on disk")

    # ── Step 9 pre-snap: unique site coordinates ──────────────────────────────
    print("\n[Step 9-prep] Pre-snapping USGS shakemap to building sites...")
    # Site coordinates are the same for all modes and seeds
    site_df = df[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
    usgs_cache = precompute_usgs_snapping(site_df)
    usgs_cache["site_df"] = site_df      # store alongside for completeness

    np.savez_compressed(
        USGS_NPZ,
        **{k: v for k, v in usgs_cache.items() if k != "site_df"},
    )
    # site_df is a DataFrame — save separately inside the pickle
    usgs_cache_pkl = os.path.join(CACHE_DIR, "usgs_snapped_cache.pkl")
    with open(usgs_cache_pkl, "wb") as f:
        pickle.dump(usgs_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  USGS cache saved → {usgs_cache_pkl}")

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*50}")
    print(f"Static precomputation complete in {elapsed:.1f}s")
    print(f"Cache files:")
    print(f"  {BUILDINGS_PKL}")
    print(f"  {usgs_cache_pkl}")
    print(f"\nYou can now run run_batch_experiments.py")


if __name__ == "__main__":
    main()
