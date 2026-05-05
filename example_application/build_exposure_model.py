"""
build_exposure_model.py
=======================
Run modes:
    python build_exposure_model.py roofnet    ← uses RoofNet roof material
    python build_exposure_model.py benchmark  ← ignores roof material entirely

OBJECTIVE: Preprocess building data containing damage state to produce roof taxonomy 
designation, alongside preparing other files necessary for simulation.
"""

import os
import sys
import io
import glob
import re
import time
import psutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely import wkt as shapely_wkt
from shapely.ops import polylabel, transform
from shapely.geometry import Point
from shapely.strtree import STRtree
import pyproj
from xml.etree import ElementTree as ET
from xml.dom import minidom
from scipy.spatial import cKDTree
import pickle

# ── Static cache paths (written once by precompute_static.py) ────────────────
_CACHE_DIR      = "./cache"
_BUILDINGS_PKL  = os.path.join(_CACHE_DIR, "static_buildings_enriched.pkl")
_USGS_CACHE_PKL = os.path.join(_CACHE_DIR, "usgs_snapped_cache.pkl")

# Fragility CSV in-memory cache — populated once per process, shared across runs
_FRAGILITY_PARSE_CACHE: dict = {}

# Tracking compute usage for reporting purposes.
parent_process = psutil.Process(os.getpid())
start_wall_time = time.perf_counter()

# Capture starting CPU times for parent and all existing children
def get_total_cpu_time(proc):
    try:
        t = proc.cpu_times()
        total = t.user + t.system
        for child in proc.children(recursive=True):
            ct = child.cpu_times()
            total += (ct.user + ct.system)
        return total
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

start_cpu_total = get_total_cpu_time(parent_process)

VALID_MODES = {"roofnet", "unreinforced", "reinforced", "benchmark"}
MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "roofnet"
if MODE not in VALID_MODES:
    raise ValueError(f"Mode must be one of {VALID_MODES}, got '{MODE}'")
print(f"Running in {MODE.upper()} mode")

# Set random seed for entire simulation
RANDOM_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
rand_generator = np.random.default_rng(seed=RANDOM_SEED)


# ── Configuration ─────────────────────────────────────────────────────────────

OUT_DIR  = f"./{MODE}/seed_{RANDOM_SEED}/"
OUT_XML  = os.path.join(OUT_DIR, "exposure_metadata.xml")
OUT_CSV  = os.path.join(OUT_DIR, "exposure_assets.csv")
OUT_TAXMAP = os.path.join(OUT_DIR, "taxonomy_mapping.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# GEM CDMX unit replacement costs (USD/m², 2021, from Yepes-Estrada et al. 2023)
UNIT_COST_USD_PER_SQM = {"Res": 350, "Com": 600, "Ind": 450}
DEFAULT_UNIT_COST = 400


# ── Taxonomy string utilities ─────────────────────────────────────────────────

def get_height_token(taxonomy_string):
    """Extracts the height token from a full taxonomy string."""
    m = re.search(r'_(H[^/]+)', taxonomy_string)
    return m.group(1) if m else None


def stories_to_height_token(n):
    """
    Maps story count to the height tokens present in the vulnerability XML.
    """
    if n <= 12:
        return f"H{n}"
    else:
        return f"H12"


# --- TAXONOMY DESIGNATION FUNCTIONS BASED ON ASSIGNMENT MODE ---

# Mapping RoofNet detection classes to IPUMS ROOF_CODEs
# There are no clean 1-to-1 connections, so some best guesses used
ROOF_CLASS_TO_CODE = {
    "AmorphousConcrete": 10,
    "ClayTiles": 10,
    "MetalSheetMaterials": 34,
    "AmorphousAsphalt": 23,
    "AmorphousMembrane": 99,
    "AmorphousFabric": 99,
    "GreenVegetative": 45,
    "Unknown": 99 # asbestos roofing cover classified as unknown
}

# Define which codes represent flexible (ductile) diaphragms
# See "Performance of the built environment in Mexico City during the September 19, 2017 Earthquake"
# for further explanation of MUR roof code assignments.
FLEXIBLE_ROOF_CODES = [34, 45, 72] 
MUR_ROOF_CODES = [21, 34, 40, 41, 45, 71, 72] 

# --- ASSIGNMENT MODES ---
def assign_benchmark(occupancy, stories):
    """
    MODE: Benchmark.
    Height-Naive Exposure Sampling: Samples material/LLRS based purely on 
    the overall occupancy distribution, then forces the correct height.
    """
    height_token = stories_to_height_token(stories)
    gem_df = gem_exposure_data[occupancy]
    
    # Sample probabilistically from the ENTIRE occupancy dataset (ignoring height)
    subset = gem_df
    if subset.empty:
        return f"UNK_{height_token}/{occupancy.upper()}"
        
    probs = subset['BUILDINGS'] / subset['BUILDINGS'].sum()
    sampled_tax = rand_generator.choice(subset['TAXONOMY'], p=probs)
    
    # Stamp the actual estimated height onto the sampled taxonomy
    parts = sampled_tax.split('/')
    if len(parts) >= 3:
        parts[2] = height_token
        return "/".join(parts)
    return sampled_tax


def assign_roofnet(occupancy, stories, roof_code, reinforcement=None):
    """
    MODE: RoofNet.
    Height-Naive Exposure Sampling: Uses census and roof proxies to filter 
    materials/LLRS, samples ignoring height, then forces the correct height.
    """
    height_token = stories_to_height_token(stories)
    gem_df = gem_exposure_data[occupancy]
    is_flexible = roof_code in FLEXIBLE_ROOF_CODES
    typically_unreinforced = roof_code in MUR_ROOF_CODES


    if occupancy == 'Res':
        # Residential: Census -> Material
        try:
            wall_dist = res_p_wall_given_roof.loc[roof_code]
            selected_wall_code = rand_generator.choice(wall_dist.index.get_level_values('WALL_CODE'), p=wall_dist.values)
        except KeyError:
            selected_wall_code = 501 
                
        if selected_wall_code in [501, 527] and reinforcement == "unreinforced":
            mat_filter = 'MUR' # assumes all masonry buildings unreinforced
        if selected_wall_code in [501, 527] and reinforcement == "reinforced":
            mat_filter = 'MCF|MR|CR' # assumes all masonry buildings reinforced
        elif selected_wall_code in [501, 527] and reinforcement is None: # i.e. normal RoofNet run
            mat_filter = 'MUR' if typically_unreinforced else 'MCF|MR|CR'
        elif 200 <= selected_wall_code <= 299:
            mat_filter = 'INF'
        elif selected_wall_code == 300:
            mat_filter = 'W'
        else:
            mat_filter = ''
            
    else:
        # Assumes prevalence of RWFD buildings in commercial, industrial context
        # See Koliou et al. (2015) and Tena-Colunga (2020) for rationale for this split
        # (RWFD commercial/industrial buildings prevalent in N.A.)
        # (Concrete structures with steel roof trusses are commmon in CDMX)
        llrs_filter = 'MCF|MR|CR' if is_flexible else 'MUR'
        mat_filter = llrs_filter

    # Filter GEM by Material/Ductility constraints (NO height filter applied)
    candidates = gem_df[gem_df['TAXONOMY'].str.contains(mat_filter)]
    
    if candidates.empty:
        # If the specific roof proxy filter is too restrictive, fall back to benchmark
        return assign_benchmark(occupancy, stories)

    probs = candidates['BUILDINGS'] / candidates['BUILDINGS'].sum()
    sampled_tax = rand_generator.choice(candidates['TAXONOMY'], p=probs)
    
    # Stamp the actual estimated height onto the sampled taxonomy
    parts = sampled_tax.split('/')
    if len(parts) >= 3:
        parts[2] = height_token
        return "/".join(parts)    
    return sampled_tax


# ── Step 1: Load merged CSV ───────────────────────────────────────────────────

# ── Steps 1–4: Load pre-computed static cache ─────────────────────────────────
# (centroids, height enrichment, raster occupancy all pre-computed by
#  precompute_static.py — loads in <1 s instead of several minutes)

if not os.path.exists(_BUILDINGS_PKL):
    raise FileNotFoundError(
        f"Static cache not found: {_BUILDINGS_PKL}\n"
        "Run precompute_static.py first."
    )

print("Loading static building cache (pre-computed)...")
_t_cache = time.perf_counter()
with open(_BUILDINGS_PKL, "rb") as _f:
    df = pickle.load(_f)
print(f"  {len(df)} buildings loaded in {time.perf_counter() - _t_cache:.2f}s")



# ── Step 5: Assign taxonomy weights per building ──────────────────────────────
#
# This step calls the probabilistic assignment functions based on the selected MODE.
# Occupancy is sourced from Step 4, and height is handled via numstories_est.

print(f"Assigning taxonomies in ({MODE} mode)...")

# Paths and Constants
GEM_DIR = '..resources/Exposure_Data'
CENSUS_PATH = 'In accordance with the IPUMS Terms of Use, we cannot share the data for this path. Interested readers are encouraged to contact the authors for further information on what data to request from IPUMS, should they have a valid request.'

# --- INITIALIZATION ---
gem_exposure_data = {}
occupancy_map = {'Res': 'Exposure_Res_Mexico_Adm1.csv', 
                 'Ind': 'Exposure_Ind_Mexico_Adm1.csv', 
                 'Com': 'Exposure_Com_Mexico_Adm1.csv'}

for occ, filename in occupancy_map.items():
    gem_exposure_df = pd.read_csv(os.path.join(GEM_DIR, filename))
    # Filter for Mexico City and keep only relevant columns
    cdmx_gem = gem_exposure_df[gem_exposure_df['NAME_1'] == 'Ciudad de México'][['TAXONOMY', 'BUILDINGS']].copy()
    gem_exposure_data[occ] = cdmx_gem

# Load Residential Census Data for P(Wall | Roof)
census_df = pd.read_parquet(CENSUS_PATH)
# P(WALL_CODE | ROOF_CODE) distribution
res_p_wall_given_roof = census_df.groupby(['ROOF_CODE', 'WALL_CODE'])['HHWT'].sum()
res_p_wall_given_roof = res_p_wall_given_roof.groupby(level=0).apply(lambda x: x / x.sum())

full_taxonomies = []

for idx, row in df.iterrows():
    occupancy = row["occupancy"]       
    stories   = row["numstories_est"]  

    if MODE == "benchmark":
        assigned_tax = assign_benchmark(occupancy, stories)
    elif MODE in ["roofnet", "reinforced", "unreinforced"]:
        roof_mat_str = str(row.get("material_class", ""))
        roof_code = ROOF_CLASS_TO_CODE.get(roof_mat_str) 
        if MODE == "roofnet":
            assigned_tax = assign_roofnet(occupancy, stories, roof_code)
        elif MODE == "unreinforced":
            assigned_tax = assign_roofnet(occupancy, stories, roof_code, MODE)
        else: # MODE == "reinforced"
            assigned_tax = assign_roofnet(occupancy, stories, roof_code, MODE)
    else:
        raise ValueError(f"Invalid VALID_MODE: {MODE}. Must be 'benchmark' or 'roofnet'.")

    full_taxonomies.append(assigned_tax)

# Update the dataframe with the results
df["full_taxonomy"] = full_taxonomies

print(f"Taxonomy assignment complete for {len(df)} buildings.")


# ── Step 6: Write exposure assets CSV ────────────────────────────────────────
print("Writing exposure_assets.csv...")
assets = pd.DataFrame({
    "id"          : df["uid"],
    "lon"         : df["lon"].round(7),
    "lat"         : df["lat"].round(7),
    "taxonomy"    : df["full_taxonomy"],
    "number"      : 1,
    "occupancy"   : df["occupancy"],
    "subtype"     : df["subtype"],
    "roof_material": df["material_class"].fillna("unknown") if MODE == "roofnet"
                     else "not_used",
    "numstories"  : df["numstories_est"],
    "height_m"    : df["height_m"].fillna(-1).round(2),
    "height_source": df.apply(
        lambda r: "gpkg" if pd.notna(r.get("height_m")) and r.get("height_m", 0) > 0
                  else ("csv" if pd.notna(r.get("height")) and r.get("height", 0) > 0
                        else "estimated"),
        axis=1
    ),
})


# Drop unnecessary columns and add necessary dummy columns
assets.drop(['occupancy','subtype','roof_material','numstories','height_m','height_source'],
                  axis=1,inplace=True)
assets['value-area'] = 1
assets['value-structural'] = 1
assets.rename(columns={'number': 'value-number'},inplace=True)
assets.to_csv(OUT_CSV, index=False)
print(f"  Saved {len(assets)} assets → {OUT_CSV}")



# # ── Step 7: Write exposure metadata XML ──────────────────────────────────────

print("Writing exposure_metadata.xml...")
nrml = ET.Element("nrml", {
    "xmlns:gml": "http://www.opengis.net/gml",
    "xmlns"    : "http://openquake.org/xmlns/nrml/0.5",
})
em = ET.SubElement(nrml, "exposureModel", {
    "id"            : f"roofnet_cdmx_2017_{MODE}",
    "category"      : "buildings",
    "taxonomySource": "GEM_Building_Taxonomy_3.3",
})
ET.SubElement(em, "description").text = (
    f"Building-level exposure model for CDMX (2017 earthquake), {MODE} mode. "
    f"Taxonomy: GEM v3.3 with height tokens. "
    f"Occupancy: industrial land from MEX_154_2017 raster"
    + ("; commercial from metal roof heuristic (WHE)." if MODE == "roofnet"
       else "; all non-industrial assigned residential.")
)

# No conversions section — costs not used in scenario damage calculation
# ET.SubElement(em, "tagNames").text = "occupancy subtype roof_material"
ET.SubElement(em, "assets").text   = "exposure_assets.csv"

fields = ET.SubElement(em, "exposureFields")
for oq, col in [
    ("id",       "id"),
    ("lon",      "lon"),
    ("lat",      "lat"),
    ("taxonomy", "taxonomy"),
    ("number",   "value-number"),
]:
    ET.SubElement(fields, "field", {"oq": oq, "input": col})

xml_body = minidom.parseString(
    ET.tostring(nrml, encoding="unicode")
).toprettyxml(indent="  ")
with open(OUT_XML, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write("\n".join(xml_body.split("\n")[1:]))
print(f"  Saved → {OUT_XML}")



# ── Step 8: Write fragility curves to XML ──────────────────────────────────────────────────────
#
# Reads fragility CSVs from the global_fragility_vulnerability repository,
# matches them to the taxonomy strings present in the exposure model,
# and writes a single consolidated fragility XML in OpenQuake NRML format.
#
# CSV filename convention in the repository:
#   {MATERIAL}_{LLRS}-{DUCTILITY}_{HEIGHT}.csv
# e.g. CR_LDUAL-DUH_H1.csv
#
# The taxonomy strings in the exposure model use the format:
#   CR_LDUAL+CDH+DUH_H1/RES
# so we strip the code level (+CDH, +CDM) and occupancy suffix for matching.


FRAGILITY_DIR = "./resources/global_fragility_vulnerability-master/fragility_curves/fragility_other_IMs/"
OUT_FRAGILITY_XML = os.path.join(OUT_DIR, "structural_fragility_model.xml")


def find_fragility_csv(taxonomy_string, fragility_dir, prevalent_map=None):
    """
    Finds the best-matching fragility CSV with specific logic for UNK and Wood.
    """
    # --- STEP 1: PRE-PROCESSING ---
    
    # A. Resolve UNK to most prevalent type for that occupancy
    if "UNK" in taxonomy_string:
        occ_part = taxonomy_string.split('/')[-1].upper()
        if prevalent_map and occ_part in prevalent_map:
            taxonomy_string = prevalent_map[occ_part]
        else:
            # Fallback if no map: common CDMX masonry
            taxonomy_string = f"MCF/LWAL+DUM/H1/{occ_part}"

    # B. Wood (W) Relaxation
    # Drop WS and WLI, and relax LPB to LFM
    taxonomy_string = taxonomy_string.replace("+WS", "").replace("+WLI", "").replace("+WBB","")
    taxonomy_string = taxonomy_string.replace("LPB", "LFM")

    # C. Handle Height Range Tokens (e.g., HBET1-2 -> H1)
    # This ensures the digit-based relaxation logic below can read the height.
    taxonomy_string = re.sub(r'HBET(\d+)-\d+', r'H\1', taxonomy_string)

    # --- STEP 2: STANDARDIZATION & MATCHING ---

    # Clean separators for filename matching (MAT_LLRS-DUCT_HEIGHT.csv)
    clean_tax = taxonomy_string.replace('+', '-').replace(':', '')
    
    parts = clean_tax.split('/')
    if len(parts) < 3:
        return None, None
    
    mat = parts[0]
    llrs_duct = parts[1]
    height = parts[2]
    # occ = parts[3] # Not usually in filenames

    # Construct primary search stem: MAT_LLRS-DUCT_HEIGHT
    stem = f"{mat}_{llrs_duct}_{height}"

    # 1. Exact Match
    exact_path = os.path.join(fragility_dir, f"{stem}.csv")
    if os.path.exists(exact_path):
        return exact_path, "exact"

    # 2. Relax Height
    height_pattern = os.path.join(fragility_dir, f"{mat}_{llrs_duct}_H*.csv")
    available_h = glob.glob(height_pattern)
    if available_h:
        target_match = re.search(r'\d+', height)
        if target_match:
            target_val = int(target_match.group())
            def h_dist(fp):
                m = re.search(r'_H(\d+)\.csv$', fp)
                return abs(int(m.group(1)) - target_val) if m else 999
            return min(available_h, key=h_dist), "height_relaxed"

    # 3. Relax Ductility
    llrs_only = llrs_duct.split('-')[0]
    duct_pattern = os.path.join(fragility_dir, f"{mat}_{llrs_only}-*_*.csv")
    available_d = glob.glob(duct_pattern)
    if available_d:
        # Prioritize matching height within the relaxed ductility set
        for fp in available_d:
            if f"_{height}.csv" in fp:
                return fp, "ductility_relaxed"
        return available_d[0], "ductility_relaxed"

    # 4. Relax Material Prefix
    mat_pattern = os.path.join(fragility_dir, f"{mat}_*.csv")
    available_m = glob.glob(mat_pattern)
    if available_m:
        return available_m[0], "material_relaxed"

    # 5. Relax Composite Material (e.g., 'CR-PC' -> 'CR')
    if '-' in mat:
        base_mat = mat.split('-')[0]
        
        # Try to find a match for the base material with the same base LLRS
        base_mat_llrs_pattern = os.path.join(fragility_dir, f"{base_mat}_{llrs_only}*.csv")
        available_base_llrs = glob.glob(base_mat_llrs_pattern)
        
        if available_base_llrs:
            # Prioritize matching the exact height if possible
            for fp in available_base_llrs:
                if f"_{height}.csv" in fp:
                    return fp, "base_material_relaxed"
            return available_base_llrs[0], "base_material_relaxed"
            
        # Ultimate fallback: Any curve for the base material
        base_mat_any_pattern = os.path.join(fragility_dir, f"{base_mat}_*.csv")
        available_base_any = glob.glob(base_mat_any_pattern)
        if available_base_any:
            return available_base_any[0], "base_material_relaxed"

    return None, None


def parse_fragility_csv(filepath):
    """
    Parses a fragility CSV into IML levels and POE arrays per limit state.
    Returns (imt, iml_list, poe_dict) where poe_dict = {ls_name: [poe values]}.
    """
    df_frag = pd.read_csv(filepath, sep=',', header=0)
    df_frag.columns = df_frag.columns.str.strip()

    iml_col = df_frag.columns[0]
    ls_cols = list(df_frag.columns[1:])

    # Ensure the values are treated as floats immediately
    imt = "PGA" if "PGA" in iml_col.upper() else iml_col.split()[0]
    
    # Force conversion to numeric to catch string errors early
    iml_values = pd.to_numeric(df_frag[iml_col], errors='coerce').tolist()
    poe_dict = {col: pd.to_numeric(df_frag[col], errors='coerce').tolist() for col in ls_cols}

    return imt, iml_values, poe_dict


def build_fragility_xml(exposure_csv_path, fragility_dir, out_xml_path):
    """
    Builds a consolidated fragility XML for all taxonomy strings present
    in the exposure CSV, matching each to its fragility CSV file.
    """
    assets_df = pd.read_csv(exposure_csv_path)
    unique_taxonomies = assets_df["taxonomy"].unique().tolist()
    print(f"  Unique taxonomy strings in exposure CSV: {len(unique_taxonomies)}")

    nrml = ET.Element("nrml", {"xmlns": "http://openquake.org/xmlns/nrml/0.5"})
    fm   = ET.SubElement(nrml, "fragilityModel", {
        "id"           : f"fragility_cdmx_{MODE}",
        "assetCategory": "buildings",
        "lossCategory" : "structural",
    })
    ET.SubElement(fm, "description").text = (
        "Structural fragility model for CDMX, derived from the GEM "
        "Global Fragility and Vulnerability Model (Martins & Silva 2020). "
        "Discrete fragility functions matched to building taxonomy strings."
    )
    ET.SubElement(fm, "limitStates").text = (
        "slight moderate extensive complete"
    )

    matched   = 0
    unmatched = []

    # Calculate the most prevalent taxonomy per occupancy to resolve 'UNK'
    prevalent_taxonomies = {}
    for occ, gem_df in gem_exposure_data.items():
        # Find the row with the highest building count
        top_tax = gem_df.loc[gem_df['BUILDINGS'].idxmax(), 'TAXONOMY']
        prevalent_taxonomies[occ.upper()] = top_tax

    for taxonomy in unique_taxonomies:
        csv_path, match_quality = find_fragility_csv(taxonomy, fragility_dir, prevalent_map=prevalent_taxonomies)

        if csv_path is None:
            unmatched.append(taxonomy)
            continue

        try:
            if csv_path not in _FRAGILITY_PARSE_CACHE:
                _FRAGILITY_PARSE_CACHE[csv_path] = parse_fragility_csv(csv_path)
            imt, iml_values, poe_dict = _FRAGILITY_PARSE_CACHE[csv_path]
            
            # --- FIX: Clean IMT string for OpenQuake compatibility ---
            # Removes trailing 's' from 'SA(0.3s)' -> 'SA(0.3)'
            if imt and "SA" in imt and imt.endswith("s)"):
                imt = imt.replace("s)", ")")
            # ---------------------------------------------------------
            
        except Exception as e:
            print(f"    Could not parse {csv_path}: {e}")
            unmatched.append(taxonomy)
            continue

        ff = ET.SubElement(fm, "fragilityFunction", {
            "id"    : taxonomy,
            "format": "discrete",
        })

        # noDamageLimit = first IML where any damage state has POE > 1e-10
        no_damage_iml = iml_values[0]
        for i, iml in enumerate(iml_values):
            if any(poe_dict[ls][i] > 1e-10 for ls in poe_dict):
                no_damage_iml = iml
                break

        ET.SubElement(ff, "imls", {
            "imt"          : imt,
            "noDamageLimit": f"{no_damage_iml:.6f}",
        }).text = " ".join(f"{v:.8f}" for v in iml_values)

        ls_name_map = {
            "Slight_damage"   : "slight",
            "Moderate_damage" : "moderate",
            "Extensive_damage": "extensive",
            "Complete_damage" : "complete",
        }
        for csv_ls, oq_ls in ls_name_map.items():
            if csv_ls in poe_dict:
                # Each 'v' is now a float, no splitting required
                poe_strings = [f"{float(v):.8f}" for v in poe_dict[csv_ls]]
                ET.SubElement(ff, "poes", {"ls": oq_ls}).text = " ".join(poe_strings)

        matched += 1
        if match_quality != "exact":
            print(f"    [{match_quality}] {taxonomy} → {os.path.basename(csv_path)}")

    print(f"  Fragility functions matched: {matched}/{len(unique_taxonomies)}")
    if unmatched:
        print(f"  Unmatched taxonomies ({len(unmatched)}):")
        for t in unmatched:
            print(f"    {t}")
    
    xml_str = minidom.parseString(
        ET.tostring(nrml, encoding="unicode")
    ).toprettyxml(indent="  ")
    with open(out_xml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("\n".join(xml_str.split("\n")[1:]))

    print(f"  Fragility XML saved → {out_xml_path}")


# Call after exposure CSV is written
build_fragility_xml(OUT_CSV, FRAGILITY_DIR, OUT_FRAGILITY_XML)




# ── Step 9: Convert USGS Shakemap XML to CSV, generate accompanying site model ─────────────────────────────

def generate_site_model(exposure_csv, xml_output):
    df = pd.read_csv(exposure_csv)[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
    print(f"Processing {len(df)} unique building locations...")

    # --- Construct NRML XML Structure ---
    nrml = ET.Element("nrml", {"xmlns": "http://openquake.org/xmlns/nrml/0.5"})
    site_model = ET.SubElement(nrml, "siteModel")

    for _, row in df.iterrows():
        ET.SubElement(site_model, "site", {
            "lon": f"{row['lon']:.5f}", 
            "lat": f"{row['lat']:.5f}",
        })

    xml_str = minidom.parseString(ET.tostring(nrml)).toprettyxml(indent="  ")
    with open(xml_output, "w", encoding="utf-8") as f:
        f.write(xml_str)
    
    print(f"Success: Filtered Site Model saved to {xml_output}")
    return df

def create_stochastic_gmf_csv_fast(usgs_cache: dict, output_csv: str,
                                    num_events: int = 100) -> None:
    """
    Vectorised replacement for create_stochastic_gmf_csv.

    Eliminates both Python for-loops via NumPy broadcasting:
      epsilons  : shape (num_events,)
      site arrays : shape (n_sites,)
      output arrays : (num_events, n_sites) — no Python iteration required

    Parameters
    ----------
    usgs_cache : dict produced by precompute_static.precompute_usgs_snapping()
    output_csv : destination path for the OQ-compatible GMF CSV
    num_events : stochastic realisations to generate (default 100)
    """
    m_pga   = usgs_cache["m_pga"]      # (n_sites,)
    m_03    = usgs_cache["m_03"]
    m_10    = usgs_cache["m_10"]
    m_06    = usgs_cache["m_06"]
    s_pga   = usgs_cache["s_pga"]
    s_03    = usgs_cache["s_03"]
    s_10    = usgs_cache["s_10"]
    s_06    = usgs_cache["s_06"]
    n_sites = int(usgs_cache["n_sites"])

    print(f"Generating {num_events} GMF realisations for {n_sites} sites "
          f"(vectorised)...")
    _t0 = time.perf_counter()

    # One epsilon per event — preserves spatial correlation (same as original)
    epsilons = rand_generator.normal(0.0, 1.0, num_events)   # (num_events,)

    # Broadcast: (num_events, 1) op (1, n_sites) → (num_events, n_sites)
    v_pga = np.exp(np.log(m_pga)[None, :] + epsilons[:, None] * s_pga[None, :])
    v_03  = np.exp(np.log(m_03)[None,  :] + epsilons[:, None] * s_03[None,  :])
    v_10  = np.exp(np.log(m_10)[None,  :] + epsilons[:, None] * s_10[None,  :])
    v_06  = np.exp(np.log(m_06)[None,  :] + epsilons[:, None] * s_06[None,  :])

    # Flat index arrays — no loop needed
    eid_arr = np.repeat(np.arange(num_events), n_sites)
    sid_arr = np.tile(np.arange(n_sites), num_events)

    pd.DataFrame({
        "rlzi":        0,
        "sid":         sid_arr,
        "eid":         eid_arr,
        "gmv_PGA":     v_pga.ravel(),
        "gmv_SA(0.3)": v_03.ravel(),
        "gmv_SA(0.6)": v_06.ravel(),
        "gmv_SA(1.0)": v_10.ravel(),
    }).to_csv(output_csv, index=False)

    print(f"  Exported {num_events * n_sites:,} rows in "
          f"{time.perf_counter() - _t0:.2f}s → {output_csv}")

# --- RUN WORKFLOW ---
base = f'./{MODE}/seed_{RANDOM_SEED}/'

# 1. Generate XML and get site coordinates
sites_df = generate_site_model(base+'exposure_assets.csv',base+'site_model.xml')


# 2. Load pre-snapped USGS cache and generate GMF realisations (vectorised)
if not os.path.exists(_USGS_CACHE_PKL):
    raise FileNotFoundError(
        f"USGS cache not found: {_USGS_CACHE_PKL}\n"
        "Run precompute_static.py first."
    )
with open(_USGS_CACHE_PKL, "rb") as _f:
    _usgs_cache = pickle.load(_f)

create_stochastic_gmf_csv_fast(_usgs_cache, base + 'ground_motion_grid.csv', num_events=100)


print(f"\nDone ({MODE} mode). Outputs in {OUT_DIR}")

end_wall_time = time.perf_counter()
end_cpu_total = get_total_cpu_time(parent_process)

wall_clock_elapsed = end_wall_time - start_wall_time
total_cpu_used = end_cpu_total - start_cpu_total
core_hours = total_cpu_used / 3600

print(f"\n{'='*30}")
print("COMPUTE AUDIT")
print(f"{'='*30}")
print(f"Total Wall-Clock Time: {wall_clock_elapsed:.2f} s")
print(f"Total CPU Time (Accumulated): {total_cpu_used:.2f} s")
print(f"Total Core-Hours: {core_hours:.6f} h")