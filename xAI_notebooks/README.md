# xAI Notebooks

This folder contains notebook-oriented code for explainability experiments on the fine-tuned RemoteCLIP roof-material classifier.

Current focus:
- interactive attribution analysis in **marimo**
- helper modules for attribution methods and aggregation
- batch export of attribution figures and spatial aggregation artifacts

## Folder structure

```text
xAI_notebooks/
├── README.md
├── remoteclip_xai_attribution_marimo.py
└── attribution_helpers/
    ├── __init__.py
    ├── feature_attribution_aggregation.py
    ├── transformer_explainability.py
    ├── captum_gradcam.py
    ├── captum_integrated_gradients.py
    └── rise.py
```

## What each file does

### `remoteclip_xai_attribution_marimo.py`
Main marimo notebook entrypoint.

What it handles:
- environment/import checks
- model + asset loading
- image sampling
- RemoteCLIP prediction sanity checks
- attribution method registration
- per-method visualization
- batch attribution export
- transformer explainability aggregation export

If you are exploring this repo and want to start somewhere, start here.

### `attribution_helpers/feature_attribution_aggregation.py`
Helper module for spatial aggregation of attribution heatmaps.

Current responsibilities:
- heatmap normalization
- center-crop mass statistics
- radial attribution profiles
- centroid / peak offset metrics
- 50% attribution radius metrics
- aggregate summary generation

Designed so same aggregation path can later support:
- Transformer Explainability
- GradCAM
- Integrated Gradients
- RISE

### `attribution_helpers/`
Implementation helpers for each attribution family.

#### `transformer_explainability.py`
Transformer attention-gradient relevance rollout for RemoteCLIP ViT-L/14.

#### `captum_gradcam.py`
Captum-based GradCAM utilities for:
- ViT token-level GradCAM
- patch embedding GradCAM

#### `captum_integrated_gradients.py`
Captum Integrated Gradients helper functions.

#### `rise.py`
RISE black-box masking attribution helper functions.

## Environment setup

Project instructions require using repo root virtual environment: `.venv`.

From repo root:

### 1. Create virtual environment if needed

```bash
python3.14 -m venv .venv
```

### 2. Activate virtual environment

```bash
source .venv/bin/activate
```

### 3. Install base project requirements

```bash
pip install -r requirements.txt
```

### 4. Install notebook/dev requirements on top

```bash
pip install -r requirements-dev.txt
```

Why both:
- `requirements.txt` installs broader project dependencies
- `requirements-dev.txt` adds notebook/xAI-specific tools like `marimo`, `captum`, `kagglehub`, and pinned `kagglesdk`

## Run notebook with marimo

From repo root, after activating `.venv`:

### Edit mode

```bash
.venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py
```

This opens notebook in interactive marimo edit mode.

### Run mode

```bash
.venv/bin/marimo run xAI_notebooks/remoteclip_xai_attribution_marimo.py
```

Use run mode when you want notebook to execute as an app/script instead of editing cells interactively.

## Typical workflow

From repo root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
.venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py
```

## Expected assets

Notebook supports local assets and KaggleHub-backed assets.

Common local assets used by notebook:
- `best_clip_model_balanced.pth`
- `xBD_cropped_roofs/xBD_cropped_roofs/`
- optional `roofnet_metadata.csv`

Asset behavior is configured inside notebook config cell.

## Outputs

Notebook batch runs write outputs under:
- `xAI_outputs/`

Current batch outputs include:
- per-method attribution PNGs
- transformer explainability spatial stats CSV
- transformer explainability spatial stats Parquet
- transformer aggregation summary CSV
- radial profile summary CSV
- radial profile plot PNG
- center-mass histogram PNG
- centroid-offset histogram PNG

### Aggregate metrics reference

The final notebook cell processes Transformer Explainability heatmaps through `attribution_helpers/feature_attribution_aggregation.py`. The per-image CSV contains these columns:

#### Raw heatmap properties

| Column | Meaning |
|---|---|
| `raw_sum` | Sum of all pixel values before normalization |
| `raw_abs_sum` | Sum of absolute pixel values before normalization |
| `raw_min` / `raw_max` | Min and max pixel value in raw heatmap |
| `negative_mass_ratio` | `\|negative\| / \|total\|` — fraction of absolute mass that is negative. Near 0 → model used mostly positive evidence; near 0.5 → equal positive/negative |
| `is_zero_sum` | True if heatmap is all zeros (attribution failed for that image) |

#### Spatial concentration

| Column | Meaning |
|---|---|
| `mass_center_25_square` | Fraction of total attribution mass inside the central 25%-area square |
| `mass_center_50_square` | Fraction of total attribution mass inside the central 50%-area square |
| `radius_for_50_mass_square` | Side fraction (0–1) of the smallest centered square that captures 50% of mass. Smaller → more concentrated |
| `radius_for_50_mass_radial` | Normalized radius (0–1) of the smallest centered circle that captures 50% of mass. Smaller → more concentrated |
| `radius_50_gap` | Square minus radial radius. Positive → mass is more circular than square; negative → mass follows square/edge pattern |

#### Centroid and peak location

| Column | Meaning |
|---|---|
| `centroid_x` / `centroid_y` | Attribution-weighted centroid in pixel coordinates |
| `centroid_offset_px` / `centroid_offset_norm` | Euclidean distance from image center to centroid, in pixels / normalized to [0, 1]. Small offset_norm → model focused near center of image |
| `peak_x` / `peak_y` | Coordinates of the single highest-attribution pixel |
| `peak_offset_px` / `peak_offset_norm` | Offset of the peak pixel from center. Compare with centroid offset to distinguish broad (centroid near center, peak off-center) vs. sharp focus |

#### Radial profile

| Column | Meaning |
|---|---|
| `radial_profile_00_20` through `radial_profile_80_100` | Attribution mass fraction in each concentric ring (0–20%, 20–40%, …, 80–100% of max radius). Monotonically decreasing → center-focused; flat → diffuse |

#### Cross-image summary (`transformer_spatial_summary.csv`)

| Metric | Meaning |
|---|---|
| `num_images` | Number of heatmaps processed |
| `zero_sum_images` | Count of all-zero heatmaps |
| `median_mass_center_25_square` / `iqr_*` | Typical fraction of mass in center 25% area, with IQR spread. High median + narrow IQR → consistent center focus |
| `median_mass_center_50_square` / `iqr_*` | Same for center 50% area |
| `median_radius_for_50_mass_square` / `iqr_*` | Typical square crop size to capture half the mass |
| `median_radius_for_50_mass_radial` / `iqr_*` | Typical radial radius to capture half the mass |
| `fraction_center25_over_50pct` | Proportion of images where >50% of mass falls in center 25% area. High → strong center bias |
| `median_centroid_offset_norm` | Typical centroid displacement. Near 0 → consistent center focus |
| `median_peak_offset_norm` | Typical peak displacement. Compare with centroid offset |
| `mean_negative_mass_ratio` | Average negative evidence across batch. If high, consider positive-only aggregation instead |
| `median_radius_50_gap` | Typical gap between square and radial 50% radii. Large positive → mass is more circular than square |

#### Generated plots

| File | How to read it |
|---|---|
| `transformer_radial_profile.png` | Mean ± 1 std of attribution mass across radial rings. Steep drop → center-concentrated; flat → diffuse |
| `transformer_center25_hist.png` | Histogram of `mass_center_25_square`. Right-skewed → most images concentrate in the center |
| `transformer_centroid_offset_hist.png` | Histogram of `centroid_offset_norm`. Tight cluster near 0 → model looks at center consistently; spread out → variable focus |

## Notes

- notebook currently uses **marimo**, not Jupyter, as primary interactive environment
- helper modules are meant to keep notebook cells thinner and easier to test
- transformer explainability aggregation is implemented first, but schema already includes `method` so future cross-method comparison is easier
- notebook contains lightweight self-install logic for some optional packages, but preferred path is still installing from `requirements.txt` and `requirements-dev.txt` first

## Troubleshooting

### Import errors
Make sure `.venv` is activated and both requirements files were installed:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Wrong Python or package set
Check interpreter:

```bash
which python
python --version
```

Expected pattern:
- python should resolve inside `.venv`
- project has been using Python `3.14.x`

### Marimo command not found
Run marimo through repo venv directly:

```bash
.venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py
```

### KaggleHub issues
This repo currently pins:
- `kagglehub==1.0.1`
- `kagglesdk==0.1.23`

Reason: newer `kagglesdk` version previously caused import breakage for notebook workflow.
