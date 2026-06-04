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
    ├── manual_gradcam.py
    ├── captum_integrated_gradients.py
    ├── rise.py
    └── dataset_split_helpers.py
```

## What each file does

### `remoteclip_xai_attribution_marimo.py`
Main marimo notebook entrypoint.

What it handles:
- environment/import checks
- model + asset loading
- dataset-split-aware image sampling via metadata CSV
- RemoteCLIP prediction sanity checks
- attribution method registration (no Captum dependency for GradCAM)
- per-method visualization
- configurable batch attribution export (method subset, split filter)
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

#### `manual_gradcam.py`
Manual GradCAM utilities for:
- ViT token-level GradCAM
- patch embedding GradCAM

#### `captum_integrated_gradients.py`
Captum Integrated Gradients helper functions.

#### `rise.py`
RISE black-box masking attribution helper functions.

#### `dataset_split_helpers.py`
Utilities for reproducible dataset-split-based image selection:
- `collect_split_image_paths()` — filters metadata CSV by split (`train`/`val`/`holdout`/`all`), joins against actual files on disk
- `write_split_helper_csvs()` — exports split-filtered CSV artifacts for auditability

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

Batch outputs depend on configured methods and split:
- per-method attribution PNGs
- per-method-family spatial stats CSV (Parquet) + aggregation summary CSV
- per-method-family radial profile, center-mass, and centroid-offset plot PNGs
- split helper CSV artifacts (written alongside notebook by default)

### Aggregate metrics reference

The batch runner processes each method family's heatmaps through `attribution_helpers/feature_attribution_aggregation.py`. The per-image CSV contains these columns:

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

#### Cross-image summary (`{method_family}_spatial_summary.csv`)

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
| `{method_family}_radial_profile.png` | Mean ± 1 std of attribution mass across radial rings. Steep drop → center-concentrated; flat → diffuse |
| `{method_family}_center25_hist.png` | Histogram of `mass_center_25_square`. Right-skewed → most images concentrate in the center |
| `{method_family}_centroid_offset_hist.png` | Histogram of `centroid_offset_norm`. Tight cluster near 0 → model looks at center consistently; spread out → variable focus |

### Holdout-set Transformer Explainability summary (286 images)

These numbers come from a full holdout-set batch run. Sample size is now large enough that patterns here should be treated as stable signals, not small-sample noise.

#### Quick table

| Section | Metric | Value | Interpretation |
|---|---|---:|---|
| Sample health | `num_images` | 286 | Full holdout set processed; statistics now meaningful. |
| Sample health | `zero_sum_images` | 0 | No failed or degenerate heatmaps. |
| Sample health | `mean_negative_mass_ratio` | 0.0 | Transformer Explainability stayed strictly non-negative across holdout. |
| Concentration | `median_mass_center_25_square` | 0.300 | Center quarter holds 30% of attribution; only weak center bias over 25% uniform baseline. |
| Concentration | `iqr_mass_center_25_square` | 0.131 | Attention concentration varies meaningfully image to image. |
| Concentration | `median_mass_center_50_square` | 0.534 | Center half holds just over 53% of attribution; model reads most of image. |
| Concentration | `iqr_mass_center_50_square` | 0.131 | Similar spread as center-quarter metric. |
| Concentration | `fraction_center25_over_50pct` | 0.059 | Only ~6% of images put majority of attribution in center quarter. |
| Half-mass area | `median_radius_for_50_mass_square` | 0.685 | Large square crop needed to capture half the attribution. |
| Half-mass area | `iqr_radius_for_50_mass_square` | 0.098 | Pattern fairly consistent across holdout. |
| Half-mass area | `median_radius_for_50_mass_radial` | 0.543 | Radial estimate confirms attribution extends well beyond center. |
| Half-mass area | `iqr_radius_for_50_mass_radial` | 0.088 | Slightly tighter spread than square estimate. |
| Shape | `median_radius_50_gap` | 0.137 | Attribution often has directional lean rather than radial symmetry. |
| Location | `median_centroid_offset_norm` | 0.105 | Average attribution mass stays broadly near image center. |
| Location | `median_peak_offset_norm` | 0.541 | Strongest individual cue usually lies near image periphery. |

#### Sample health

- `num_images: 286` — full holdout set processed.
- `zero_sum_images: 0` — every image produced a valid, non-degenerate attribution heatmap; no failed or blank outputs needed exclusion.
- `mean_negative_mass_ratio: 0.0` — Transformer Explainability stayed strictly non-negative across all images, so normalization was clean and consistent throughout.

#### How concentrated is model attention?

- `median_mass_center_25_square: 0.300` — on a typical image, the center quarter of pixels contains 30% of total attribution. Uniform-random baseline would be 25%, so center preference exists but is weak.
- `iqr_mass_center_25_square: 0.131` — meaningful image-to-image spread; some samples are more center-focused, many are not.
- `median_mass_center_50_square: 0.534` — the center half of the image captures just over 53% of attribution on median. Against a 50% uniform baseline, this again suggests only weak center bias.
- `iqr_mass_center_50_square: 0.131` — similar spread as center-25 metric; concentration is not uniform across dataset.
- `fraction_center25_over_50pct: 0.059` — only about 6% of images (~17/286) place more than half their attribution mass inside the center quarter. Most images show diffuse, spread-out attention.

#### How much area captures half the signal?

- `median_radius_for_50_mass_square: 0.685` — square center-crops must span 68.5% of image half-width to capture 50% of attribution on a typical image. Large crop, weak central concentration.
- `iqr_radius_for_50_mass_square: 0.098` — moderate spread; this pattern is fairly consistent across holdout.
- `median_radius_for_50_mass_radial: 0.543` — radially, 54.3% of max image radius is needed to capture half the attribution mass. Same story: attribution extends well beyond center.
- `iqr_radius_for_50_mass_radial: 0.088` — slightly tighter than square estimate, consistent with radial measure being less sensitive to directional asymmetry.
- `median_radius_50_gap: 0.137` — square and radial estimates differ by ~14 percentage points on median, suggesting attribution is not radially symmetric and often has directional lean.

#### Where is model actually looking?

- `median_centroid_offset_norm: 0.105` — attribution center of mass sits only ~10.5% of image half-width away from geometric center. In aggregate, attention is still broadly centered.
- `median_peak_offset_norm: 0.541` — the single strongest attribution pixel sits ~54.1% of image half-width away from center on median. Strongest individual cue is typically near image periphery, not roof center.

#### Main interpretation

Most important tension in these results:

- **Centroid story:** `0.105` suggests average attribution mass stays roughly centered.
- **Peak story:** `0.541` suggests highest-value individual evidence often lives near image edge.

Interpretation: model appears to use broadly distributed contextual evidence, while its strongest single cue often comes from peripheral content. That peripheral cue could be neighboring structures, shadows, vegetation, or road-edge context rather than rooftop material alone.

## Notes

- notebook currently uses **marimo**, not Jupyter, as primary interactive environment
- helper modules are meant to keep notebook cells thinner and easier to test
- GradCAM methods are fully manual (no Captum `LayerGradCam` dependency) — `manual_gradcam.py` replaces the old `captum_gradcam.py`
- batch runner supports method selection via `CONFIG.batch.methods` and dataset-split filtering via `CONFIG.batch.split`
- aggregation schema includes a `method` column, enabling cross-method spatial metric comparison
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
