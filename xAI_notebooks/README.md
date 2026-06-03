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
├── feature_attribution_aggregation.py
└── attribution_helpers/
    ├── __init__.py
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

### `feature_attribution_aggregation.py`
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
