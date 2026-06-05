import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import importlib

    _torch = importlib.import_module("torch")
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"Startup device check: using {device.upper()}")
    print(f"CUDA available: {_torch.cuda.is_available()}")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RemoteCLIP Roof Material XAI Attribution Scaffold

    Plug-and-play marimo notebook for feature attribution on fine-tuned RemoteCLIP ViT-L/14 roof-material classifier.

    Supported attribution families in this notebook:
    - Transformer Explainability
    - Manual GradCAM attribution
    - Captum Integrated Gradients
    - RISE black-box masking attribution

    How notebook is organized:
    1. environment and imports
    2. labels, prompts, and preprocessing
    3. asset resolution and model loading
    4. image sampling and prediction sanity check
    5. attribution method registration and visualization
    6. one output block per attribution method

    Each result section aims to answer two questions:
    - **what class did model predict?**
    - **which image regions most supported that prediction?**

    Read output panels as:
    - **left:** original roof image
    - **middle:** normalized attribution heatmap by itself
    - **right:** attribution heatmap overlaid on input image
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Project venv and optional dependencies

    **What this section does:** makes sure notebook runs from project virtual environment and can self-install notebook-only dependencies if imports are missing.

    This marimo notebook is intended to run from project venv at `../.venv`.

    From repo root, launch with:

    ```bash
    .venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py
    ```

    **Expected output:** next cell prints either:
    - missing packages being installed into `sys.executable`, or
    - confirmation that all required notebook packages already import cleanly.

    If install step runs, rerunning cell later should usually print clean "already installed" status.
    """)
    return


@app.cell
def _():
    import importlib.util
    import subprocess
    import sys

    REQUIRED_PACKAGES = {
        "captum": ["captum"],
        "kagglehub": ["kagglehub", "kagglesdk==0.1.23"],
        "open_clip": ["open_clip_torch"],
        "marimo": ["marimo"],
        "pandas": ["pandas", "pyarrow"],
    }

    packages_to_install = []
    for import_name, pip_specs in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            packages_to_install.extend(pip_specs)

    if packages_to_install:
        print(f"Installing missing packages into {sys.executable}: {packages_to_install}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages_to_install])
    else:
        print(f"All optional notebook packages import from {sys.executable}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and notebook config

    **What this section does:** imports core libraries, then defines one centralized typed config block used by all later cells.

    **Expected output:** device selection and inferred repository root. Config cell below then exposes all main notebook knobs in one place for interactive editing.
    """)
    return


@app.cell
def _():
    import os
    import random
    import tempfile
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Callable, Dict, List, Optional, Tuple

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    from attribution_helpers import batch_recovery
    from attribution_helpers import feature_attribution_aggregation as faa
    from attribution_helpers import manual_gradcam
    from attribution_helpers import captum_integrated_gradients
    from attribution_helpers import dataset_split_helpers
    from attribution_helpers import rise
    from attribution_helpers.transformer_explainability import transformer_explainability

    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("Install open_clip_torch: pip install open_clip_torch") from exc

    try:
        import kagglehub
    except ImportError:
        kagglehub = None

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        IntegratedGradients = None

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    REPO_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "xAI_notebooks" else Path.cwd().resolve()

    print(f"Device: {DEVICE}")
    print(f"Repo root: {REPO_ROOT}")
    return (
        Callable,
        DEVICE,
        Dict,
        Image,
        IntegratedGradients,
        List,
        Optional,
        Path,
        REPO_ROOT,
        Tuple,
        batch_recovery,
        captum_integrated_gradients,
        dataclass,
        dataset_split_helpers,
        faa,
        kagglehub,
        manual_gradcam,
        nn,
        np,
        open_clip,
        pd,
        plt,
        random,
        rise,
        tempfile,
        torch,
        transformer_explainability,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Centralized notebook config

    **What this section does:** collects main notebook settings into one typed config object with grouped sections for assets, preprocessing, visualization, attribution, and batch defaults.

    **How to use it:** edit values in this cell, then rerun dependent cells. Notebook is designed so most routine experimentation should start here.
    """)
    return


@app.cell
def _(Optional, Path, REPO_ROOT, dataclass):
    @dataclass(frozen=True)
    class AssetConfig:
        # =========================
        # Asset source selection
        # =========================
        asset_mode: str  # "local" or "kagglehub"
        kaggle_dataset: str  # KaggleHub dataset slug used when asset_mode="kagglehub"
        local_model_weights: Path  # Local checkpoint path for fine-tuned RemoteCLIP weights
        local_image_dir: Path  # Local directory containing cropped roof images
        local_metadata_csv: Path  # Optional local metadata CSV path
        image_exts: tuple[str, ...]  # File suffixes treated as valid image assets

    @dataclass(frozen=True)
    class PreprocessConfig:
        # =========================
        # Image preprocessing
        # =========================
        image_size: tuple[int, int]  # Input resolution expected by RemoteCLIP
        clip_mean: tuple[float, float, float]  # CLIP channel means for normalization
        clip_std: tuple[float, float, float]  # CLIP channel stds for normalization

    @dataclass(frozen=True)
    class VisualizationConfig:
        # =========================
        # Attribution figure display
        # =========================
        overlay_alpha: float  # Transparency for overlay panel
        cmap: str  # Matplotlib colormap for attribution heatmaps
        figure_size: tuple[int, int]  # Figure size for three-panel attribution plots
        preview_figure_size: tuple[int, int]  # Figure size for sampled input image preview

    @dataclass(frozen=True)
    class IntegratedGradientsConfig:
        # =========================
        # Integrated Gradients
        # =========================
        n_steps: int  # Number of integration steps for Captum IG

    @dataclass(frozen=True)
    class RiseConfig:
        # =========================
        # RISE black-box attribution
        # =========================
        num_masks: int  # Number of random masks to sample
        mask_grid_size: int  # Low-resolution grid size before upsampling masks
        p_save: float  # Probability each mask cell stays visible
        batch_size: int  # Number of masked images scored per forward chunk
        return_diagnostics: bool  # Print runtime/sampling diagnostics during notebook runs

    @dataclass(frozen=True)
    class BatchConfig:
        # =========================
        # Future batch runner defaults
        # =========================
        num_images: Optional[int]  # Placeholder count for future batch processing
        split: str  # Dataset split selector for batch runs
        methods: tuple[str, ...]  # Exact attribution methods to run in batch mode
        target: str  # Attribution target policy for future batch runs
        output_dir: Path  # Directory where future batch outputs should be written
        helper_csv_dir: Path  # Optional directory for exported split helper CSV artifacts

    @dataclass(frozen=True)
    class NotebookConfig:
        # =========================
        # Root notebook config
        # =========================
        seed: int  # Global seed for Python/numpy/torch and stochastic methods
        model_name: str  # OpenCLIP model architecture name
        pretrained_weights: str  # Base pretrained weights used before fine-tuned checkpoint load
        assets: AssetConfig
        preprocess: PreprocessConfig
        visualization: VisualizationConfig
        integrated_gradients: IntegratedGradientsConfig
        rise: RiseConfig
        batch: BatchConfig

    CONFIG = NotebookConfig(
        seed=42,
        model_name="ViT-L-14",
        pretrained_weights="laion2b_s32b_b82k",
        assets=AssetConfig(
            asset_mode="local",
            kaggle_dataset="doubleblindreview/xbd-roof-images",
            local_model_weights=REPO_ROOT / "best_clip_model_balanced.pth",
            local_image_dir=REPO_ROOT / "RoofNet-Images",
            local_metadata_csv=REPO_ROOT / "roofnet_metadata.csv",
            image_exts=(".jpg", ".jpeg", ".png", ".webp"),
        ),
        preprocess=PreprocessConfig(
            image_size=(224, 224),
            clip_mean=(0.48145466, 0.4578275, 0.40821073),
            clip_std=(0.26862954, 0.26130258, 0.27577711),
        ),
        visualization=VisualizationConfig(
            overlay_alpha=0.45,
            cmap="inferno",
            figure_size=(12, 4),
            preview_figure_size=(4, 4),
        ),
        integrated_gradients=IntegratedGradientsConfig(
            n_steps=50,
        ),
        rise=RiseConfig(
            num_masks=512,
            mask_grid_size=12,
            p_save=0.5,
            batch_size=32,
            return_diagnostics=True,
        ),
        batch=BatchConfig(
            num_images=None,
            split="holdout",
            methods=("transformer_explainability",),
            target="predicted_top1",
            output_dir=REPO_ROOT / "xAI_outputs",
            helper_csv_dir=REPO_ROOT / "xAI_notebooks",
        ),
    )

    CONFIG
    return (CONFIG,)


@app.cell
def _(CONFIG, np, random, torch):
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    print(f"Seeded python/numpy/torch with CONFIG.seed={CONFIG.seed}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Labels, prompts, and preprocessing

    **What this section does:** defines roof-material label set, derives city-conditioned text prompts from filenames, and builds CLIP-compatible image preprocessing.

    **Why this matters:** classifier prediction is not plain image-only classification; image is compared against text prompts like `{material} in {city_name}`. Small changes here affect every prediction and attribution result downstream.

    **Expected output:** no rich display yet; this block mainly prepares reusable functions and constants for later cells.
    """)
    return


@app.cell
def _(CONFIG, List, Path, torch, transforms):
    MATERIAL_CLASSES = [
        "Thatch", "StoneSlates", "ClayTiles", "AsphaltTiles",
        "ConcreteTiles", "WoodTiles", "MetalSheetMaterials", "PolycarbonateSheetMaterials",
        "GlassSheetMaterials", "AmorphousConcrete", "AmorphousAsphalt",
        "AmorphousMembrane", "AmorphousFabric", "Unknown", "GreenVegetative"
    ]

    def extract_city_name_from_filename(filename: str) -> str:
        base = Path(filename).stem
        if '-' in base:
            city_part = base.split('-')[0]
            return city_part.replace('_', ' ').title()
        if 'height' in base:
            city_part = base.split('_height')[0]
            return city_part.replace('_', ' ').title()
        if 'imsat' in base:
            city_part = base.split('_imsat')[0]
            return city_part.replace('_', ' ').title()
        return base

    def build_prompts(city_name: str) -> List[str]:
        return [f"{material} in {city_name}" for material in MATERIAL_CLASSES]

    preprocess = transforms.Compose([
        transforms.Resize(CONFIG.preprocess.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(CONFIG.preprocess.clip_mean),
                             std=list(CONFIG.preprocess.clip_std)),
    ])

    def denormalize_clip_tensor(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(CONFIG.preprocess.clip_mean, device=x.device).view(3, 1, 1)
        std = torch.tensor(CONFIG.preprocess.clip_std, device=x.device).view(3, 1, 1)
        return (x * std + mean).clamp(0, 1)

    return (
        MATERIAL_CLASSES,
        build_prompts,
        extract_city_name_from_filename,
        preprocess,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Asset resolver: local or KaggleHub

    **What this section does:** resolves where notebook should find model weights, image directory, and optional metadata.

    Use this section when switching between:
    - **local mode:** assets already present inside repo/workstation layout
    - **KaggleHub mode:** assets downloaded on demand from dataset mirror

    **Expected output:** resolved filesystem paths plus small diagnostics showing which asset source was selected. If this section fails, later model-loading and inference cells will also fail.
    """)
    return


@app.cell
def _(CONFIG, List, Optional, Path, REPO_ROOT, dataclass, kagglehub):
    @dataclass
    class AssetPaths:
        root: Path
        model_weights: Path
        image_dir: Path
        metadata_csv: Optional[Path] = None

    def _find_first(root: Path, patterns: List[str]) -> Optional[Path]:
        for pattern in patterns:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
        return None

    def resolve_assets(asset_mode: str = CONFIG.assets.asset_mode) -> AssetPaths:
        if asset_mode == "local":
            assets = AssetPaths(
                root=REPO_ROOT,
                model_weights=CONFIG.assets.local_model_weights,
                image_dir=CONFIG.assets.local_image_dir,
                metadata_csv=CONFIG.assets.local_metadata_csv if CONFIG.assets.local_metadata_csv.exists() else None,
            )
        elif asset_mode == "kagglehub":
            if kagglehub is None:
                raise ImportError("Install kagglehub: pip install kagglehub")
            kaggle_root = Path(kagglehub.dataset_download(CONFIG.assets.kaggle_dataset)).resolve()
            print("Path to dataset files:", kaggle_root)
            model_weights = _find_first(kaggle_root, ["best_clip_model_balanced.pth", "*.pth"])
            image_dir = _find_first(kaggle_root, ["xBD_cropped_roofs", "*cropped*roofs*"])
            metadata_csv = _find_first(kaggle_root, ["roofnet_metadata.csv", "*.csv"])
            if model_weights is None:
                raise FileNotFoundError(f"Could not find .pth weights under {kaggle_root}")
            if image_dir is None or not image_dir.is_dir():
                # fallback: use parent directory of first image
                first_image = next((p for p in kaggle_root.rglob("*") if p.suffix.lower() in CONFIG.assets.image_exts), None)
                if first_image is None:
                    raise FileNotFoundError(f"Could not find image files under {kaggle_root}")
                image_dir = first_image.parent
            assets = AssetPaths(kaggle_root, model_weights, image_dir, metadata_csv)
        else:
            raise ValueError("asset_mode must be 'local' or 'kagglehub'")

        if not assets.model_weights.exists():
            raise FileNotFoundError(f"Missing model weights: {assets.model_weights}")
        if not assets.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {assets.image_dir}")
        return assets

    assets = resolve_assets(CONFIG.assets.asset_mode)
    assets
    return (assets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Load fine-tuned RemoteCLIP

    **What this section does:** instantiates base RemoteCLIP ViT-L/14 architecture, loads fine-tuned roof-material checkpoint, moves model to active device, and fetches matching tokenizer.

    **Expected output:** path to weight file that was loaded. This is important provenance signal: attribution is only meaningful if notebook is pointing at intended checkpoint.
    """)
    return


@app.cell
def _(CONFIG, DEVICE, Path, assets, open_clip, torch):
    def load_remoteclip_model(weights_path: Path, device: str = DEVICE):
        model, _, _ = open_clip.create_model_and_transforms(CONFIG.model_name, pretrained=CONFIG.pretrained_weights)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(CONFIG.model_name)
        return model, tokenizer

    model, tokenizer = load_remoteclip_model(assets.model_weights)
    print("Loaded model from:", assets.model_weights)
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Image selection and inference wrapper

    **What this section does:** finds candidate roof images, samples one example, preprocesses it, builds city-aware prompts, and runs top-k prediction sanity check.

    **Expected output:**
    - number of images discovered under active image directory
    - sampled filename
    - parsed city context
    - top-5 class probabilities for sampled image
    - displayed input image titled with top-1 predicted material

    Treat this as main pre-attribution checkpoint. If predicted label or prompt context looks wrong here, attribution maps later may still render but answer wrong question.
    """)
    return


@app.cell
def _(
    CONFIG,
    DEVICE,
    Dict,
    Image,
    List,
    MATERIAL_CLASSES,
    Optional,
    Path,
    Tuple,
    assets,
    build_prompts,
    dataset_split_helpers,
    extract_city_name_from_filename,
    model,
    pd,
    plt,
    preprocess,
    random,
    tokenizer,
    torch,
):
    def list_images(image_dir: Path, limit: Optional[int] = None) -> List[Path]:
        paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in CONFIG.assets.image_exts)
        return paths[:limit] if limit else paths

    def load_image_tensor(image_path: Path, device: str = DEVICE) -> Tuple[Image.Image, torch.Tensor]:
        pil = Image.open(image_path).convert("RGB")
        tensor = preprocess(pil).unsqueeze(0).to(device)
        return pil, tensor

    @torch.no_grad()
    def predict(image_tensor: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
        tokenized = tokenizer(prompts).to(image_tensor.device)
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(tokenized)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        return {"logits": logits.squeeze(0), "probs": probs.squeeze(0)}

    def target_score_forward(image_tensor: torch.Tensor, target_idx: int, prompts: List[str]) -> torch.Tensor:
        # Captum-compatible scalar target score. Keep gradients enabled for image path.
        tokenized = tokenizer(prompts).to(image_tensor.device)
        image_features = model.encode_image(image_tensor)
        with torch.no_grad():
            text_features = model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        return logits[:, target_idx]

    all_images = list_images(assets.image_dir)
    print(f"Found {len(all_images)} images under {assets.image_dir}")

    split_diagnostics = None
    helper_csv_outputs = {}
    if assets.metadata_csv is not None and assets.metadata_csv.exists():
        metadata_df = pd.read_csv(assets.metadata_csv, low_memory=False)
        print(f"Loaded metadata from: {assets.metadata_csv}")
        helper_csv_outputs = dataset_split_helpers.write_split_helper_csvs(
            metadata_df,
            output_dir=CONFIG.batch.helper_csv_dir,
        )
        images, split_diagnostics = dataset_split_helpers.collect_split_image_paths(
            image_dir=assets.image_dir,
            metadata_df=metadata_df,
            split=CONFIG.batch.split,
            image_exts=CONFIG.assets.image_exts,
        )
    else:
        print("No metadata CSV found. Falling back to unfiltered image discovery.")
        images = all_images

    if not images:
        raise ValueError(f"No images available after applying split filter {CONFIG.batch.split!r}.")

    print(f"Using {len(images)} images for split={CONFIG.batch.split!r}")
    sample_path = random.choice(images)
    city_name = extract_city_name_from_filename(sample_path.name)
    prompts = build_prompts(city_name)
    pil_img, image_tensor = load_image_tensor(sample_path)

    pred = predict(image_tensor, prompts)
    topk = torch.topk(pred["probs"], k=5)
    print("Sample:", sample_path.name)
    print("City prompt context:", city_name)
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"{MATERIAL_CLASSES[idx]:>28s}: {score:.3f}")

    plt.figure(figsize=CONFIG.visualization.preview_figure_size)
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title(MATERIAL_CLASSES[topk.indices[0].item()])
    return (
        image_tensor,
        images,
        load_image_tensor,
        pil_img,
        predict,
        prompts,
        target_score_forward,
        topk,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Visualization helpers

    **What this section does:** standardizes how attribution arrays become viewable figures.

    **How to read the figure:**
    - **Input:** original RGB roof tile/crop
    - **Attribution:** normalized heatmap only
    - **Overlay:** heatmap placed on top of original image for spatial interpretation

    Heatmaps are min-max normalized for display. Good for visual comparison within one run, but raw display intensity should not be over-interpreted as a calibrated score across unrelated methods or images.
    """)
    return


@app.cell
def _(CONFIG, Image, np, plt, torch):
    def normalize_attr(attr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        attr = np.asarray(attr, dtype=np.float32)
        raw_min = float(np.nanmin(attr))
        raw_max = float(np.nanmax(attr))
        attr = np.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
        attr = attr - attr.min()
        denom = float(attr.max())
        if denom <= eps:
            print(
                "WARNING: Attribution heatmap is constant/zero before display normalization; "
                f"raw min={raw_min:.6g}, raw max={raw_max:.6g}."
            )
            return np.zeros_like(attr, dtype=np.float32)
        return attr / (denom + eps)

    def show_attribution(
        image: Image.Image,
        heatmap: np.ndarray,
        title: str,
        alpha: float = CONFIG.visualization.overlay_alpha,
        cmap: str = CONFIG.visualization.cmap,
    ):
        heatmap = normalize_attr(heatmap)
        resample = getattr(Image, "Resampling", Image).BILINEAR
        heatmap = np.asarray(Image.fromarray(heatmap).resize(image.size, resample=resample))
        fig, axes = plt.subplots(1, 3, figsize=CONFIG.visualization.figure_size)
        axes[0].imshow(image); axes[0].set_title("Input"); axes[0].axis("off")
        axes[1].imshow(heatmap, cmap=cmap); axes[1].set_title("Attribution"); axes[1].axis("off")
        axes[2].imshow(image); axes[2].imshow(heatmap, cmap=cmap, alpha=alpha); axes[2].set_title(title); axes[2].axis("off")
        plt.tight_layout()
        plt.close(fig)
        return fig

    def tensor_attr_to_heatmap(attr: torch.Tensor) -> np.ndarray:
        # Expected shapes: [1, C, H, W] or [C, H, W]
        attr = attr.detach().float().cpu()
        if attr.ndim == 4:
            attr = attr[0]
        if attr.ndim == 3:
            attr = attr.abs().sum(dim=0)
        return attr.numpy()

    return (show_attribution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Attribution method registry

    Each method accepts `(image_tensor, target_idx, prompts)` and returns a 2D heatmap array. Add/replace methods without changing visualization code.
    """)
    return


@app.cell
def _(Callable, Dict, List, np, torch):
    AttributionFn = Callable[[torch.Tensor, int, List[str]], np.ndarray]
    ATTRIBUTION_METHODS: Dict[str, AttributionFn] = {}

    def register_attribution(name: str):
        def decorator(fn: AttributionFn):
            ATTRIBUTION_METHODS[name] = fn
            return fn
        return decorator

    return ATTRIBUTION_METHODS, register_attribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Transformer Explainability

    **What this section does:** registers attention-based attribution method for RemoteCLIP ViT-L/14.

    Intuition:
    - inspect visual transformer attention at each layer
    - weight attention by gradient signal from chosen target score
    - roll relevance from CLS token back onto spatial patch grid

    **Expected output later:** a 2D relevance map showing which patches most supported current top-1 image-text similarity score.

    Notes:
    - patches each visual transformer attention block to request `need_weights=True`
    - captures attention weights plus their gradients with respect to target image-text similarity
    - builds gradient-weighted per-layer relevance matrices
    - rolls CLS relevance back onto 16×16 patch grid, then upsamples to 224×224
    """)
    return


@app.cell
def _(
    CONFIG,
    List,
    model,
    np,
    register_attribution,
    tokenizer,
    torch,
    transformer_explainability,
):
    @register_attribution("transformer_explainability")
    def transformer_explainability_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        return transformer_explainability(
            model=model,
            tokenizer=tokenizer,
            image_tensor=image_tensor,
            prompts=prompts,
            target_idx=target_idx,
            image_size=CONFIG.preprocess.image_size,
            verbose=verbose,
        )

    TRANSFORMER_EXPLAINABILITY_REGISTERED = True
    return (TRANSFORMER_EXPLAINABILITY_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Manual GradCAM methods

    **What this section does:** registers two manual GradCAM-style methods that probe different stages of visual encoder.

    GradCAM variants:
    - `vit_token_gradcam`: penultimate visual transformer block, CLS dropped, patch tokens reshaped to grid
    - `manual_patch_gradcam`: raw patch projection layer `model.visual.conv1`

    Interpretation guide:
    - **ViT-token GradCAM:** later, more semantic focus after transformer processing
    - **Patch-embed GradCAM:** earlier, lower-level spatial evidence near image-to-patch projection stage
    """)
    return


@app.cell
def _(
    List,
    manual_gradcam,
    model,
    nn,
    np,
    register_attribution,
    target_score_forward,
    torch,
):
    class TargetScoreModule(nn.Module):
        def __init__(self, prompts: List[str], target_idx: int):
            super().__init__()
            self.prompts = prompts
            self.target_idx = target_idx

        def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
            return target_score_forward(image_tensor, self.target_idx, self.prompts)

    def inspect_visual_layers(model) -> None:
        for name, module in model.visual.named_modules():
            if any(key in name.lower() for key in ["block", "resblock", "attn", "ln_post"]):
                print(name, "->", module.__class__.__name__)

    @register_attribution("vit_token_gradcam")
    def vit_token_gradcam_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        score_module = TargetScoreModule(prompts, target_idx)
        model.zero_grad(set_to_none=True)
        try:
            return manual_gradcam.vit_token_gradcam_heatmap(
                model=model,
                score_forward=score_module,
                image_tensor=image_tensor,
                verbose=verbose,
            )
        finally:
            model.zero_grad(set_to_none=True)

    @register_attribution("manual_patch_gradcam")
    def manual_patch_gradcam_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        score_module = TargetScoreModule(prompts, target_idx)
        model.zero_grad(set_to_none=True)
        try:
            return manual_gradcam.manual_patch_gradcam_heatmap(
                model=model,
                score_forward=score_module,
                image_tensor=image_tensor,
                verbose=verbose,
            )
        finally:
            model.zero_grad(set_to_none=True)

    MANUAL_GRADCAM_METHODS_REGISTERED = True
    return (MANUAL_GRADCAM_METHODS_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Captum Integrated Gradients

    **What this section does:** registers input-space attribution methods based on path-integrated gradients from zero baseline to actual image tensor.

    Integrated Gradients variants:
    - `captum_integrated_gradients_abs`: absolute channel-sum input attribution
    - `captum_integrated_gradients_positive`: positive-only channel-sum input attribution

    Both use zero baseline, `n_steps=50`, and print Captum convergence delta.

    **Expected output later:** heatmaps often look smoother than raw gradient methods. If absolute and positive variants look very similar, that usually means negative attribution mass was small for this sample.
    """)
    return


@app.cell
def _(
    CONFIG,
    IntegratedGradients,
    List,
    captum_integrated_gradients,
    model,
    nn,
    np,
    register_attribution,
    target_score_forward,
    torch,
):
    class IntegratedGradientsTargetScoreModule(nn.Module):
        def __init__(self, prompts: List[str], target_idx: int):
            super().__init__()
            self.prompts = prompts
            self.target_idx = target_idx

        def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
            return target_score_forward(image_tensor, self.target_idx, self.prompts)

    def _integrated_gradients_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        reduction: str,
        verbose: bool = True,
    ) -> np.ndarray:
        if IntegratedGradients is None:
            raise ImportError("Install captum: pip install captum")
        score_module = IntegratedGradientsTargetScoreModule(prompts, target_idx)
        model.zero_grad(set_to_none=True)
        try:
            return captum_integrated_gradients.integrated_gradients_heatmap(
                score_forward=score_module,
                image_tensor=image_tensor,
                integrated_gradients_cls=IntegratedGradients,
                reduction=reduction,
                n_steps=CONFIG.integrated_gradients.n_steps,
                verbose=verbose,
            )
        finally:
            model.zero_grad(set_to_none=True)

    @register_attribution("captum_integrated_gradients_abs")
    def captum_integrated_gradients_abs_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        return _integrated_gradients_attr(image_tensor, target_idx, prompts, reduction="abs", verbose=verbose)

    @register_attribution("captum_integrated_gradients_positive")
    def captum_integrated_gradients_positive_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        return _integrated_gradients_attr(image_tensor, target_idx, prompts, reduction="positive", verbose=verbose)

    CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED = True
    return (CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. RISE raw-image black-box attribution

    **What this section does:** registers black-box attribution method that repeatedly masks image and measures how target score changes.

    `rise_raw_image` uses randomized input sampling with mean-baseline masks in CLIP-normalized tensor space. It runs many masked RemoteCLIP forwards in chunks, so expect it to be slower than gradient/attention methods.

    **Expected output later:** spatial saliency map plus concise diagnostics about mask sampling and runtime. Good contrast against gradient-based methods because it does not rely on internal gradients through model layers.
    """)
    return


@app.cell
def _(CONFIG, List, model, np, register_attribution, rise, tokenizer, torch):
    def rise_unscaled_target_score(image_batch: torch.Tensor, target_idx: int, prompts: List[str]) -> torch.Tensor:
        tokenized = tokenizer(prompts).to(image_batch.device)
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(tokenized)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features @ text_features.T[:, target_idx]

    @register_attribution("rise_raw_image")
    def rise_raw_image_attr(
        image_tensor: torch.Tensor,
        target_idx: int,
        prompts: List[str],
        *,
        verbose: bool = True,
    ) -> np.ndarray:
        generator_device = image_tensor.device if image_tensor.device.type != "mps" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(CONFIG.seed)
        model.eval()
        return rise.rise_heatmap(
            score_forward=lambda masked_batch: rise_unscaled_target_score(masked_batch, target_idx, prompts),
            image_tensor=image_tensor,
            num_masks=CONFIG.rise.num_masks,
            mask_grid_size=CONFIG.rise.mask_grid_size,
            p_save=CONFIG.rise.p_save,
            batch_size=CONFIG.rise.batch_size,
            mask_device=image_tensor.device,
            return_diagnostics=CONFIG.rise.return_diagnostics,
            generator=generator,
            verbose=verbose,
        )[0]

    RISE_METHODS_REGISTERED = True
    return (RISE_METHODS_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Run one attribution method

    **What this section does:** provides one shared runner that takes registered method name, computes heatmap for current top-1 class, and replaces cell output with standardized visualization figure.

    Each method-specific block below reuses same sampled image, same prompt set, and same top-1 prediction target. That makes visual differences easier to attribute to explanation method rather than changed inputs.
    """)
    return


@app.cell
def _(
    ATTRIBUTION_METHODS: "Dict[str, AttributionFn]",
    CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,
    MANUAL_GRADCAM_METHODS_REGISTERED,
    MATERIAL_CLASSES,
    RISE_METHODS_REGISTERED,
    TRANSFORMER_EXPLAINABILITY_REGISTERED,
    image_tensor,
    mo,
    pil_img,
    prompts,
    show_attribution,
    topk,
):
    _ = (
        MANUAL_GRADCAM_METHODS_REGISTERED,
        CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,
        RISE_METHODS_REGISTERED,
        TRANSFORMER_EXPLAINABILITY_REGISTERED,
    )

    def run_attribution_method(method_name: str) -> None:
        target_idx = int(topk.indices[0].item())
        try:
            method_heatmap = ATTRIBUTION_METHODS[method_name](image_tensor, target_idx, prompts)
            method_fig = show_attribution(pil_img, method_heatmap, f"{method_name}: {MATERIAL_CLASSES[target_idx]}")
            mo.output.replace(method_fig)
        except KeyError as exc:
            available_methods = ", ".join(sorted(ATTRIBUTION_METHODS))
            raise KeyError(f"Unknown attribution method {method_name!r}. Available: {available_methods}") from exc
        except NotImplementedError as exc:
            print(exc)

    return (run_attribution_method,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run Transformer Explainability

    Use this output to inspect attention-derived relevance after information has flowed through full visual transformer stack. Expect broader semantic regions rather than very sharp pixel-level contours.
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("transformer_explainability")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run ViT-token GradCAM

    This view asks which late transformer tokens most supported top-1 score. Compare against Transformer Explainability to see whether both methods highlight similar roof subregions.
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("vit_token_gradcam")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run patch-embed GradCAM

    This view focuses earlier in encoder. Useful for checking whether coarse evidence already appears at patch projection stage or only emerges after transformer mixing.
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("manual_patch_gradcam")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run Integrated Gradients absolute channel sum

    Absolute reduction treats both positive and negative channel contributions as magnitude. Good first view when you want total sensitivity regardless of sign.
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("captum_integrated_gradients_abs")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run Integrated Gradients positive-only channel sum

    Positive-only reduction suppresses negative evidence and keeps only features that increase target score. Compare this directly against absolute variant to judge whether inhibitory evidence mattered.
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("captum_integrated_gradients_positive")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run RISE raw-image black-box attribution

    Black-box view: saliency comes from repeated masking experiments rather than backprop through internals.

    Slow path: generates 512 soft masks and runs RemoteCLIP in chunks. Increase `num_masks` to 1024–2000+ only for higher-quality offline/publication runs.

    Expect this cell to take longer than earlier methods. Use it as robustness check when you want attribution that depends less on internal architectural assumptions.
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("rise_raw_image")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Batch attribution visualizations

    This cell mirrors earlier visualization runners, but loops across a batch of images.

    What it does:
    - takes first `CONFIG.batch.num_images` discovered images
    - runs all registered attribution methods
    - saves rendered attribution figures under `xAI_outputs/`
    - writes a resume manifest and skips already-finished image+method jobs on rerun

    What it does not do:
    - no aggregate statistics
    - no summary plots
    """)
    return


@app.cell
def _(
    ATTRIBUTION_METHODS: "Dict[str, AttributionFn]",
    CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,
    CONFIG,
    MANUAL_GRADCAM_METHODS_REGISTERED,
    MATERIAL_CLASSES,
    Path,
    RISE_METHODS_REGISTERED,
    TRANSFORMER_EXPLAINABILITY_REGISTERED,
    batch_recovery,
    build_prompts,
    extract_city_name_from_filename,
    faa,
    images,
    load_image_tensor,
    mo,
    plt,
    predict,
    show_attribution,
    tempfile,
    torch,
):
    _ = (
        MANUAL_GRADCAM_METHODS_REGISTERED,
        CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,
        RISE_METHODS_REGISTERED,
        TRANSFORMER_EXPLAINABILITY_REGISTERED,
    )

    def run_batch_attribution_visualizations() -> dict:
        CONFIG.batch.output_dir.mkdir(parents=True, exist_ok=True)

        method_groups = {
            "transformer_explainability": ["transformer_explainability"],
            "manual_gradcam": [
                "vit_token_gradcam",
                "manual_patch_gradcam",
            ],
            "captum_integrated_gradients": [
                "captum_integrated_gradients_abs",
                "captum_integrated_gradients_positive",
            ],
            "rise": ["rise_raw_image"],
        }

        all_methods = tuple(method for methods in method_groups.values() for method in methods)
        requested_methods = tuple(CONFIG.batch.methods)
        if not requested_methods or requested_methods == ("all",):
            selected_methods = all_methods
        else:
            unknown_methods = [method for method in requested_methods if method not in ATTRIBUTION_METHODS]
            if unknown_methods:
                available_methods = ", ".join(sorted(ATTRIBUTION_METHODS))
                raise ValueError(
                    f"Unknown batch methods: {unknown_methods}. Available methods: {available_methods}"
                )
            selected_methods = requested_methods

        selected_method_groups = {
            family_name: [method for method in family_methods if method in selected_methods]
            for family_name, family_methods in method_groups.items()
        }
        selected_method_groups = {
            family_name: family_methods
            for family_name, family_methods in selected_method_groups.items()
            if family_methods
        }
        print(f"Batch method selection: {selected_methods}")

        if CONFIG.batch.target != "predicted_top1":
            raise NotImplementedError(
                f"Batch runner currently supports only CONFIG.batch.target='predicted_top1', got {CONFIG.batch.target!r}"
            )

        selected_images = images if CONFIG.batch.num_images is None else images[: CONFIG.batch.num_images]
        if not selected_images:
            raise ValueError("No images available for batch attribution run.")

        manifest_path = CONFIG.batch.output_dir / "batch_run_manifest.json"
        manifest = batch_recovery.load_manifest(manifest_path)
        transformer_family_dir = CONFIG.batch.output_dir / "transformer_explainability"
        transformer_stats_csv_path = transformer_family_dir / "transformer_spatial_stats.csv"

        saved_outputs = []
        skipped_outputs = []
        failed_jobs = []
        total_jobs = sum(len(family_methods) for family_methods in selected_method_groups.values()) * len(selected_images)
        print(
            f"Expected batch workload: images={len(selected_images)}, "
            f"methods={len(selected_methods)}, total_jobs={total_jobs}"
        )
        if total_jobs >= 100:
            print(
                "WARNING: Large batch run requested. "
                "Consider reducing split size, num_images, or methods before continuing."
            )

        preexisting_done = 0
        for family_name, family_methods in selected_method_groups.items():
            method_dir = CONFIG.batch.output_dir / family_name
            method_dir.mkdir(parents=True, exist_ok=True)
            for image_path in selected_images:
                output_stem = image_path.stem.replace(" ", "_")
                for method_name in family_methods:
                    output_path = method_dir / f"{output_stem}__{method_name}.png"
                    job_id = batch_recovery.make_job_id(image_path.name, method_name)
                    batch_recovery.upsert_job(
                        manifest,
                        job_id=job_id,
                        image_id=image_path.name,
                        method_name=method_name,
                        output_path=output_path,
                    )
                    if batch_recovery.resolve_job_action(manifest, job_id=job_id, output_path=output_path) == "skip":
                        preexisting_done += 1
        batch_recovery.save_manifest(manifest_path, manifest)
        print(f"Resume scan: skip_existing={preexisting_done}, rerun_remaining={total_jobs - preexisting_done}")

        with mo.status.progress_bar(
            total=total_jobs,
            title="Batch attribution",
            subtitle=f"0/{total_jobs} jobs accounted for",
            completion_title="Batch attribution complete",
        ) as progress:
            completed_jobs = 0
            for family_name, family_methods in selected_method_groups.items():
                method_dir = CONFIG.batch.output_dir / family_name
                method_dir.mkdir(parents=True, exist_ok=True)

                for image_path in selected_images:
                    batch_city_name = extract_city_name_from_filename(image_path.name)
                    batch_prompts = build_prompts(batch_city_name)
                    batch_pil_img, batch_image_tensor = load_image_tensor(image_path)
                    batch_pred = predict(batch_image_tensor, batch_prompts)
                    batch_target_idx = int(torch.argmax(batch_pred["probs"]).item())
                    batch_target_label = MATERIAL_CLASSES[batch_target_idx]
                    output_stem = image_path.stem.replace(" ", "_")

                    for method_name in family_methods:
                        batch_fig = None
                        output_path = method_dir / f"{output_stem}__{method_name}.png"
                        job_id = batch_recovery.make_job_id(image_path.name, method_name)
                        action = batch_recovery.resolve_job_action(
                            manifest,
                            job_id=job_id,
                            output_path=output_path,
                        )
                        if action == "skip":
                            skipped_outputs.append(str(output_path))
                            completed_jobs += 1
                            progress.update(
                                title="Batch attribution",
                                subtitle=(
                                    f"{completed_jobs}/{total_jobs} | skipped | "
                                    f"{image_path.name} | {method_name}"
                                ),
                            )
                            continue

                        batch_recovery.mark_job_running(manifest, job_id)
                        batch_recovery.save_manifest(manifest_path, manifest)
                        try:
                            batch_heatmap = ATTRIBUTION_METHODS[method_name](
                                batch_image_tensor,
                                batch_target_idx,
                                batch_prompts,
                                verbose=False,
                            )
                            batch_fig = show_attribution(
                                batch_pil_img,
                                batch_heatmap,
                                f"{method_name}: {batch_target_label}",
                            )
                            with tempfile.NamedTemporaryFile(
                                suffix=".png",
                                dir=method_dir,
                                delete=False,
                            ) as tmp_file:
                                temp_output_path = Path(tmp_file.name)
                            batch_fig.savefig(temp_output_path, bbox_inches="tight")
                            plt.close(batch_fig)
                            batch_recovery.atomic_replace_file(temp_output_path, output_path)
                            saved_outputs.append(str(output_path))

                            if method_name == "transformer_explainability":
                                spatial_stats = faa.compute_spatial_stats(
                                    batch_heatmap,
                                    method="transformer_explainability",
                                    image_id=image_path.name,
                                )
                                spatial_stats["target_idx"] = batch_target_idx
                                spatial_stats["target_label"] = batch_target_label
                                spatial_stats["city_name"] = batch_city_name
                                batch_recovery.append_stats_row(transformer_stats_csv_path, spatial_stats)

                            batch_recovery.mark_job_done(manifest, job_id)
                            batch_recovery.save_manifest(manifest_path, manifest)
                        except Exception as exc:
                            if batch_fig is not None:
                                plt.close(batch_fig)
                            batch_recovery.mark_job_failed(manifest, job_id, repr(exc))
                            batch_recovery.save_manifest(manifest_path, manifest)
                            failed_jobs.append({
                                "job_id": job_id,
                                "image_id": image_path.name,
                                "method_name": method_name,
                                "error": repr(exc),
                            })
                            print(f"FAILED batch job {job_id}: {exc!r}")
                            raise

                        completed_jobs += 1
                        progress.update(
                            title="Batch attribution",
                            subtitle=(
                                f"{completed_jobs}/{total_jobs} | "
                                f"{image_path.name} | {method_name}"
                            ),
                        )

        manifest_summary = batch_recovery.summarize_jobs(manifest)
        print(
            "Batch manifest summary: "
            f"done={manifest_summary.get('done', 0)}, "
            f"failed={manifest_summary.get('failed', 0)}, "
            f"pending={manifest_summary.get('pending', 0)}, "
            f"running={manifest_summary.get('running', 0)}"
        )
        return {
            "num_images": CONFIG.batch.num_images,
            "num_selected_images": len(selected_images),
            "selected_images": [path.name for path in selected_images],
            "method_groups": selected_method_groups,
            "selected_methods": list(selected_methods),
            "split": CONFIG.batch.split,
            "target": CONFIG.batch.target,
            "output_dir": str(CONFIG.batch.output_dir),
            "saved_outputs": saved_outputs,
            "skipped_outputs": skipped_outputs,
            "failed_jobs": failed_jobs,
            "manifest_path": str(manifest_path),
            "transformer_stats_csv": str(transformer_stats_csv_path),
            "manifest_summary": manifest_summary,
        }

    BATCH_RUN_RESULTS = run_batch_attribution_visualizations()
    BATCH_RUN_RESULTS
    return (BATCH_RUN_RESULTS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Batch aggregate statistics

    This cell only computes aggregate transformer attribution statistics using
    `xAI_notebooks/attribution_helpers/feature_attribution_aggregation.py`.
    It reloads persisted per-image stats from disk, so aggregation can resume after kernel death.
    """)
    return


@app.cell
def _(BATCH_RUN_RESULTS, CONFIG, Path, faa, pd, plt):
    transformer_stats_csv_path = Path(BATCH_RUN_RESULTS["transformer_stats_csv"])
    if not transformer_stats_csv_path.exists():
        raise ValueError(
            "No persisted transformer stats CSV found for aggregation. "
            f"Expected: {transformer_stats_csv_path}"
        )

    transformer_family_dir = CONFIG.batch.output_dir / "transformer_explainability"
    transformer_stats_df = pd.read_csv(transformer_stats_csv_path)
    if transformer_stats_df.empty:
        raise ValueError("Transformer stats CSV is empty; no completed transformer jobs to aggregate.")

    stats_csv_path = transformer_family_dir / "transformer_spatial_stats.csv"
    stats_parquet_path = transformer_family_dir / "transformer_spatial_stats.parquet"
    summary_csv_path = transformer_family_dir / "transformer_spatial_summary.csv"
    radial_profile_csv_path = transformer_family_dir / "transformer_radial_profile_summary.csv"
    radial_profile_png_path = transformer_family_dir / "transformer_radial_profile.png"
    center_mass_hist_png_path = transformer_family_dir / "transformer_center25_hist.png"
    centroid_offset_hist_png_path = transformer_family_dir / "transformer_centroid_offset_hist.png"

    transformer_stats_df = transformer_stats_df.drop_duplicates(subset=["image_id", "method"], keep="last")
    transformer_stats_df.to_csv(stats_csv_path, index=False)
    transformer_stats_df.to_parquet(stats_parquet_path, index=False)

    aggregate = faa.aggregate_spatial_stats(transformer_stats_df)
    summary_df = pd.DataFrame([aggregate["summary"]])
    radial_profile_df = aggregate["radial_profile"]
    summary_df.to_csv(summary_csv_path, index=False)
    radial_profile_df.to_csv(radial_profile_csv_path, index=False)

    radial_fig, radial_ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(radial_profile_df)))
    radial_ax.plot(x, radial_profile_df["mean"], marker="o", label="mean")
    radial_ax.fill_between(
        x,
        radial_profile_df["mean"] - radial_profile_df["std"],
        radial_profile_df["mean"] + radial_profile_df["std"],
        alpha=0.25,
        label="±1 std",
    )
    radial_ax.set_xticks(x)
    radial_ax.set_xticklabels(radial_profile_df["ring"], rotation=30, ha="right")
    radial_ax.set_ylabel("Attribution mass")
    radial_ax.set_title("Transformer radial attribution profile")
    radial_ax.legend()
    radial_fig.tight_layout()
    radial_fig.savefig(radial_profile_png_path, bbox_inches="tight")
    plt.close(radial_fig)

    center_hist_fig, center_hist_ax = plt.subplots(figsize=(6, 4))
    center_hist_ax.hist(transformer_stats_df["mass_center_25_square"], bins=10)
    center_hist_ax.set_title("Center 25% attribution mass")
    center_hist_ax.set_xlabel("Mass fraction")
    center_hist_ax.set_ylabel("Image count")
    center_hist_fig.tight_layout()
    center_hist_fig.savefig(center_mass_hist_png_path, bbox_inches="tight")
    plt.close(center_hist_fig)

    centroid_hist_fig, centroid_hist_ax = plt.subplots(figsize=(6, 4))
    centroid_hist_ax.hist(transformer_stats_df["centroid_offset_norm"], bins=10)
    centroid_hist_ax.set_title("Centroid offset distribution")
    centroid_hist_ax.set_xlabel("Normalized offset")
    centroid_hist_ax.set_ylabel("Image count")
    centroid_hist_fig.tight_layout()
    centroid_hist_fig.savefig(centroid_offset_hist_png_path, bbox_inches="tight")
    plt.close(centroid_hist_fig)

    BATCH_AGGREGATION_RESULTS = {
        "stats_csv": str(stats_csv_path),
        "stats_parquet": str(stats_parquet_path),
        "summary_csv": str(summary_csv_path),
        "radial_profile_csv": str(radial_profile_csv_path),
        "radial_profile_png": str(radial_profile_png_path),
        "center_mass_hist_png": str(center_mass_hist_png_path),
        "centroid_offset_hist_png": str(centroid_offset_hist_png_path),
        "summary": aggregate["summary"],
    }
    BATCH_AGGREGATION_RESULTS
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
