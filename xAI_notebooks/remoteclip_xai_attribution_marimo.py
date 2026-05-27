import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RemoteCLIP Roof Material XAI Attribution Scaffold

    Plug-and-play notebook scaffold for feature attribution on the fine-tuned RemoteCLIP ViT-L/14 roof-material classifier.

    Supported hooks planned:
    - Transformer Explainability
    - Captum LayerGradCAM / GradCAM-style attribution
    - Captum Integrated Gradients

    > This notebook is intentionally a scaffold: model loading, asset resolution, inference, visualization, and Transformer Explainability are runnable; Captum-based attribution methods remain to be filled in next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Project venv and optional dependencies

    This marimo notebook is intended to run from the project venv at `../.venv`.

    From the repo root, launch with:

    ```bash
    .venv/bin/marimo edit xAI_notebooks/remoteclip_xai_attribution_marimo.py
    ```

    The next cell installs missing notebook-only packages into that same venv by calling `sys.executable -m pip`.
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
    ## 1. Imports and global config
    """)
    return


@app.cell
def _():
    import os
    import random
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Callable, Dict, List, Optional, Tuple

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    import captum_gradcam
    import captum_integrated_gradients
    from transformer_explainability import transformer_explainability

    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("Install open_clip_torch: pip install open_clip_torch") from exc

    try:
        import kagglehub
    except ImportError:
        kagglehub = None

    try:
        from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
    except ImportError:
        IntegratedGradients = LayerGradCam = LayerAttribution = None

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    REPO_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "xAI_notebooks" else Path.cwd().resolve()

    ASSET_MODE = "local"  # options: "local", "kagglehub"
    KAGGLE_DATASET = "doubleblindreview/xbd-roof-images"

    LOCAL_MODEL_WEIGHTS = REPO_ROOT / "best_clip_model_balanced.pth"
    LOCAL_IMAGE_DIR = REPO_ROOT / "xBD_cropped_roofs" / "xBD_cropped_roofs"
    LOCAL_METADATA_CSV = REPO_ROOT / "roofnet_metadata.csv"

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Repo root: {REPO_ROOT}")
    return (
        ASSET_MODE,
        Callable,
        DEVICE,
        Dict,
        IMAGE_EXTS,
        Image,
        IntegratedGradients,
        KAGGLE_DATASET,
        LOCAL_IMAGE_DIR,
        LOCAL_METADATA_CSV,
        LOCAL_MODEL_WEIGHTS,
        LayerGradCam,
        List,
        Optional,
        Path,
        REPO_ROOT,
        Tuple,
        captum_gradcam,
        captum_integrated_gradients,
        dataclass,
        kagglehub,
        nn,
        np,
        open_clip,
        plt,
        random,
        torch,
        transformer_explainability,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Labels, prompts, and preprocessing
    """)
    return


@app.cell
def _(List, Path, torch, transforms):
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    def denormalize_clip_tensor(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(3, 1, 1)
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
    ## 3. Asset resolver: local or KaggleHub
    """)
    return


@app.cell
def _(
    ASSET_MODE,
    IMAGE_EXTS,
    KAGGLE_DATASET,
    LOCAL_IMAGE_DIR,
    LOCAL_METADATA_CSV,
    LOCAL_MODEL_WEIGHTS,
    List,
    Optional,
    Path,
    REPO_ROOT,
    dataclass,
    kagglehub,
):
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

    def resolve_assets(asset_mode: str = ASSET_MODE) -> AssetPaths:
        if asset_mode == "local":
            assets = AssetPaths(
                root=REPO_ROOT,
                model_weights=LOCAL_MODEL_WEIGHTS,
                image_dir=LOCAL_IMAGE_DIR,
                metadata_csv=LOCAL_METADATA_CSV if LOCAL_METADATA_CSV.exists() else None,
            )
        elif asset_mode == "kagglehub":
            if kagglehub is None:
                raise ImportError("Install kagglehub: pip install kagglehub")
            kaggle_root = Path(kagglehub.dataset_download(KAGGLE_DATASET)).resolve()
            print("Path to dataset files:", kaggle_root)
            model_weights = _find_first(kaggle_root, ["best_clip_model_balanced.pth", "*.pth"])
            image_dir = _find_first(kaggle_root, ["xBD_cropped_roofs", "*cropped*roofs*"])
            metadata_csv = _find_first(kaggle_root, ["roofnet_metadata.csv", "*.csv"])
            if model_weights is None:
                raise FileNotFoundError(f"Could not find .pth weights under {kaggle_root}")
            if image_dir is None or not image_dir.is_dir():
                # fallback: use parent directory of first image
                first_image = next((p for p in kaggle_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS), None)
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

    assets = resolve_assets(ASSET_MODE)
    assets
    return (assets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Load fine-tuned RemoteCLIP
    """)
    return


@app.cell
def _(DEVICE, Path, assets, open_clip, torch):
    def load_remoteclip_model(weights_path: Path, device: str = DEVICE):
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        return model, tokenizer

    model, tokenizer = load_remoteclip_model(assets.model_weights)
    print("Loaded model from:", assets.model_weights)
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Image selection and inference wrapper
    """)
    return


@app.cell
def _(
    DEVICE,
    Dict,
    IMAGE_EXTS,
    Image,
    List,
    MATERIAL_CLASSES,
    Optional,
    Path,
    Tuple,
    assets,
    build_prompts,
    extract_city_name_from_filename,
    model,
    plt,
    preprocess,
    random,
    tokenizer,
    torch,
):
    def list_images(image_dir: Path, limit: Optional[int] = None) -> List[Path]:
        paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
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

    images = list_images(assets.image_dir)
    print(f"Found {len(images)} images under {assets.image_dir}")
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

    plt.figure(figsize=(4, 4))
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title(MATERIAL_CLASSES[topk.indices[0].item()])
    return image_tensor, pil_img, prompts, target_score_forward, topk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Visualization helpers
    """)
    return


@app.cell
def _(Image, np, plt, torch):
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

    def show_attribution(image: Image.Image, heatmap: np.ndarray, title: str, alpha: float = 0.45, cmap: str = "inferno"):
        heatmap = normalize_attr(heatmap)
        resample = getattr(Image, "Resampling", Image).BILINEAR
        heatmap = np.asarray(Image.fromarray(heatmap).resize(image.size, resample=resample))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
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
    ## 7. Attribution method registry

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
    ## 8. Transformer Explainability

    Implemented for RemoteCLIP ViT-L/14.

    Notes:
    - patches each visual transformer attention block to request `need_weights=True`
    - captures attention weights plus their gradients with respect to target image-text similarity
    - builds gradient-weighted per-layer relevance matrices
    - rolls CLS relevance back onto the 16×16 patch grid, then upsamples to 224×224
    """)
    return


@app.cell
def _(
    List,
    model,
    np,
    register_attribution,
    tokenizer,
    torch,
    transformer_explainability,
):
    @register_attribution("transformer_explainability")
    def transformer_explainability_attr(image_tensor: torch.Tensor, target_idx: int, prompts: List[str]) -> np.ndarray:
        return transformer_explainability(
            model=model,
            tokenizer=tokenizer,
            image_tensor=image_tensor,
            prompts=prompts,
            target_idx=target_idx,
            image_size=(224, 224),
        )

    TRANSFORMER_EXPLAINABILITY_REGISTERED = True
    return (TRANSFORMER_EXPLAINABILITY_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Captum GradCAM

    Captum LayerGradCAM variants:
    - `captum_gradcam_vit_tokens`: last visual transformer block, CLS dropped, patch tokens reshaped to grid
    - `captum_gradcam_patch_embed`: raw patch projection layer `model.visual.conv1`
    """)
    return


@app.cell
def _(
    LayerGradCam,
    List,
    captum_gradcam,
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

    @register_attribution("captum_gradcam_vit_tokens")
    def captum_gradcam_vit_tokens_attr(image_tensor: torch.Tensor, target_idx: int, prompts: List[str]) -> np.ndarray:
        if LayerGradCam is None:
            raise ImportError("Install captum: pip install captum")
        score_module = TargetScoreModule(prompts, target_idx)
        model.zero_grad(set_to_none=True)
        try:
            return captum_gradcam.vit_token_gradcam_heatmap(
                model=model,
                score_forward=score_module,
                image_tensor=image_tensor,
                layer_gradcam_cls=LayerGradCam,
            )
        finally:
            model.zero_grad(set_to_none=True)

    @register_attribution("captum_gradcam_patch_embed")
    def captum_gradcam_patch_embed_attr(image_tensor: torch.Tensor, target_idx: int, prompts: List[str]) -> np.ndarray:
        if LayerGradCam is None:
            raise ImportError("Install captum: pip install captum")
        score_module = TargetScoreModule(prompts, target_idx)
        model.zero_grad(set_to_none=True)
        try:
            return captum_gradcam.patch_embed_gradcam_heatmap(
                model=model,
                score_forward=score_module,
                image_tensor=image_tensor,
                layer_gradcam_cls=LayerGradCam,
            )
        finally:
            model.zero_grad(set_to_none=True)

    CAPTUM_GRADCAM_METHODS_REGISTERED = True
    return (CAPTUM_GRADCAM_METHODS_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Captum Integrated Gradients

    Integrated Gradients variants:
    - `captum_integrated_gradients_abs`: absolute channel-sum input attribution
    - `captum_integrated_gradients_positive`: positive-only channel-sum input attribution

    Both use a zero baseline, `n_steps=50`, and print Captum convergence delta.
    """)
    return


@app.cell
def _(
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
                n_steps=50,
            )
        finally:
            model.zero_grad(set_to_none=True)

    @register_attribution("captum_integrated_gradients_abs")
    def captum_integrated_gradients_abs_attr(image_tensor: torch.Tensor, target_idx: int, prompts: List[str]) -> np.ndarray:
        return _integrated_gradients_attr(image_tensor, target_idx, prompts, reduction="abs")

    @register_attribution("captum_integrated_gradients_positive")
    def captum_integrated_gradients_positive_attr(image_tensor: torch.Tensor, target_idx: int, prompts: List[str]) -> np.ndarray:
        return _integrated_gradients_attr(image_tensor, target_idx, prompts, reduction="positive")

    CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED = True
    return (CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Run one attribution method
    """)
    return


@app.cell
def _(
    ATTRIBUTION_METHODS: "Dict[str, AttributionFn]",
    CAPTUM_GRADCAM_METHODS_REGISTERED,
    CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,
    MATERIAL_CLASSES,
    TRANSFORMER_EXPLAINABILITY_REGISTERED,
    image_tensor,
    mo,
    pil_img,
    prompts,
    show_attribution,
    topk,
):
    _ = (
        CAPTUM_GRADCAM_METHODS_REGISTERED,
        CAPTUM_INTEGRATED_GRADIENTS_METHODS_REGISTERED,
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
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("captum_gradcam_vit_tokens")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run patch-embed GradCAM
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("captum_gradcam_patch_embed")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run Integrated Gradients absolute channel sum
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
    """)
    return


@app.cell
def _(run_attribution_method):
    run_attribution_method("captum_integrated_gradients_positive")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Batch/run config placeholder

    TODO: iterate over selected images and methods, save overlays to `xAI_outputs/`.
    """)
    return


@app.cell
def _(ATTRIBUTION_METHODS: "Dict[str, AttributionFn]", REPO_ROOT):
    OUTPUT_DIR = REPO_ROOT / "xAI_outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

    BATCH_CONFIG = {
        "num_images": 8,
        "methods": list(ATTRIBUTION_METHODS.keys()),
        "target": "predicted_top1",
        "output_dir": str(OUTPUT_DIR),
    }
    BATCH_CONFIG
    return


if __name__ == "__main__":
    app.run()
