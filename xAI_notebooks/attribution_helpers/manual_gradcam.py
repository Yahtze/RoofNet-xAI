"""Manual GradCAM helpers for RemoteCLIP ViT visual attribution."""

from __future__ import annotations

from typing import Callable, Tuple
import warnings

import numpy as np
import torch


def _grid_size(model: torch.nn.Module) -> Tuple[int, int]:
    grid_size = getattr(model.visual, "grid_size", None)
    if grid_size is None:
        raise AttributeError("model.visual.grid_size is required for ViT token GradCAM")
    if isinstance(grid_size, tuple):
        return int(grid_size[0]), int(grid_size[1])
    size = int(grid_size)
    return size, size


def _warn_and_print(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    print(f"WARNING: {message}")


def _tensor_stats(name: str, tensor: torch.Tensor, *, verbose: bool = True) -> None:
    if not verbose:
        return
    tensor = tensor.detach().float().cpu()
    print(
        f"{name}: shape={tuple(tensor.shape)}, min={float(tensor.min()):.6g}, "
        f"max={float(tensor.max()):.6g}, abs_max={float(tensor.abs().max()):.6g}, "
        f"abs_mean={float(tensor.abs().mean()):.6g}"
    )


def normalize_heatmap(heatmap: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    raw_heatmap = heatmap.detach().float().cpu()
    heatmap = torch.relu(raw_heatmap)

    # GradCAM is positive-evidence by default. If ReLU collapses a non-zero map,
    # surface that we are showing magnitude / inverted evidence instead of silently
    # implying positive evidence.
    if float(heatmap.max()) <= eps and float(raw_heatmap.abs().max()) > eps:
        _warn_and_print(
            "GradCAM ReLU attribution collapsed to all zeros; showing absolute attribution magnitude instead. "
            "Interpret this heatmap as inverted/negative evidence, not positive evidence."
        )
        heatmap = raw_heatmap.abs()

    heatmap = heatmap - heatmap.min()
    denom = heatmap.max() - heatmap.min()
    if float(denom) <= eps:
        _warn_and_print("GradCAM heatmap is constant after normalization; returning zero heatmap.")
        return torch.zeros_like(heatmap).numpy().astype(np.float32)
    heatmap = heatmap / denom
    return heatmap.numpy().astype(np.float32)


def spatial_attr_to_heatmap(
    attr: torch.Tensor,
    *,
    label: str = "GradCAM spatial attribution",
    verbose: bool = True,
) -> np.ndarray:
    attr = attr.detach().float().cpu()
    _tensor_stats(f"{label} raw attribution", attr, verbose=verbose)
    if attr.ndim == 4:
        attr = attr[0]
    if attr.ndim == 3:
        attr = attr.mean(dim=0)
    if attr.ndim != 2:
        raise ValueError(f"Expected spatial attribution with 2, 3, or 4 dims, got shape {tuple(attr.shape)}")
    _tensor_stats(f"{label} channel-reduced attribution before ReLU", attr, verbose=verbose)
    relu_attr = torch.relu(attr)
    _tensor_stats(f"{label} attribution after ReLU before normalization", relu_attr, verbose=verbose)
    return normalize_heatmap(attr)


def token_attr_to_heatmap(attr: torch.Tensor, grid_size: Tuple[int, int]) -> np.ndarray:
    attr = attr.detach().float().cpu()
    grid_h, grid_w = grid_size
    expected_tokens = 1 + grid_h * grid_w

    if attr.ndim == 4:
        if attr.shape[0] == 1:
            attr = attr[0]
        else:
            raise ValueError(f"Expected batch size 1 for token GradCAM, got shape {tuple(attr.shape)}")

    if attr.ndim == 3:
        # Token outputs may appear as [batch, tokens, dim] or [tokens, batch, dim].
        if attr.shape[0] == 1 and attr.shape[1] == expected_tokens:
            tokens = attr[0]
            token_scores = tokens.mean(dim=-1)
        elif attr.shape[0] == expected_tokens and attr.shape[1] == 1:
            tokens = attr[:, 0, :]
            token_scores = tokens.mean(dim=-1)
        elif attr.shape[0] == 1 and attr.shape[2] == expected_tokens:
            tokens = attr[0]
            token_scores = tokens.mean(dim=0)
        else:
            raise ValueError(f"Expected {expected_tokens} tokens for grid {grid_h}x{grid_w}, got shape {tuple(attr.shape)}")
    elif attr.ndim == 2:
        if attr.shape[0] == expected_tokens:
            token_scores = attr.mean(dim=-1)
        elif attr.shape[1] == expected_tokens:
            token_scores = attr.mean(dim=0)
        else:
            raise ValueError(f"Expected {expected_tokens} tokens for grid {grid_h}x{grid_w}, got shape {tuple(attr.shape)}")
    elif attr.ndim == 1:
        if attr.numel() != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} tokens for grid {grid_h}x{grid_w}, got {attr.numel()}")
        token_scores = attr
    else:
        raise ValueError(f"Expected token attribution with 1, 2, 3, or 4 dims, got shape {tuple(attr.shape)}")

    patch_scores = token_scores[1:]
    if patch_scores.numel() != grid_h * grid_w:
        raise ValueError(f"Expected {grid_h * grid_w} patch tokens after dropping CLS, got {patch_scores.numel()}")
    return normalize_heatmap(patch_scores.reshape(grid_h, grid_w))


def manual_spatial_gradcam_heatmap(
    *,
    score_forward: Callable[[torch.Tensor], torch.Tensor],
    image_tensor: torch.Tensor,
    layer: torch.nn.Module,
    label: str,
    verbose: bool = True,
) -> np.ndarray:
    captured = {}

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured["activation"] = output
        if torch.is_tensor(output) and output.requires_grad:
            output.retain_grad()
            if verbose:
                print(f"{label} forward hook activation shape: {tuple(output.shape)}")
        else:
            print(f"WARNING: {label} forward hook activation has no gradient-bearing tensor output")

    handle = layer.register_forward_hook(hook)
    try:
        score = score_forward(image_tensor.requires_grad_(True))
        if verbose:
            print(f"{label} target logit: {float(score.detach()):.6g}")
        score.backward()
    finally:
        handle.remove()

    activation = captured.get("activation")
    if activation is None or activation.grad is None:
        raise RuntimeError(f"Missing layer activation or gradient for {label}")

    act = activation.detach().float()
    grad = activation.grad.detach().float()
    _tensor_stats(f"{label} captured activation", act, verbose=verbose)
    _tensor_stats(f"{label} captured activation gradient", grad, verbose=verbose)

    if act.ndim != 4:
        raise ValueError(f"Expected 4D spatial activation for {label}, got shape {tuple(act.shape)}")

    weights = grad.mean(dim=(2, 3), keepdim=True)
    attr = (weights * act).sum(dim=1)
    _tensor_stats(f"{label} weighted channel-sum attribution", attr, verbose=verbose)
    return spatial_attr_to_heatmap(attr, label=label, verbose=verbose)


def manual_patch_gradcam_heatmap(
    *,
    model: torch.nn.Module,
    score_forward: Callable[[torch.Tensor], torch.Tensor],
    image_tensor: torch.Tensor,
    verbose: bool = True,
) -> np.ndarray:
    return manual_spatial_gradcam_heatmap(
        score_forward=score_forward,
        image_tensor=image_tensor,
        layer=model.visual.conv1,
        label="Manual patch-embed GradCAM",
        verbose=verbose,
    )


def manual_token_gradcam_heatmap(
    *,
    model: torch.nn.Module,
    score_forward: Callable[[torch.Tensor], torch.Tensor],
    image_tensor: torch.Tensor,
    layer: torch.nn.Module,
    grid_size: Tuple[int, int],
    verbose: bool = True,
) -> np.ndarray:
    captured = {}

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured["activation"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(hook)
    try:
        score = score_forward(image_tensor.requires_grad_(True))
        score.backward()
    finally:
        handle.remove()

    activation = captured.get("activation")
    if activation is None or activation.grad is None:
        raise RuntimeError("Missing layer activation or gradient for manual ViT token GradCAM")

    if verbose:
        print(f"Manual ViT token GradCAM target logit: {float(score.detach()):.6g}")

    act = activation.detach().float()
    grad = activation.grad.detach().float()
    grid_h, grid_w = grid_size
    expected_tokens = 1 + grid_h * grid_w

    if act.ndim != 3:
        raise ValueError(f"Expected 3D token activation, got shape {tuple(act.shape)}")
    if act.shape[1] == expected_tokens:
        channel_dim = 2
    elif act.shape[0] == expected_tokens:
        channel_dim = 2
    else:
        raise ValueError(f"Expected {expected_tokens} tokens for grid {grid_h}x{grid_w}, got activation shape {tuple(act.shape)}")

    token_scores = (grad * act).sum(dim=channel_dim)
    if token_scores.ndim == 2:
        token_scores = token_scores[0] if token_scores.shape[0] == 1 else token_scores[:, 0]
    if torch.all(token_scores[1:] <= 0):
        token_scores = token_scores.abs()
    return token_attr_to_heatmap(token_scores, grid_size)


def vit_token_gradcam_heatmap(
    *,
    model: torch.nn.Module,
    score_forward: Callable[[torch.Tensor], torch.Tensor],
    image_tensor: torch.Tensor,
    layer_index: int = -2,
    verbose: bool = True,
) -> np.ndarray:
    layer = model.visual.transformer.resblocks[layer_index]
    grid_size = _grid_size(model)
    if verbose:
        print("ViT-token GradCAM using manual token GradCAM path.")
    return manual_token_gradcam_heatmap(
        model=model,
        score_forward=score_forward,
        image_tensor=image_tensor,
        layer=layer,
        grid_size=grid_size,
        verbose=verbose,
    )
