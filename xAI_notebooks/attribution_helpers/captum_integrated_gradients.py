"""Captum Integrated Gradients helpers for RemoteCLIP input attribution."""

from __future__ import annotations

from typing import Callable, Literal, Type
import warnings

import numpy as np
import torch

Reduction = Literal["abs", "positive"]


def _warn_and_print(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    print(f"WARNING: {message}")


def normalize_heatmap(heatmap: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    heatmap = heatmap.detach().float().cpu()
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max() - heatmap.min()
    if float(denom) <= eps:
        _warn_and_print("Integrated Gradients heatmap is constant after normalization; returning zero heatmap.")
        return torch.zeros_like(heatmap).numpy().astype(np.float32)
    heatmap = heatmap / denom
    return heatmap.numpy().astype(np.float32)


def _single_image_attr(attr: torch.Tensor) -> torch.Tensor:
    attr = attr.detach().float().cpu()
    if attr.ndim == 4:
        if attr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for Integrated Gradients attribution, got shape {tuple(attr.shape)}")
        attr = attr[0]
    if attr.ndim != 3:
        raise ValueError(f"Expected Integrated Gradients attribution with shape [C, H, W] or [1, C, H, W], got {tuple(attr.shape)}")
    return attr


def print_attribution_diagnostics(
    attr: torch.Tensor,
    eps: float = 1e-12,
    *,
    verbose: bool = True,
) -> None:
    attr = _single_image_attr(attr)
    abs_map = attr.abs().sum(dim=0)
    positive_map = torch.relu(attr).sum(dim=0)
    negative_map = torch.relu(-attr).sum(dim=0)
    abs_mass = float(abs_map.sum())
    positive_mass = float(positive_map.sum())
    negative_mass = float(negative_map.sum())
    negative_to_abs_ratio = negative_mass / max(abs_mass, eps)
    abs_vs_positive_max_diff = float((abs_map - positive_map).abs().max())
    if verbose:
        print(
            "Integrated Gradients attribution diagnostics: "
            f"positive_mass={positive_mass:.6g}, negative_mass={negative_mass:.6g}, "
            f"negative_to_abs_ratio={negative_to_abs_ratio:.6g}, "
            f"abs_vs_positive_max_diff={abs_vs_positive_max_diff:.6g}"
        )


def input_attr_to_heatmap(attr: torch.Tensor, *, reduction: Reduction) -> np.ndarray:
    attr = _single_image_attr(attr)

    if reduction == "abs":
        heatmap = attr.abs().sum(dim=0)
    elif reduction == "positive":
        heatmap = torch.relu(attr).sum(dim=0)
    else:
        raise ValueError(f"Unknown Integrated Gradients reduction {reduction!r}; expected 'abs' or 'positive'")

    return normalize_heatmap(heatmap)


def integrated_gradients_heatmap(
    *,
    score_forward: Callable[[torch.Tensor], torch.Tensor],
    image_tensor: torch.Tensor,
    integrated_gradients_cls: Type,
    reduction: Reduction,
    n_steps: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    baseline = torch.zeros_like(image_tensor)
    ig = integrated_gradients_cls(score_forward)
    attr, delta = ig.attribute(
        image_tensor.requires_grad_(True),
        baselines=baseline,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    delta_tensor = delta.detach().float().cpu()
    if verbose:
        print(
            "Integrated Gradients convergence delta: "
            f"shape={tuple(delta_tensor.shape)}, max_abs={float(delta_tensor.abs().max()):.6g}, "
            f"mean_abs={float(delta_tensor.abs().mean()):.6g}"
        )
    print_attribution_diagnostics(attr, verbose=verbose)
    return input_attr_to_heatmap(attr, reduction=reduction)
