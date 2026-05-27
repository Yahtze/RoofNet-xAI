"""RISE (Randomized Input Sampling for Explanation) helpers for RemoteCLIP."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _validate_single_image(image_tensor: torch.Tensor) -> tuple[int, int]:
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError(
            "Expected image_tensor shape [1, C, H, W] for RISE, "
            f"got {tuple(image_tensor.shape)}"
        )
    if image_tensor.shape[2] <= 0 or image_tensor.shape[3] <= 0:
        raise ValueError(f"Expected positive image height/width for RISE, got {tuple(image_tensor.shape)}")
    return int(image_tensor.shape[2]), int(image_tensor.shape[3])


def _validate_parameters(num_masks: int, mask_grid_size: int, p_save: float, batch_size: int) -> None:
    if num_masks <= 0:
        raise ValueError(f"num_masks must be positive, got {num_masks}")
    if mask_grid_size <= 0:
        raise ValueError(f"mask_grid_size must be positive, got {mask_grid_size}")
    if not 0.0 < p_save <= 1.0:
        raise ValueError(f"p_save must be in (0, 1], got {p_save}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")


def generate_rise_masks(
    *,
    num_masks: int,
    image_size: tuple[int, int],
    mask_grid_size: int,
    p_save: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate shared-channel soft RISE masks with shape [N, 1, H, W]."""
    height, width = image_size
    low_res = torch.rand(
        (num_masks, 1, mask_grid_size, mask_grid_size),
        device=device,
        generator=generator,
    ) < p_save
    masks = F.interpolate(
        low_res.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return masks.clamp_(0.0, 1.0)


def _coerce_scores(scores: torch.Tensor, expected_batch: int) -> torch.Tensor:
    scores = scores.detach().float().flatten()
    if scores.numel() != expected_batch:
        raise ValueError(
            f"score_forward must return one scalar per masked image; expected {expected_batch}, got {scores.numel()}"
        )
    return scores


def _format_diagnostics(diagnostics: dict[str, float | int]) -> str:
    return (
        "RISE diagnostics: "
        f"num_masks={diagnostics['num_masks']}, "
        f"mask_grid_size={diagnostics['mask_grid_size']}, "
        f"p_save={diagnostics['p_save']:.6g}, "
        f"batch_size={diagnostics['batch_size']}, "
        f"score_mean={diagnostics['score_mean']:.6g}, "
        f"score_std={diagnostics['score_std']:.6g}, "
        f"score_min={diagnostics['score_min']:.6g}, "
        f"score_max={diagnostics['score_max']:.6g}, "
        f"mask_mean={diagnostics['mask_mean']:.6g}, "
        f"mask_variance={diagnostics['mask_variance']:.6g}, "
        f"generation_time_ms={diagnostics['generation_time_ms']:.3f}, "
        f"forward_time_ms={diagnostics['forward_time_ms']:.3f}"
    )


def rise_heatmap(
    *,
    score_forward: Callable[[torch.Tensor], torch.Tensor],
    image_tensor: torch.Tensor,
    num_masks: int = 512,
    mask_grid_size: int = 12,
    p_save: float = 0.5,
    batch_size: int = 32,
    mask_device: Optional[torch.device | str] = None,
    return_diagnostics: bool = False,
    generator: Optional[torch.Generator] = None,
    masks: Optional[torch.Tensor] = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int]]:
    """Compute raw RISE heatmap using mean-baseline normalized-space masking.

    Hidden pixels become zero in CLIP-normalized tensor space. The estimator uses
    actual per-pixel mask-count normalization: sum(score_i * mask_i) / sum(mask_i).
    """
    height, width = _validate_single_image(image_tensor)
    _validate_parameters(num_masks, mask_grid_size, p_save, batch_size)

    image_device = image_tensor.device
    resolved_mask_device = torch.device(mask_device) if mask_device is not None else image_device

    generation_start = time.perf_counter()
    if masks is None:
        masks = generate_rise_masks(
            num_masks=num_masks,
            image_size=(height, width),
            mask_grid_size=mask_grid_size,
            p_save=p_save,
            device=resolved_mask_device,
            generator=generator,
        )
    else:
        masks = masks.detach().float().to(resolved_mask_device)
        if masks.ndim != 4 or masks.shape[1] != 1 or masks.shape[2:] != (height, width):
            raise ValueError(
                "Expected masks shape [N, 1, H, W] matching image size, "
                f"got {tuple(masks.shape)} for image {(height, width)}"
            )
        num_masks = int(masks.shape[0])
    generation_time_ms = (time.perf_counter() - generation_start) * 1000.0

    weighted_sum = torch.zeros((height, width), dtype=torch.float32, device=resolved_mask_device)
    mask_sum = torch.zeros((height, width), dtype=torch.float32, device=resolved_mask_device)
    score_chunks = []

    forward_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, num_masks, batch_size):
            batch_masks = masks[start : start + batch_size]
            masked_batch = image_tensor.to(resolved_mask_device) * batch_masks
            if masked_batch.device != image_device:
                masked_batch = masked_batch.to(image_device)
            scores = _coerce_scores(score_forward(masked_batch), masked_batch.shape[0]).to(resolved_mask_device)
            score_chunks.append(scores.cpu())
            weighted_sum += (batch_masks[:, 0] * scores.view(-1, 1, 1)).sum(dim=0)
            mask_sum += batch_masks[:, 0].sum(dim=0)
    forward_time_ms = (time.perf_counter() - forward_start) * 1000.0

    heatmap = weighted_sum / mask_sum.clamp_min(1e-12)
    heatmap_np = heatmap.detach().cpu().numpy().astype(np.float32)

    if not return_diagnostics:
        return heatmap_np

    all_scores = torch.cat(score_chunks) if score_chunks else torch.empty(0)
    diagnostics: dict[str, float | int] = {
        "num_masks": int(num_masks),
        "mask_grid_size": int(mask_grid_size),
        "p_save": float(p_save),
        "batch_size": int(batch_size),
        "score_mean": float(all_scores.mean()) if all_scores.numel() else 0.0,
        "score_std": float(all_scores.std(unbiased=False)) if all_scores.numel() else 0.0,
        "score_min": float(all_scores.min()) if all_scores.numel() else 0.0,
        "score_max": float(all_scores.max()) if all_scores.numel() else 0.0,
        "mask_mean": float(masks.float().mean().detach().cpu()),
        "mask_variance": float(masks.float().var(unbiased=False).detach().cpu()),
        "generation_time_ms": float(generation_time_ms),
        "forward_time_ms": float(forward_time_ms),
    }
    print(_format_diagnostics(diagnostics))
    return heatmap_np, diagnostics
