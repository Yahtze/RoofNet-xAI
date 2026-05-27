from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


VIT_L_14_PATCH_GRID_SIZE = 16
VIT_L_14_TOKEN_COUNT = 1 + (VIT_L_14_PATCH_GRID_SIZE * VIT_L_14_PATCH_GRID_SIZE)


@dataclass
class AttentionCapture:
    attention: torch.Tensor | None = None
    gradient: torch.Tensor | None = None


@dataclass
class TransformerExplainabilitySession:
    model: torch.nn.Module
    captures: Dict[int, AttentionCapture] = field(default_factory=dict)
    _original_attention_methods: Dict[int, object] = field(default_factory=dict)

    def clear(self) -> None:
        self.captures.clear()

    def install(self) -> None:
        self.clear()
        for layer_idx, block in enumerate(self.model.visual.transformer.resblocks):
            self._original_attention_methods[layer_idx] = block.attention
            self.captures[layer_idx] = AttentionCapture()
            block.attention = types.MethodType(self._build_attention_wrapper(layer_idx), block)

    def remove(self) -> None:
        for layer_idx, block in enumerate(self.model.visual.transformer.resblocks):
            original = self._original_attention_methods.get(layer_idx)
            if original is not None:
                block.attention = original
        self._original_attention_methods.clear()

    def _build_attention_wrapper(self, layer_idx: int):
        session = self

        def attention_wrapper(block, q_x, k_x=None, v_x=None, attn_mask=None):
            k_x = k_x if k_x is not None else q_x
            v_x = v_x if v_x is not None else q_x
            attn_mask_local = attn_mask.to(q_x.dtype) if attn_mask is not None else None
            attn_output, attn_weights = block.attn(
                q_x,
                k_x,
                v_x,
                need_weights=True,
                average_attn_weights=False,
                attn_mask=attn_mask_local,
            )
            session.captures[layer_idx].attention = attn_weights
            attn_weights.register_hook(session._make_gradient_hook(layer_idx))
            return attn_output

        return attention_wrapper

    def _make_gradient_hook(self, layer_idx: int):
        def hook(grad: torch.Tensor) -> None:
            self.captures[layer_idx].gradient = grad.detach()

        return hook

    def ordered_attention_and_gradients(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(len(self.model.visual.transformer.resblocks)):
            capture = self.captures.get(layer_idx)
            if capture is None or capture.attention is None or capture.gradient is None:
                raise RuntimeError(f"Missing attention capture for visual transformer layer {layer_idx}")
            pairs.append((capture.attention.detach(), capture.gradient.detach()))
        return pairs


def get_visual_patch_grid_size(model: torch.nn.Module) -> int:
    grid_size = getattr(model.visual, "grid_size", None)
    if grid_size is None:
        raise AttributeError("model.visual.grid_size is required for transformer explainability")
    if isinstance(grid_size, tuple):
        if grid_size[0] != grid_size[1]:
            raise ValueError(f"Expected square patch grid, got {grid_size}")
        return int(grid_size[0])
    return int(grid_size)


def gradient_weighted_attention(attention: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    if attention.ndim != 4 or gradients.ndim != 4:
        raise ValueError("Expected attention and gradients with shape [batch, heads, seq, seq]")
    if attention.shape != gradients.shape:
        raise ValueError(f"Attention/gradient shape mismatch: {attention.shape} vs {gradients.shape}")

    attention_mean = attention.mean(dim=1)[0]
    gradient_mean = gradients.mean(dim=1)[0]
    relevance = torch.relu(attention_mean * gradient_mean)
    relevance = normalize_rows(relevance)
    identity = torch.eye(relevance.size(-1), device=relevance.device, dtype=relevance.dtype)
    return relevance + identity


def normalize_rows(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    row_sums = matrix.sum(dim=-1, keepdim=True)
    return torch.where(row_sums > eps, matrix / row_sums.clamp_min(eps), matrix)


def rollout_cls_patch_relevance(layer_relevances: Sequence[torch.Tensor]) -> torch.Tensor:
    if not layer_relevances:
        raise ValueError("Need at least one layer relevance matrix for rollout")

    rollout = torch.eye(
        layer_relevances[0].size(-1),
        device=layer_relevances[0].device,
        dtype=layer_relevances[0].dtype,
    )
    for layer_relevance in layer_relevances:
        rollout = rollout @ layer_relevance
    return rollout[0, 1:]


def patch_relevance_to_heatmap(
    patch_relevance: torch.Tensor,
    image_size: Tuple[int, int] = (224, 224),
    patch_grid_size: int = VIT_L_14_PATCH_GRID_SIZE,
) -> np.ndarray:
    expected_patches = patch_grid_size * patch_grid_size
    if patch_relevance.numel() != expected_patches:
        raise ValueError(
            f"Expected {expected_patches} patch scores for grid {patch_grid_size}x{patch_grid_size}, got {patch_relevance.numel()}"
        )

    grid = patch_relevance.reshape(1, 1, patch_grid_size, patch_grid_size).float()
    upsampled = F.interpolate(grid, size=image_size, mode="bilinear", align_corners=False)[0, 0]
    upsampled = upsampled - upsampled.min()
    upsampled = upsampled / upsampled.max().clamp_min(1e-6)
    return upsampled.detach().cpu().numpy()


def collect_layer_relevances(attention_gradient_pairs: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
    return [gradient_weighted_attention(attention, gradient) for attention, gradient in attention_gradient_pairs]


def compute_similarity_score(
    model: torch.nn.Module,
    tokenizer,
    image_tensor: torch.Tensor,
    prompts: Sequence[str],
    target_idx: int,
) -> torch.Tensor:
    tokenized = tokenizer(list(prompts)).to(image_tensor.device)
    image_features = model.encode_image(image_tensor)
    with torch.no_grad():
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_features @ text_features.T
    return logits[:, target_idx].sum()


def transformer_explainability(
    model: torch.nn.Module,
    tokenizer,
    image_tensor: torch.Tensor,
    prompts: Sequence[str],
    target_idx: int,
    image_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    patch_grid_size = get_visual_patch_grid_size(model)
    expected_token_count = 1 + patch_grid_size * patch_grid_size
    session = TransformerExplainabilitySession(model)

    image_for_grad = image_tensor.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    session.install()
    try:
        with torch.enable_grad():
            score = compute_similarity_score(model, tokenizer, image_for_grad, prompts, target_idx)
            score.backward()
        layer_relevances = collect_layer_relevances(session.ordered_attention_and_gradients())
        patch_relevance = rollout_cls_patch_relevance(layer_relevances)
        if patch_relevance.numel() != expected_token_count - 1:
            raise ValueError(
                f"Expected {expected_token_count - 1} patch tokens from visual transformer, got {patch_relevance.numel()}"
            )
        return patch_relevance_to_heatmap(
            patch_relevance,
            image_size=image_size,
            patch_grid_size=patch_grid_size,
        )
    finally:
        session.remove()
        model.zero_grad(set_to_none=True)
