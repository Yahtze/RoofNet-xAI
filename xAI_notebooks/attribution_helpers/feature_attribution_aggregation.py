from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_RADIAL_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
)
DEFAULT_SQUARE_AREA_FRACS: tuple[float, ...] = (0.25, 0.5)


@dataclass(frozen=True)
class NormalizedHeatmap:
    heatmap: np.ndarray
    raw_sum: float
    raw_abs_sum: float
    raw_min: float
    raw_max: float
    negative_mass_ratio: float
    is_zero_sum: bool
    normalization_mode: str



def _as_float_array(heatmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {arr.shape}")
    return arr



def image_center_xy(shape: tuple[int, int]) -> tuple[float, float]:
    height, width = shape
    return ((width - 1) / 2.0, (height - 1) / 2.0)



def coordinate_grids(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    return yy, xx



def normalized_radial_distances(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = coordinate_grids(shape)
    center_x, center_y = image_center_xy(shape)
    distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    max_radius = float(distances.max())
    if max_radius <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    return distances / max_radius



def normalize_heatmap(
    heatmap: np.ndarray,
    *,
    mode: str = "l1_abs",
    eps: float = 1e-12,
) -> NormalizedHeatmap:
    arr = _as_float_array(heatmap)
    raw_sum = float(arr.sum())
    raw_abs_sum = float(np.abs(arr).sum())
    raw_min = float(arr.min())
    raw_max = float(arr.max())
    negative_mass = float(np.abs(arr[arr < 0]).sum())
    negative_mass_ratio = negative_mass / raw_abs_sum if raw_abs_sum > eps else 0.0

    if mode != "l1_abs":
        raise ValueError(f"Unsupported normalization mode: {mode!r}")

    if raw_abs_sum <= eps:
        return NormalizedHeatmap(
            heatmap=np.zeros_like(arr, dtype=np.float32),
            raw_sum=raw_sum,
            raw_abs_sum=raw_abs_sum,
            raw_min=raw_min,
            raw_max=raw_max,
            negative_mass_ratio=negative_mass_ratio,
            is_zero_sum=True,
            normalization_mode=mode,
        )

    norm_heatmap = np.abs(arr) / raw_abs_sum
    return NormalizedHeatmap(
        heatmap=norm_heatmap.astype(np.float32, copy=False),
        raw_sum=raw_sum,
        raw_abs_sum=raw_abs_sum,
        raw_min=raw_min,
        raw_max=raw_max,
        negative_mass_ratio=negative_mass_ratio,
        is_zero_sum=False,
        normalization_mode=mode,
    )



def make_radial_masks(
    shape: tuple[int, int],
    bands: Iterable[tuple[float, float]] = DEFAULT_RADIAL_BANDS,
) -> dict[str, np.ndarray]:
    distances = normalized_radial_distances(shape)
    masks: dict[str, np.ndarray] = {}
    for inner, outer in bands:
        if not (0.0 <= inner < outer <= 1.0):
            raise ValueError(f"Invalid radial band {(inner, outer)!r}")
        label = f"{int(round(inner * 100)):02d}_{int(round(outer * 100)):02d}"
        if inner == 0.0:
            mask = distances <= outer
        else:
            mask = (distances > inner) & (distances <= outer)
        masks[label] = mask
    return masks



def make_square_center_masks(
    shape: tuple[int, int],
    area_fracs: Iterable[float] = DEFAULT_SQUARE_AREA_FRACS,
) -> dict[str, np.ndarray]:
    height, width = shape
    yy, xx = coordinate_grids(shape)
    center_x, center_y = image_center_xy(shape)
    masks: dict[str, np.ndarray] = {}
    for area_frac in area_fracs:
        if not (0.0 < area_frac <= 1.0):
            raise ValueError(f"Invalid square area fraction: {area_frac!r}")
        side_frac = float(np.sqrt(area_frac))
        half_side_x = (width * side_frac) / 2.0
        half_side_y = (height * side_frac) / 2.0
        mask = (
            (np.abs(xx - center_x) <= half_side_x)
            & (np.abs(yy - center_y) <= half_side_y)
        )
        masks[f"{int(round(area_frac * 100)):02d}"] = mask
    return masks



def center_crop_mass_fraction(norm_heatmap: np.ndarray, mask: np.ndarray) -> float:
    arr = _as_float_array(norm_heatmap)
    if mask.shape != arr.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match heatmap shape {arr.shape}")
    return float(arr[mask].sum())



def attribution_weighted_centroid(norm_heatmap: np.ndarray) -> tuple[float, float]:
    arr = _as_float_array(norm_heatmap)
    total = float(arr.sum())
    if total <= 0.0:
        return image_center_xy(arr.shape)
    yy, xx = coordinate_grids(arr.shape)
    centroid_x = float((arr * xx).sum() / total)
    centroid_y = float((arr * yy).sum() / total)
    return centroid_x, centroid_y



def centroid_offset_from_center(norm_heatmap: np.ndarray) -> tuple[float, float, float, float]:
    arr = _as_float_array(norm_heatmap)
    centroid_x, centroid_y = attribution_weighted_centroid(arr)
    center_x, center_y = image_center_xy(arr.shape)
    offset_px = float(np.hypot(centroid_x - center_x, centroid_y - center_y))
    norm_distances = normalized_radial_distances(arr.shape)
    yy, xx = coordinate_grids(arr.shape)
    centroid_norm = float(
        np.interp(
            offset_px,
            np.sort(np.hypot(xx - center_x, yy - center_y).ravel()),
            np.sort(norm_distances.ravel()),
        )
    )
    return centroid_x, centroid_y, offset_px, centroid_norm



def radial_profile(
    norm_heatmap: np.ndarray,
    radial_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    arr = _as_float_array(norm_heatmap)
    return {label: float(arr[mask].sum()) for label, mask in radial_masks.items()}



def peak_offset_from_center(heatmap: np.ndarray) -> tuple[int, int, float, float]:
    arr = _as_float_array(heatmap)
    peak_y, peak_x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    center_x, center_y = image_center_xy(arr.shape)
    offset_px = float(np.hypot(peak_x - center_x, peak_y - center_y))
    distances = normalized_radial_distances(arr.shape)
    offset_norm = float(distances[peak_y, peak_x])
    return int(peak_x), int(peak_y), offset_px, offset_norm



def _square_mask_for_side_fraction(shape: tuple[int, int], side_frac: float) -> np.ndarray:
    if not (0.0 <= side_frac <= 1.0):
        raise ValueError(f"Invalid square side fraction: {side_frac!r}")
    height, width = shape
    yy, xx = coordinate_grids(shape)
    center_x, center_y = image_center_xy(shape)
    half_side_x = (width * side_frac) / 2.0
    half_side_y = (height * side_frac) / 2.0
    return (np.abs(xx - center_x) <= half_side_x) & (np.abs(yy - center_y) <= half_side_y)



def smallest_center_crop_for_mass(
    norm_heatmap: np.ndarray,
    *,
    target_mass: float = 0.5,
    mode: str = "square",
    num_thresholds: int = 512,
) -> float:
    arr = _as_float_array(norm_heatmap)
    if not (0.0 < target_mass <= 1.0):
        raise ValueError(f"Invalid target_mass: {target_mass!r}")
    if float(arr.sum()) <= 0.0:
        return 0.0

    if mode == "radial":
        distances = normalized_radial_distances(arr.shape)
        thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)
        for radius in thresholds:
            if float(arr[distances <= radius].sum()) >= target_mass:
                return float(radius)
        return 1.0

    if mode == "square":
        side_fracs = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)
        for side_frac in side_fracs:
            mask = _square_mask_for_side_fraction(arr.shape, float(side_frac))
            if float(arr[mask].sum()) >= target_mass:
                return float(side_frac)
        return 1.0

    raise ValueError(f"Unsupported mode: {mode!r}")



def compute_spatial_stats(
    heatmap: np.ndarray,
    *,
    method: str,
    image_id: str,
    square_area_fracs: Iterable[float] = DEFAULT_SQUARE_AREA_FRACS,
    radial_bands: Iterable[tuple[float, float]] = DEFAULT_RADIAL_BANDS,
    normalization_mode: str = "l1_abs",
) -> dict[str, float | str | bool]:
    raw_heatmap = _as_float_array(heatmap)
    normalized = normalize_heatmap(raw_heatmap, mode=normalization_mode)
    norm_heatmap = normalized.heatmap
    square_masks = make_square_center_masks(raw_heatmap.shape, square_area_fracs)
    radial_masks = make_radial_masks(raw_heatmap.shape, radial_bands)

    centroid_x, centroid_y, centroid_offset_px, centroid_offset_norm = centroid_offset_from_center(norm_heatmap)
    peak_x, peak_y, peak_offset_px, peak_offset_norm = peak_offset_from_center(raw_heatmap)
    radial_mass = radial_profile(norm_heatmap, radial_masks)
    radius_for_50_mass_square = smallest_center_crop_for_mass(norm_heatmap, target_mass=0.5, mode="square")
    radius_for_50_mass_radial = smallest_center_crop_for_mass(norm_heatmap, target_mass=0.5, mode="radial")

    stats: dict[str, float | str | bool] = {
        "method": method,
        "image_id": image_id,
        "normalization_mode": normalized.normalization_mode,
        "raw_sum": normalized.raw_sum,
        "raw_abs_sum": normalized.raw_abs_sum,
        "raw_min": normalized.raw_min,
        "raw_max": normalized.raw_max,
        "negative_mass_ratio": normalized.negative_mass_ratio,
        "is_zero_sum": normalized.is_zero_sum,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_offset_px": centroid_offset_px,
        "centroid_offset_norm": centroid_offset_norm,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "peak_offset_px": peak_offset_px,
        "peak_offset_norm": peak_offset_norm,
        "radius_for_50_mass_square": radius_for_50_mass_square,
        "radius_for_50_mass_radial": radius_for_50_mass_radial,
        "radius_50_gap": radius_for_50_mass_square - radius_for_50_mass_radial,
    }

    for area_key, mask in square_masks.items():
        stats[f"mass_center_{int(area_key)}_square"] = center_crop_mass_fraction(norm_heatmap, mask)

    for label, value in radial_mass.items():
        inner, outer = label.split("_")
        stats[f"radial_profile_{int(inner)}_{int(outer)}"] = value

    return stats



def aggregate_spatial_stats(df):
    import pandas as pd

    if df.empty:
        raise ValueError("Cannot aggregate empty spatial stats DataFrame")

    radial_cols = [col for col in df.columns if col.startswith("radial_profile_")]
    summary = {
        "num_images": int(len(df)),
        "zero_sum_images": int(df["is_zero_sum"].sum()) if "is_zero_sum" in df else 0,
        "median_mass_center_25_square": float(df["mass_center_25_square"].median()),
        "iqr_mass_center_25_square": float(df["mass_center_25_square"].quantile(0.75) - df["mass_center_25_square"].quantile(0.25)),
        "median_mass_center_50_square": float(df["mass_center_50_square"].median()),
        "iqr_mass_center_50_square": float(df["mass_center_50_square"].quantile(0.75) - df["mass_center_50_square"].quantile(0.25)),
        "median_radius_for_50_mass_square": float(df["radius_for_50_mass_square"].median()),
        "iqr_radius_for_50_mass_square": float(df["radius_for_50_mass_square"].quantile(0.75) - df["radius_for_50_mass_square"].quantile(0.25)),
        "median_radius_for_50_mass_radial": float(df["radius_for_50_mass_radial"].median()),
        "iqr_radius_for_50_mass_radial": float(df["radius_for_50_mass_radial"].quantile(0.75) - df["radius_for_50_mass_radial"].quantile(0.25)),
        "fraction_center25_over_50pct": float((df["mass_center_25_square"] > 0.5).mean()),
        "median_centroid_offset_norm": float(df["centroid_offset_norm"].median()),
        "median_peak_offset_norm": float(df["peak_offset_norm"].median()),
        "mean_negative_mass_ratio": float(df["negative_mass_ratio"].mean()),
        "median_radius_50_gap": float(df["radius_50_gap"].median()),
    }

    radial_profile_df = pd.DataFrame(
        {
            "ring": radial_cols,
            "mean": [float(df[col].mean()) for col in radial_cols],
            "std": [float(df[col].std(ddof=0)) for col in radial_cols],
        }
    )

    return {
        "summary": summary,
        "radial_profile": radial_profile_df,
    }
