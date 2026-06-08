from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GroundingPrediction:
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    phrases: list[str]
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class SamMaskPrediction:
    masks: list[np.ndarray]
    scores: list[float]
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class DependencyStatus:
    ok: bool
    message: str


def _as_existing_path(path_like: str | Path, *, label: str) -> Path:
    path = Path(path_like).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def check_grounding_sam_dependencies() -> dict[str, DependencyStatus]:
    statuses: dict[str, DependencyStatus] = {}
    try:
        import groundingdino  # noqa: F401

        statuses["groundingdino"] = DependencyStatus(True, "groundingdino import OK")
    except Exception as exc:
        statuses["groundingdino"] = DependencyStatus(False, f"groundingdino import failed: {exc!r}")
    try:
        import segment_anything  # noqa: F401

        statuses["segment_anything"] = DependencyStatus(True, "segment_anything import OK")
    except Exception as exc:
        statuses["segment_anything"] = DependencyStatus(False, f"segment_anything import failed: {exc!r}")
    return statuses


def load_groundingdino_model(config_path: str | Path, weights_path: str | Path, *, device: str = "cpu") -> Any:
    config_path = _as_existing_path(config_path, label="GroundingDINO config")
    weights_path = _as_existing_path(weights_path, label="GroundingDINO weights")
    from groundingdino.util.inference import load_model

    return load_model(str(config_path), str(weights_path), device=device)


def load_sam_predictor(weights_path: str | Path, *, model_type: str = "vit_h", device: str = "cpu") -> Any:
    weights_path = _as_existing_path(weights_path, label="SAM weights")
    from segment_anything import SamPredictor, sam_model_registry

    sam_model = sam_model_registry[model_type](checkpoint=str(weights_path))
    sam_model.to(device=device)
    return SamPredictor(sam_model)


def run_groundingdino_inference(
    model: Any,
    image: Image.Image,
    *,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str = "cpu",
) -> GroundingPrediction:
    from groundingdino.util.inference import predict

    image_rgb = image.convert("RGB")
    image_np = np.asarray(image_rgb)
    boxes, logits, phrases = predict(
        model=model,
        image=image_np,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    height, width = image_np.shape[:2]
    if boxes is None or len(boxes) == 0:
        return GroundingPrediction(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            confidences=np.zeros((0,), dtype=np.float32),
            phrases=[],
            image_shape=(height, width),
        )
    boxes = np.asarray(boxes, dtype=np.float32)
    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, [0, 2]] *= width
    boxes_xyxy[:, [1, 3]] *= height
    boxes_xyxy[:, 0] -= boxes_xyxy[:, 2] / 2.0
    boxes_xyxy[:, 1] -= boxes_xyxy[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]
    return GroundingPrediction(
        boxes_xyxy=boxes_xyxy,
        confidences=np.asarray(logits, dtype=np.float32),
        phrases=[str(p) for p in phrases],
        image_shape=(height, width),
    )


def run_sam_box_refinement(
    predictor: Any,
    image: Image.Image,
    boxes_xyxy: Sequence[Sequence[float]],
) -> SamMaskPrediction:
    image_rgb = image.convert("RGB")
    image_np = np.asarray(image_rgb)
    predictor.set_image(image_np)
    masks: list[np.ndarray] = []
    scores: list[float] = []
    for box in boxes_xyxy:
        box_array = np.asarray(box, dtype=np.float32)
        mask_set, score_set, _ = predictor.predict(box=box_array, multimask_output=True)
        best_idx = int(np.argmax(score_set))
        masks.append(np.asarray(mask_set[best_idx], dtype=bool))
        scores.append(float(score_set[best_idx]))
    return SamMaskPrediction(masks=masks, scores=scores, image_shape=image_np.shape[:2])
