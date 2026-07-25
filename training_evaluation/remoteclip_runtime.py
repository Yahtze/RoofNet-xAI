"""
remoteclip_runtime.py
=====================
Single source of truth for instantiating RemoteCLIP ViT-L/14,
loading fine-tuned weights, returning validation preprocessing and tokenizer,
and fingerprinting preprocessing configuration.

Every consumer of the fine-tuned checkpoint should import from here
rather than constructing the model or transforms independently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

MODEL_NAME = "ViT-L-14"
PREPROCESSING_ID = "remoteclip-vit-l-14-default-val-v1"


def preprocessing_spec() -> dict[str, object]:
    """Return the canonical validation preprocessing specification."""
    return {
        "resize": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    }


def load_finetuned_remoteclip(
    weights_path: Path | str,
    device: str,
) -> tuple:
    """Load fine-tuned RemoteCLIP model, tokenizer, and validation preprocess.

    Constructs the base ViT-L-14 architecture without LAION pretrained weights,
    then loads the fine-tuned state dict. This mirrors the construction in
    remoteclip_finetune.py which loads RemoteCLIP base weights after building
    the architecture.

    Returns:
        (model, tokenizer, preprocess_val) tuple
    """
    import open_clip

    weights_path = Path(weights_path)

    # Build base architecture without pretrained weights
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)

    # Load fine-tuned state dict
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, tokenizer, preprocess_val


def checkpoint_sha256(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file using chunked reads."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_fingerprint(value: object) -> str:
    """Compute SHA-256 of a JSON-serializable value using canonical JSON."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
