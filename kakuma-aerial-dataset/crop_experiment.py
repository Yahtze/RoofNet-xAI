"""
crop_experiment.py
==================
Pure reusable functions for the alpha-derived crop inference pipeline.

Provides:
- Collision-safe sample ID generation from source relative paths
- Alpha-bounding-box crop creation with padding
- Square padding via edge replication
- JPEG writing with content-match guard
- Atomic CSV manifest writing
- Metadata and fingerprint helpers
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, ImageOps


def make_sample_id(source_relpath: str) -> str:
    """Generate a collision-safe sample ID from a source relative path.

    Uses the stem of the path plus a SHA-256 prefix of the normalized path
    to ensure distinct IDs even when the same basename appears in different
    class directories.
    """
    normalized = source_relpath.replace("\\", "/")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{Path(normalized).stem}--{digest}"


def crop_from_alpha(rgba: Image.Image, padding: int = 8) -> Image.Image:
    """Crop an RGBA image to the alpha bounding box expanded by *padding* pixels.

    The returned image is RGB (alpha channel discarded after bounds calculation).
    Raises ValueError if the alpha footprint is entirely empty.
    """
    pixels = np.asarray(rgba.convert("RGBA"))
    mask = pixels[..., 3] > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty alpha footprint")

    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(pixels.shape[1], int(xs.max()) + padding + 1)
    bottom = min(pixels.shape[0], int(ys.max()) + padding + 1)

    return Image.fromarray(pixels[..., :3], "RGB").crop((left, top, right, bottom))


def square_pad_rgb(image: Image.Image) -> Image.Image:
    """Pad an RGB image to a square using numpy edge-mode replication.

    The shorter dimension is expanded so width == height. Padding is applied
    symmetrically (extra pixel on the right/bottom if odd difference).
    """
    w, h = image.size
    if w == h:
        return image.copy()

    arr = np.asarray(image)
    diff = abs(w - h)
    pad_before = diff // 2
    pad_after = diff - pad_before

    if w < h:
        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
    else:
        pad_width = ((pad_before, pad_after), (0, 0), (0, 0))

    padded = np.pad(arr, pad_width, mode="edge")
    return Image.fromarray(padded, "RGB")


def square_pad_rgb_edge(image: Image.Image) -> Image.Image:
    """Pad an RGB image to a square using numpy edge-mode replication.

    This is an alternative to square_pad_rgb that uses numpy.pad with mode='edge'.
    """
    w, h = image.size
    if w == h:
        return image.copy()

    arr = np.asarray(image)
    diff = abs(w - h)
    pad_before = diff // 2
    pad_after = diff - pad_before

    if w < h:
        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
    else:
        pad_width = ((pad_before, pad_after), (0, 0), (0, 0))

    padded = np.pad(arr, pad_width, mode="edge")
    return Image.fromarray(padded, "RGB")


def write_prepared_jpeg(
    image: Image.Image,
    output_dir: Path,
    sample_id: str,
    quality: int = 95,
) -> Path:
    """Write an RGB JPEG to *output_dir*/<sample_id>.jpg.

    Refuses to overwrite an existing file whose image bytes do not match
    the requested output. Returns the Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sample_id}.jpg"
    output_path = output_dir / filename

    if output_path.exists():
        # Check if existing file matches by re-encoding and comparing
        existing = Image.open(output_path).convert("RGB")
        if existing.size == image.size and existing.mode == "RGB":
            # Compare pixel data
            existing_arr = np.asarray(existing)
            new_arr = np.asarray(image.convert("RGB"))
            if np.array_equal(existing_arr, new_arr):
                return output_path
        raise FileExistsError(
            f"Refusing to overwrite {output_path}: existing image bytes differ"
        )

    rgb_image = image.convert("RGB")
    rgb_image.save(str(output_path), "JPEG", quality=quality)
    return output_path


MANIFEST_COLUMNS = [
    "sample_id",
    "source_relpath",
    "prepared_filename",
    "gt_class",
    "status",
    "error",
]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a CSV manifest atomically with the six required columns.

    Each row dict must contain the six keys listed in MANIFEST_COLUMNS.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [",".join(MANIFEST_COLUMNS)]
    for row in rows:
        values = []
        for col in MANIFEST_COLUMNS:
            val = str(row.get(col, ""))
            # Escape commas and quotes in values
            if "," in val or '"' in val or "\n" in val:
                val = '"' + val.replace('"', '""') + '"'
            values.append(val)
        lines.append(",".join(values))

    content = "\n".join(lines) + "\n"

    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".csv"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
