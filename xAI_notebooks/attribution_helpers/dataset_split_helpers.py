from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import unicodedata

import pandas as pd

VALID_SPLITS = ("train", "val", "holdout", "all")


def _normalize_filename(value: object) -> str:
    return unicodedata.normalize("NFC", str(value).strip())


def normalize_split_name(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    return normalized


def _normalized_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if "filename" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'filename' column.")
    if "split" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'split' column.")

    normalized = df.copy()
    normalized["filename"] = normalized["filename"].fillna("").map(_normalize_filename)
    normalized["split"] = normalized["split"].fillna("unknown").astype(str).str.strip().str.lower()
    normalized = normalized[normalized["filename"] != ""].copy()
    return normalized


def _build_image_lookup(image_dir: Path, image_exts: Sequence[str]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    normalized_exts = {ext.lower() for ext in image_exts}

    for path in sorted(image_dir.rglob("*")):
        normalized_name = _normalize_filename(path.name)
        if path.is_file() and path.suffix.lower() in normalized_exts and normalized_name not in lookup:
            lookup[normalized_name] = path

    return lookup


def collect_split_image_paths(
    image_dir: Path,
    metadata_df: pd.DataFrame,
    split: str,
    image_exts: Sequence[str],
) -> Tuple[List[Path], Dict[str, object]]:
    requested_split = normalize_split_name(split)
    normalized_df = _normalized_metadata(metadata_df)

    available_split_counts = {
        str(name): int(count)
        for name, count in normalized_df["split"].value_counts().items()
        if str(name) != "unknown"
    }

    if requested_split == "all":
        filtered_df = normalized_df[normalized_df["split"].isin(["train", "val", "holdout"])]
    else:
        filtered_df = normalized_df[normalized_df["split"] == requested_split]

    print(
        f"Batch split filter: requested={requested_split}, "
        f"matched_metadata_rows={len(filtered_df)}, available_counts={available_split_counts}"
    )

    image_lookup = _build_image_lookup(Path(image_dir), image_exts)
    selected_paths: List[Path] = []
    missing_files: List[str] = []

    for filename in filtered_df["filename"].tolist():
        image_path = image_lookup.get(filename)
        if image_path is None:
            missing_files.append(filename)
            continue
        selected_paths.append(image_path)

    diagnostics: Dict[str, object] = {
        "requested_split": requested_split,
        "matched_rows": int(len(filtered_df)),
        "selected_images": int(len(selected_paths)),
        "missing_files": missing_files,
        "available_split_counts": available_split_counts,
    }

    print(
        f"Selected {len(selected_paths)} images from {image_dir}. "
        f"Missing files after metadata join: {len(missing_files)}"
    )
    if missing_files:
        print("First missing filenames:", missing_files[:10])

    return selected_paths, diagnostics


def write_split_helper_csvs(metadata_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    normalized_df = _normalized_metadata(metadata_df)
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    split_frames = {
        "train": normalized_df[normalized_df["split"] == "train"].copy(),
        "val": normalized_df[normalized_df["split"] == "val"].copy(),
        "holdout": normalized_df[normalized_df["split"] == "holdout"].copy(),
        "all": normalized_df[normalized_df["split"].isin(["train", "val", "holdout"])].copy(),
    }

    outputs: Dict[str, Path] = {}
    for split_name, frame in split_frames.items():
        out_path = export_dir / f"roofnet_{split_name}_images.csv"
        frame.sort_values(["filename"]).to_csv(out_path, index=False)
        outputs[split_name] = out_path

    print("Exported split helper CSV artifacts:")
    for split_name, out_path in outputs.items():
        print(f"  {split_name}: {out_path}")

    return outputs
