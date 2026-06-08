from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


MANIFEST_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_job_id(image_id: str, method_name: str) -> str:
    return f"{Path(image_id).name}__{method_name}"


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("manifest_version", MANIFEST_VERSION)
        data.setdefault("created_at", utc_now_iso())
        data.setdefault("updated_at", data["created_at"])
        data.setdefault("jobs", {})
        return data
    now = utc_now_iso()
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": now,
        "updated_at": now,
        "jobs": {},
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = utc_now_iso()
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest)


def upsert_job(
    manifest: dict[str, Any],
    *,
    job_id: str,
    image_id: str,
    method_name: str,
    output_path: Path,
) -> dict[str, Any]:
    jobs = manifest.setdefault("jobs", {})
    now = utc_now_iso()
    job = jobs.setdefault(
        job_id,
        {
            "job_id": job_id,
            "image_id": image_id,
            "method_name": method_name,
            "output_path": str(output_path),
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "error": None,
        },
    )
    job["image_id"] = image_id
    job["method_name"] = method_name
    job["output_path"] = str(output_path)
    job["updated_at"] = now
    return job


def resolve_job_action(manifest: dict[str, Any], *, job_id: str, output_path: Path) -> str:
    job = manifest.get("jobs", {}).get(job_id)
    if job and job.get("status") == "done" and output_path.exists():
        return "skip"
    return "run"


def _mark_job_status(manifest: dict[str, Any], job_id: str, status: str, error: str | None = None) -> None:
    job = manifest["jobs"][job_id]
    job["status"] = status
    job["updated_at"] = utc_now_iso()
    job["error"] = error
    if status == "running":
        job["started_at"] = job["updated_at"]
    if status in {"done", "failed"}:
        job["finished_at"] = job["updated_at"]


def mark_job_running(manifest: dict[str, Any], job_id: str) -> None:
    _mark_job_status(manifest, job_id, "running")


def mark_job_done(manifest: dict[str, Any], job_id: str) -> None:
    _mark_job_status(manifest, job_id, "done")


def mark_job_failed(manifest: dict[str, Any], job_id: str, error: str) -> None:
    _mark_job_status(manifest, job_id, "failed", error=error)


def append_stats_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def atomic_replace_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(destination_path)


def summarize_jobs(manifest: dict[str, Any]) -> dict[str, int]:
    summary = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for job in manifest.get("jobs", {}).values():
        status = job.get("status", "pending")
        summary.setdefault(status, 0)
        summary[status] += 1
    return summary
