"""
run_batch_experiments.py
=======================
Orchestrates the full 400-run experiment loop:

    4 modes  ×  100 random seeds  =  400 OpenQuake scenario_damage runs

For each (mode, seed) pair:
  1. Runs build_exposure_model.py {mode} {seed}          → writes files to
       <OQ_BASE>/{mode}/seed_{seed}/
  2. Writes a job.ini into that directory, pointing at the freshly-written
       exposure, fragility, site-model, and GMF files.
  3. Invokes `oq engine --run job.ini` and captures the calc_id from stdout.
  4. Exports the damage-by-asset CSV with `oq export damages-rlzs {calc_id}`
       and copies it to <RESULTS_DIR>/{mode}_seed_{seed}.csv.

Usage:
    python run_batch_experiments.py [--modes roofnet benchmark unreinforced reinforced]
                                    [--seeds 100]
                                    [--workers 1]
                                    [--dry-run]

Prerequisites:
  • patch_build_exposure_model.py has already been applied once.
  • `oq` CLI is on your PATH  (OpenQuake Engine 3.25.1).
  • All data paths inside build_exposure_model.py are correct.
"""

import argparse
import os
import re
import subprocess
import platform
import psutil
import sys
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

# Absolute path to the (patched) exposure-builder script
BUILD_SCRIPT = ("./build_exposure_model.py")

# Root directory where mode/seed sub-trees live
OQ_BASE = "./openquake_results"

# Where all exported result CSVs land (flat layout: {mode}_seed_{seed}.csv)
RESULTS_DIR = os.path.join(OQ_BASE, "batch_results")

# Template job.ini — values are substituted per run
JOB_INI_TEMPLATE = """\
[general]
description = {mode} seed {seed} — scenario_damage
calculation_mode = scenario_damage
random_seed = {seed}

[site_params]
site_model_file = site_model.xml
reference_vs30_value = 760

[exposure]
exposure_file = exposure_metadata.xml

[hazard]
gmfs_file = ground_motion_grid.csv
number_of_ground_motion_fields = 100

[fragility]
structural_fragility_file = structural_fragility_model.xml
"""

ALL_MODES = ["roofnet", "benchmark", "unreinforced", "reinforced"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def run_dir(mode: str, seed: int) -> Path:
    return Path(OQ_BASE) / mode / f"seed_{seed}"


def write_job_ini(mode: str, seed: int) -> Path:
    """Write a job.ini into the run directory and return its path."""
    ini_content = JOB_INI_TEMPLATE.format(mode=mode, seed=seed)
    ini_path = run_dir(mode, seed) / "job.ini"
    ini_path.write_text(ini_content, encoding="utf-8")
    return ini_path


def run_build_script(mode: str, seed: int, dry_run: bool = False) -> bool:
    """
    Execute build_exposure_model.py for the given (mode, seed).
    Returns True on success.
    """
    cmd = [sys.executable, BUILD_SCRIPT, mode, str(seed)]
    label = f"[{mode}/seed_{seed}] build"
    if dry_run:
        print(f"  DRY-RUN {label}: {' '.join(cmd)}")
        # Create dummy output dir so later steps don't fail
        run_dir(mode, seed).mkdir(parents=True, exist_ok=True)
        return True

    print(f"  {label} ...", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ERROR {label} (rc={result.returncode}, {elapsed:.1f}s)")
        print(result.stderr[-2000:])        # last 2 kB of stderr
        return False

    print(f"  OK    {label} ({elapsed:.1f}s)")
    return True


def run_openquake(ini_path: Path, dry_run: bool = False) -> int | None:
    """
    Run `oq engine --run <ini_path>` and return the integer calc_id.
    Returns None on failure.
    """
    cmd = ["oq", "engine", "--run", str(ini_path)]
    label = f"[oq] {ini_path.parent.parent.name}/{ini_path.parent.name}"

    if dry_run:
        print(f"  DRY-RUN {label}: {' '.join(cmd)}")
        return 999_999      # placeholder calc_id

    print(f"  {label} ...", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ini_path.parent),   # OQ resolves relative file paths from cwd
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ERROR {label} (rc={result.returncode}, {elapsed:.1f}s)")
        print(result.stderr[-2000:])
        return None

    # Parse calc_id from stdout.
    # OQ emits a line like:
    #   "Calculation 42 finished correctly in ..."
    # Parse calc_id from stdout/stderr.
    # Legacy check (Pre OQ 3.25)
    match = re.search(r"Calculation (\d+) finished correctly", result.stdout)
    if not match:
        match = re.search(r"Calculation (\d+) finished correctly", result.stderr)

    # --- NEW OQ 3.25+ PARSING LOGIC ---
    if not match:
        # 1st fallback: Look for the final HDF5 database save path
        match = re.search(r"calc_(\d+)\.hdf5", result.stderr)
        
    if not match:
        # 2nd fallback: Extract from the standard OQ log prefix (e.g. "... #85 INFO]")
        match = re.search(r"#(\d+) (?:INFO|WARNING|ERROR)\]", result.stderr)
    # ----------------------------------

    if not match:
        print(f"\n--- DEBUG: STDOUT ---\n{result.stdout}")
        print(f"--- DEBUG: STDERR ---\n{result.stderr}\n---------------------\n")
        
        print(f"  WARNING {label}: calc_id not found in OQ output — "
              f"trying `oq engine --list-calculations`")
        calc_id = get_latest_calc_id()
    else:
        calc_id = int(match.group(1))

    print(f"  OK    {label} calc_id={calc_id} ({elapsed:.1f}s)")
    return calc_id


def get_latest_calc_id() -> int | None:
    """Fall-back: query `oq engine --list-calculations` for the latest ID."""
    result = subprocess.run(
        ["oq", "engine", "--list-calculations"],
        capture_output=True, text=True,
    )
    # Lines look like: "42 | scenario_damage | ..."
    ids = re.findall(r"^\s*(\d+)\s*\|", result.stdout, re.MULTILINE)
    return int(ids[-1]) if ids else None


def export_results(calc_id: int, mode: str, seed: int,
                   dry_run: bool = False) -> Path | None:
    """
    Export damages-rlzs for calc_id into a temporary directory,
    then copy the CSV to RESULTS_DIR/{mode}_seed_{seed}.csv.
    Returns the destination path, or None on failure.
    """
    dest = Path(RESULTS_DIR) / f"{mode}_seed_{seed}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        dest.write_text("# dry-run placeholder\n", encoding="utf-8")
        print(f"  DRY-RUN export → {dest}")
        return dest

    # Export to a temp sub-dir so we can glob for the CSV
    export_dir = run_dir(mode, seed) / "oq_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "oq", "export", "damages-rlzs",
        str(calc_id),
        "-e", "csv",
        "-d", str(export_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR export calc_id={calc_id}: {result.stderr[-1000:]}")
        return None

    # Find the exported CSV
    csvs = list(export_dir.glob("avg_damages-rlz*.csv"))
    if not csvs:
        print(f"  ERROR: no avg_damages-rlzs CSV found in {export_dir}")
        return None

    shutil.copy2(csvs[0], dest)
    print(f"  Exported → {dest}")
    return dest


# ── Single-run wrapper (used by both serial and parallel modes) ────────────────

def run_one(mode: str, seed: int, dry_run: bool = False) -> dict:
    """
    Execute the full pipeline for one (mode, seed) combination.
    Returns a status dict.
    """
    status = {
        "mode": mode, "seed": seed, "success": False,
        "calc_id": None, "result_csv": None,
        "build_time_s": 0.0, "oq_time_s": 0.0, "total_time_s": 0.0
    }

    t_start = time.perf_counter()

    # Step 1 – build exposure / fragility / GMF
    t_build = time.perf_counter()
    if not run_build_script(mode, seed, dry_run):
        status["total_time_s"] = time.perf_counter() - t_start
        return status
    status["build_time_s"] = time.perf_counter() - t_build

    # Step 2 – write job.ini
    ini_path = write_job_ini(mode, seed)

    # Step 3 – OpenQuake
    t_oq = time.perf_counter()
    calc_id = run_openquake(ini_path, dry_run)
    status["oq_time_s"] = time.perf_counter() - t_oq
    
    if calc_id is None:
        status["total_time_s"] = time.perf_counter() - t_start
        return status
    status["calc_id"] = calc_id

    # Step 4 – export
    dest = export_results(calc_id, mode, seed, dry_run)
    if dest is None:
        status["total_time_s"] = time.perf_counter() - t_start
        return status

    status["result_csv"] = str(dest)
    status["success"] = True
    status["total_time_s"] = time.perf_counter() - t_start
    return status


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch experiment runner: 4 modes × N seeds → OQ results"
    )
    t_master_start = time.perf_counter()
    parser.add_argument(
        "--modes", nargs="+", default=ALL_MODES,
        choices=ALL_MODES, metavar="MODE",
        help="Which modes to run (default: all four)",
    )
    parser.add_argument(
        "--seeds", type=int, default=100,
        help="Number of random seeds per mode (default: 100)",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="First seed index (default: 0, i.e. seeds 0–99)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers. OQ itself is multi-threaded, so 1 is usually "
             "safest unless you have many cores and a large RAM budget.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing anything.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (mode, seed) pairs whose result CSV already exists.",
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    pairs = [(mode, seed) for mode in args.modes for seed in seeds]
    total = len(pairs)

    print(f"{'DRY-RUN ' if args.dry_run else ''}Batch experiment: "
          f"{len(args.modes)} modes × {len(seeds)} seeds = {total} runs")
    print(f"Results directory: {RESULTS_DIR}\n")

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Filter already-completed runs when --resume is set
    if args.resume:
        filtered = []
        for mode, seed in pairs:
            dest = Path(RESULTS_DIR) / f"{mode}_seed_{seed}.csv"
            if dest.exists():
                print(f"  SKIP (exists): {mode}/seed_{seed}")
            else:
                filtered.append((mode, seed))
        pairs = filtered
        print(f"\n{len(pairs)} runs remaining after resume filter.\n")

    statuses = []
    failed   = []

    if args.workers == 1:
        # ── Serial mode ──
        for i, (mode, seed) in enumerate(pairs, 1):
            print(f"\n── Run {i}/{len(pairs)}: {mode}  seed={seed} ──")
            s = run_one(mode, seed, args.dry_run)
            statuses.append(s)
            if not s["success"]:
                failed.append((mode, seed))
    else:
        # ── Parallel mode ──
        print(f"Running with {args.workers} parallel workers.\n"
              "Note: OQ uses multiple threads internally; ensure you have "
              "sufficient RAM before increasing workers.\n")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, mode, seed, args.dry_run): (mode, seed)
                for mode, seed in pairs
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                mode, seed = futures[fut]
                try:
                    s = fut.result()
                except Exception as exc:
                    print(f"  EXCEPTION {mode}/seed_{seed}: {exc}")
                    s = {"mode": mode, "seed": seed, "success": False}
                statuses.append(s)
                if not s["success"]:
                    failed.append((mode, seed))
                print(f"  [{done}/{len(pairs)}] {mode}/seed_{seed} "
                      f"{'OK' if s['success'] else 'FAILED'}")

# ── Summary ──
    experiment_elapsed = time.perf_counter() - t_master_start 
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"{'='*60}")
    success_count = sum(1 for s in statuses if s["success"])
    print(f"  Succeeded : {success_count}/{len(statuses)}")
    print(f"  Failed    : {len(failed)}")
    
    if failed:
        print("\n  Failed runs:")
        for mode, seed in failed:
            print(f"    {mode}  seed={seed}")

    # --- COMPUTE RESOURCES SUMMARY FOR PAPER ---
    
    # Gather hardware info
    cpu_arch = platform.machine()
    os_name = platform.system()
    cpu_count = os.cpu_count()
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    
    # Calculate compute times
    successful_runs = [s for s in statuses if s["success"]]
    if successful_runs:
        avg_build = sum(s["build_time_s"] for s in successful_runs) / len(successful_runs)
        avg_oq = sum(s["oq_time_s"] for s in successful_runs) / len(successful_runs)
        avg_total = sum(s["total_time_s"] for s in successful_runs) / len(successful_runs)
        total_cpu_hours = sum(s["total_time_s"] for s in statuses) / 3600.0
    else:
        avg_build = avg_oq = avg_total = total_cpu_hours = 0.0

    print(f"\n{'='*60}")
    print("COMPUTE RESOURCE SUMMARY (For Publication)")
    print(f"{'='*60}")
    print("Hardware Environment:")
    print(f"  • Compute Type: CPU ({os_name} {cpu_arch})") # Note: adjust if using a specific cloud VM instance
    print(f"  • Logical Cores: {cpu_count}")
    print(f"  • Total System Memory: {ram_gb} GB")
    print(f"  • Parallel Workers Used: {args.workers}")
    print("\nCompute Time Metrics:")
    print(f"  • Avg. Prep Time per run: {avg_build:.1f}s")
    print(f"  • Avg. OpenQuake execution per run: {avg_oq:.1f}s")
    print(f"  • Avg. Total Time per run: {avg_total:.1f}s")
    print(f"  • Total Wall-clock time for batch: {experiment_elapsed/60:.1f} minutes")
    print(f"  • Cumulative Compute Effort: {total_cpu_hours:.2f} task-hours")
    print(f"{'='*60}\n")

    # Write a manifest CSV for the post-processing script
    import csv
    manifest_path = Path(RESULTS_DIR) / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        # Added the time tracking fields to the CSV output
        writer = csv.DictWriter(
            f, fieldnames=["mode", "seed", "success", "calc_id", "result_csv", 
                           "build_time_s", "oq_time_s", "total_time_s"]
        )
        writer.writeheader()
        writer.writerows(statuses)
    print(f"  Manifest written → {manifest_path}")
    print("  Run analyze_uncertainty.py to compute 2-sigma error bars.\n")


if __name__ == "__main__":
    main()
