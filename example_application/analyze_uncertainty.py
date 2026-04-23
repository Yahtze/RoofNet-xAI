"""
analyze_uncertainty.py
=======================
Post-processes all 400 OpenQuake scenario_damage result CSVs produced by
run_batch_experiments.py.

For each (mode, seed) pair it:
  1. Merges the OQ damage output with the RoofNet building metadata.
  2. Applies the hybrid scoring classification (majority-rules / weighted dot-
     product) to assign a predicted damage subtype to every building.
  3. Computes the percentage of buildings in each of the four ground-truth
     categories: no-damage, minor-damage, major-damage, destroyed.
  4. Computes a row-normalised confusion matrix (true subtype vs predicted).

Across the 100 seeds within each mode it then computes:
  • Mean percentage  (μ)
  • Standard deviation (σ)
  • 2σ confidence interval  [μ − 2σ,  μ + 2σ]

  For the confusion matrices it computes, per cell:
  • Mean normalised value  (μ_ij)
  • Standard deviation     (σ_ij)
  • 2σ interval            [μ_ij − 2σ_ij,  μ_ij + 2σ_ij]

Outputs:
  • <RESULTS_DIR>/summary_statistics.csv         — one row per (mode, damage_class)
  • <RESULTS_DIR>/per_run_classifications.csv    — full 400-row table
  • <RESULTS_DIR>/uncertainty_plot.png           — bar chart with 2σ error bars
  • <RESULTS_DIR>/confusion_matrices.png         — 4 normalised CMs with ±2σ per cell
  • <RESULTS_DIR>/confusion_matrix_stats.csv     — full cell-level statistics table

Usage:
    python analyze_uncertainty.py [--results-dir /path/to/batch_results]
                                  [--meta-csv /path/to/merged_roofnet_polygons.csv]
                                  [--plot] [--verbose]
"""

import argparse
import os
import time
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Default paths (mirror those in build_exposure_model.py) ───────────────────

DEFAULT_RESULTS_DIR = (
    "./openquake_results/batch_results"
)
DEFAULT_META_CSV = (
    "/Users/benjamintarver/NYU CERA Scripting/roofnet/notebooks/"
    "merged_roofnet_polygons.csv"
)

# ── Damage classification constants ───────────────────────────────────────────

PROB_COLS = [
    "structural-no_damage",
    "structural-slight",
    "structural-moderate",
    "structural-extensive",
    "structural-complete",
]

DAMAGE_MAP = {
    "structural-no_damage":  "no-damage",
    "structural-slight":     "minor-damage",
    "structural-moderate":   "minor-damage",   # grouped per user schema
    "structural-extensive":  "major-damage",
    "structural-complete":   "destroyed",
}

DAMAGE_MAP_NUMERICAL = {
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Short axis labels for confusion matrix tick marks
CM_TICK_LABELS = ["No\nDamage", "Minor\nDamage", "Major\nDamage", "Destroyed"]

WEIGHTS = np.array([1, 2, 3, 4, 5])


# ── Core classification logic (mirrors user's hybrid scorer) ──────────────────

def classify_buildings(df_oq: pd.DataFrame,
                        df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OQ results with metadata and apply hybrid damage classification.

    Parameters
    ----------
    df_oq   : DataFrame with at least asset_id + PROB_COLS
    df_meta : DataFrame with uid, material_class, subtype, wkt

    Returns
    -------
    merged_df with an added 'predicted_subtype' column
    """
    merged = pd.merge(
        df_oq,
        df_meta[["uid", "material_class", "subtype", "wkt"]],
        left_on="asset_id",
        right_on="uid",
        how="inner",
    )

    predicted = []
    for _, row in merged.iterrows():
        probs = row[PROB_COLS].values.astype(float)
        max_prob = probs.max()
        if max_prob > 0.5:
            best_col = PROB_COLS[int(probs.argmax())]
            predicted.append(DAMAGE_MAP[best_col])
        else:
            weighted_class = int(round(float(np.dot(probs, WEIGHTS))))
            weighted_class = max(1, min(4, weighted_class))
            predicted.append(DAMAGE_MAP_NUMERICAL[weighted_class])

    merged["predicted_subtype"] = predicted
    return merged


def load_oq_result(csv_path: str) -> pd.DataFrame | None:
    """
    Load an OpenQuake dmg_by_asset CSV.
    OQ writes two header rows; `header=1` skips the first descriptive row.
    Handles missing files and empty frames gracefully.
    """
    try:
        df = pd.read_csv(csv_path, header=1)
    except Exception as exc:
        print(f"  WARNING: could not read {csv_path}: {exc}")
        return None

    if df.empty:
        print(f"  WARNING: empty result file {csv_path}")
        return None

    df.columns = df.columns.str.strip()

    missing = [c for c in ["asset_id"] + PROB_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: {csv_path} missing columns {missing} — skipping")
        return None

    return df


# ── Per-run percentage computation ────────────────────────────────────────────

def compute_classification_pcts(merged: pd.DataFrame) -> dict:
    """
    Return {damage_class: percentage} across the full building stock.
    """
    counts = merged["predicted_subtype"].value_counts()
    total  = len(merged)
    return {
        cls: float(counts.get(cls, 0)) / total * 100.0
        for cls in DAMAGE_CLASSES
    }


# ── Per-run confusion matrix ───────────────────────────────────────────────────

def compute_confusion_matrix(merged: pd.DataFrame) -> np.ndarray:
    """
    Build a row-normalised confusion matrix for a single run.

    Rows  = true damage class  (merged['subtype'], ground truth from RoofNet)
    Cols  = predicted class     (merged['predicted_subtype'])
    Order = DAMAGE_CLASSES      ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

    Each row is divided by its sum so values represent the fraction of
    buildings in each true class that were assigned to each predicted class.
    Rows with zero buildings are left as zero rather than producing NaN.

    Returns
    -------
    cm : np.ndarray, shape (4, 4), dtype float64
        Row-normalised confusion matrix.  cm[i, j] is the fraction of
        buildings whose true class is DAMAGE_CLASSES[i] that were predicted
        as DAMAGE_CLASSES[j].
    """
    n   = len(DAMAGE_CLASSES)
    cm  = np.zeros((n, n), dtype=np.float64)
    idx = {cls: i for i, cls in enumerate(DAMAGE_CLASSES)}

    true_labels = merged["subtype"].map(idx)
    pred_labels = merged["predicted_subtype"].map(idx)

    # Drop rows where either label is unrecognised (maps to NaN)
    valid       = true_labels.notna() & pred_labels.notna()
    true_labels = true_labels[valid].astype(int).values
    pred_labels = pred_labels[valid].astype(int).values

    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1

    # Row-normalise: each row sums to 1 (or stays 0 if the class is absent)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)

    return cm


# ── Confusion matrix statistics across seeds ──────────────────────────────────

def aggregate_confusion_matrices(
    cm_list: list,
) -> tuple:
    """
    Stack N per-seed confusion matrices and return cell-wise mean and
    sample standard deviation.

    Parameters
    ----------
    cm_list : list of (4, 4) normalised confusion matrices

    Returns
    -------
    mu    : (4, 4) mean across seeds
    sigma : (4, 4) sample std dev across seeds  (ddof=1)
    """
    stack = np.stack(cm_list, axis=0)   # (N, 4, 4)
    mu    = stack.mean(axis=0)
    sigma = stack.std(axis=0, ddof=1)
    return mu, sigma


# ── Plot: four confusion matrices with ±2σ cell annotations ──────────────────

def generate_confusion_matrix_plots(
    cm_stacks: dict,
    modes: list,
    results_dir: Path,
) -> None:
    """
    Render one row of four confusion matrices (one per mode).

    Each cell shows:
        μ
      ±2σ
    where μ is the mean normalised fraction and 2σ is twice the sample
    standard deviation across the 100 seeds.

    Cell background colour encodes μ (Blues colormap); text colour switches
    to white on dark cells for legibility.  A single shared colourbar spans
    0–1 so intensities are directly comparable across modes.

    Also writes confusion_matrix_stats.csv with every cell's full statistics.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    n_modes = len(modes)
    n_cls   = len(DAMAGE_CLASSES)

    fig, axes = plt.subplots(
        1, n_modes,
        figsize=(5.2 * n_modes, 5.5),
    )
    if n_modes == 1:
        axes = [axes]

    # Single shared scale across all four panels so colours are comparable
    cmap = plt.cm.Blues                          # type: ignore[attr-defined]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    stats_rows = []

    for ax, mode in zip(axes, modes):
        cm_list = cm_stacks.get(mode, [])
        if not cm_list:
            ax.set_title(f"{mode}\n(no data)", fontsize=11)
            ax.axis("off")
            continue

        mu, sigma = aggregate_confusion_matrices(cm_list)
        n_runs    = len(cm_list)

        # ── Heatmap background ────────────────────────────────────────────────
        ax.imshow(mu, cmap=cmap, norm=norm, aspect="equal",
                  interpolation="nearest")

        # ── Cell annotations  (mean on first line, ±2σ on second) ────────────
        for i in range(n_cls):
            for j in range(n_cls):
                mu_val    = float(mu[i, j])
                sigma_val = float(sigma[i, j])
                two_sigma = 2.0 * sigma_val

                # White text on dark cells, black on light cells
                text_color = "white" if norm(mu_val) > 0.55 else "black"

                ax.text(
                    j, i,
                    f"{mu_val:.3f}\n±{two_sigma:.3f}",
                    ha="center", va="center",
                    fontsize=8.5,
                    color=text_color,
                    # Bold the diagonal (correct predictions)
                    fontweight="bold" if i == j else "normal",
                    linespacing=1.6,
                )

                # Accumulate for CSV
                lo = max(0.0, mu_val - two_sigma)
                hi = min(1.0, mu_val + two_sigma)
                stats_rows.append({
                    "mode":            mode,
                    "true_class":      DAMAGE_CLASSES[i],
                    "predicted_class": DAMAGE_CLASSES[j],
                    "n_runs":          n_runs,
                    "mean":            round(mu_val,    6),
                    "std":             round(sigma_val, 6),
                    "ci_2sigma_lo":    round(lo,        6),
                    "ci_2sigma_hi":    round(hi,        6),
                })

        # ── Axes formatting ───────────────────────────────────────────────────
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(CM_TICK_LABELS, fontsize=8)
        ax.set_yticklabels(CM_TICK_LABELS, fontsize=8)
        ax.set_xlabel("Predicted class", fontsize=9, labelpad=6)
        ax.set_ylabel("True class", fontsize=9, labelpad=6)

        # White grid lines to separate cells cleanly
        for k in range(n_cls + 1):
            offset = k - 0.5
            ax.axhline(offset, color="white", linewidth=1.2, clip_on=False)
            ax.axvline(offset, color="white", linewidth=1.2, clip_on=False)

        ax.set_title(
            f"{mode.capitalize()}\n(n = {n_runs} seeds)",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

    # ── Shared colourbar ──────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.15,
                        wspace=0.38)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.65])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),  # type: ignore[attr-defined]
        cax=cbar_ax,
    )
    cb.set_label("Mean normalised fraction", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Normalised Confusion Matrices by Mode\n"
        "Cell text: mean  ±2σ  across 100 random seeds",
        fontsize=12,
        y=0.97,
    )

    plot_path = results_dir / "confusion_matrices.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot saved → {plot_path}")

    # ── Save cell-level statistics CSV ────────────────────────────────────────
    df_stats  = pd.DataFrame(stats_rows)
    stats_csv = results_dir / "confusion_matrix_stats.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"Confusion matrix statistics saved → {stats_csv}")


# ── Main analysis ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 2-sigma uncertainty bands from 400-run batch results"
    )
    t_master_start = time.perf_counter()
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing batch_results CSVs (default: {DEFAULT_RESULTS_DIR})"
    )
    parser.add_argument(
        "--meta-csv", default=DEFAULT_META_CSV,
        help=f"Path to merged_roofnet_polygons.csv (default: {DEFAULT_META_CSV})"
    )
    parser.add_argument(
        "--modes", nargs="+",
        default=["roofnet", "benchmark", "unreinforced", "reinforced"],
        help="Which modes to analyse (default: all four)"
    )
    parser.add_argument(
        "--seeds", type=int, default=100,
        help="Number of seeds per mode expected (default: 100)"
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="First seed index (default: 0)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate uncertainty_plot.png and confusion_matrices.png"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-run classification percentages"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f"Results directory not found: {results_dir}")

    # ── Load metadata ──
    print(f"Loading building metadata from {args.meta_csv} ...")
    df_meta = pd.read_csv(args.meta_csv)
    print(f"  {len(df_meta)} buildings loaded.")

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # ── Collect per-run percentages and confusion matrices ────────────────────
    records   = []
    cm_stacks = {mode: [] for mode in args.modes}   # mode → list of (4,4) arrays

    for mode in args.modes:
        loaded  = 0
        skipped = 0
        for seed in seeds:
            csv_path = results_dir / f"{mode}_seed_{seed}.csv"
            if not csv_path.exists():
                if args.verbose:
                    print(f"  MISSING: {csv_path.name}")
                skipped += 1
                continue

            df_oq = load_oq_result(str(csv_path))
            if df_oq is None:
                skipped += 1
                continue

            merged = classify_buildings(df_oq, df_meta)

            # Classification percentages (existing)
            pcts   = compute_classification_pcts(merged)
            records.append({"mode": mode, "seed": seed, **pcts})

            # Confusion matrix (new)
            cm_stacks[mode].append(compute_confusion_matrix(merged))

            loaded += 1

            if args.verbose:
                pct_str = "  ".join(
                    f"{cls}: {pcts[cls]:.1f}%" for cls in DAMAGE_CLASSES
                )
                print(f"  {mode}/seed_{seed}  {pct_str}")

        print(f"Mode '{mode}': {loaded} runs loaded, {skipped} skipped.")

    if not records:
        sys.exit("No valid results found. Check --results-dir and file paths.")

    df_runs = pd.DataFrame(records)

    # ── Save full per-run table ──
    per_run_path = results_dir / "per_run_classifications.csv"
    df_runs.to_csv(per_run_path, index=False)
    print(f"\nPer-run classifications saved → {per_run_path}")

    # ── Compute and print 2-sigma summary ─────────────────────────────────────
    summary_rows = []
    print("\n" + "="*72)
    print(f"{'MODE':<16} {'DAMAGE CLASS':<16} {'MEAN %':>8} {'STD %':>8} "
          f"{'2σ LOW':>9} {'2σ HIGH':>9} {'N RUNS':>7}")
    print("="*72)

    for mode in args.modes:
        df_mode = df_runs[df_runs["mode"] == mode]
        if df_mode.empty:
            print(f"  No data for mode '{mode}' — skipping")
            continue
        n_runs = len(df_mode)

        for cls in DAMAGE_CLASSES:
            values = df_mode[cls].values
            mu     = float(np.mean(values))
            sigma  = float(np.std(values, ddof=1))
            lo     = max(0.0,   mu - 2 * sigma)
            hi     = min(100.0, mu + 2 * sigma)

            summary_rows.append({
                "mode":          mode,
                "damage_class":  cls,
                "mean_pct":      round(mu,    4),
                "std_pct":       round(sigma, 4),
                "ci_2sigma_lo":  round(lo,    4),
                "ci_2sigma_hi":  round(hi,    4),
                "n_runs":        n_runs,
            })

            print(f"{mode:<16} {cls:<16} {mu:>7.2f}%  {sigma:>6.2f}%  "
                  f"[{lo:>6.2f}%, {hi:>6.2f}%]  {n_runs:>6}")

        print("-"*72)

    print("="*72)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / "summary_statistics.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved → {summary_path}")

    # ── Print confusion matrix summary to terminal ────────────────────────────
    print("\n" + "="*72)
    print("CONFUSION MATRIX SUMMARY  (mean normalised fraction  ±2σ)")
    print("Rows = true class   |   Columns = predicted class")
    print("="*72)
    for mode in args.modes:
        cm_list = cm_stacks.get(mode, [])
        if not cm_list:
            continue
        mu, sigma = aggregate_confusion_matrices(cm_list)
        print(f"\n  Mode: {mode}  (n = {len(cm_list)} seeds)")
        header = f"  {'':18s}" + "".join(f"{c:>18s}" for c in DAMAGE_CLASSES)
        print(header)
        print("  " + "-" * (18 + 18 * len(DAMAGE_CLASSES)))
        for i, true_cls in enumerate(DAMAGE_CLASSES):
            row_str = f"  {true_cls:<18s}"
            for j in range(len(DAMAGE_CLASSES)):
                cell = f"{mu[i,j]:.3f}±{2*sigma[i,j]:.3f}"
                row_str += f"{cell:>18s}"
            print(row_str)
    print("\n" + "="*72)

    # ── Optional plots ────────────────────────────────────────────────────────
    if args.plot:
        generate_uncertainty_plot(df_summary, results_dir)
        generate_confusion_matrix_plots(cm_stacks, args.modes, results_dir)
        
    # ── Summary ──
    experiment_elapsed = time.perf_counter() - t_master_start 
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  • Total Wall-clock time for batch: {experiment_elapsed/60:.1f} minutes")
    print(f"{'='*60}\n")


def generate_uncertainty_plot(df_summary: pd.DataFrame,
                               results_dir: Path) -> None:
    """
    Create a grouped bar chart with 2σ error bars.
    One group per damage class; bars coloured by mode.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    modes   = df_summary["mode"].unique()
    n_modes = len(modes)
    colors  = plt.cm.tab10.colors[:n_modes]  # type: ignore[attr-defined]

    fig, axes = plt.subplots(
        1, len(DAMAGE_CLASSES),
        figsize=(5 * len(DAMAGE_CLASSES), 5),
        sharey=False,
    )
    if len(DAMAGE_CLASSES) == 1:
        axes = [axes]

    for ax, cls in zip(axes, DAMAGE_CLASSES):
        sub = df_summary[df_summary["damage_class"] == cls].reset_index(drop=True)
        x   = np.arange(len(sub))
        ax.bar(
            x,
            sub["mean_pct"],
            yerr=2 * sub["std_pct"],
            capsize=5,
            color=[colors[list(modes).index(m)] for m in sub["mode"]],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sub["mode"], rotation=20, ha="right", fontsize=9)
        ax.set_title(cls, fontsize=11, fontweight="bold")
        ax.set_ylabel("Buildings (%)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Damage Classification Percentages by Mode\n"
        "Error bars = ±2σ across 100 random seeds",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    plot_path = results_dir / "uncertainty_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Uncertainty plot saved → {plot_path}")


if __name__ == "__main__":
    main()
