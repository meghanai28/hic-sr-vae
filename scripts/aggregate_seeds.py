"""Aggregate eval CSVs across seeds.

Consumes one or more ``metrics.csv`` files from ``scripts/evaluate.py``. The
evaluate CSV is wide: one row per sample, with columns ``{method}_{metric}``
(e.g. ``SR-VAE_mse``, ``HiCPlus_ssim``).

For each input file, this script computes a per-method mean of each metric
(that is one "seed" point). It then reports mean +/- std across the supplied
files (seeds), and for a single reference file runs a paired Wilcoxon
signed-rank test of ``SR-VAE`` vs every other method, using the per-sample
rows in that file.

Example:
    py scripts/aggregate_seeds.py \\
        --csvs runs/paper_full/eval/metrics.csv \\
               runs/paper_full_seed43/eval/metrics.csv \\
               runs/paper_full_seed44/eval/metrics.csv \\
        --paired-csv runs/paper_full/eval/metrics.csv \\
        --out runs/paper_full/eval/seed_summary.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


KNOWN_METHODS = ["LR", "Bicubic", "Gaussian", "HiCPlus", "SR-VAE"]


def _num(val) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def parse_columns(header: list[str]) -> dict[str, list[str]]:
    """Map metric -> list of methods that have that metric column.

    A column "SR-VAE_mse" maps metric=mse, method=SR-VAE. We match against
    KNOWN_METHODS so method names containing underscores (none today) or hyphens
    are handled correctly.
    """
    metrics: dict[str, list[str]] = {}
    for col in header:
        for m in KNOWN_METHODS:
            prefix = f"{m}_"
            if col.startswith(prefix):
                suffix = col[len(prefix):]
                metrics.setdefault(suffix, []).append(m)
                break
    return metrics


def read_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return header, rows


def per_file_means(rows: list[dict], metrics: dict[str, list[str]]) -> dict[tuple[str, str], float]:
    out = {}
    for metric, methods in metrics.items():
        for m in methods:
            col = f"{m}_{metric}"
            vals = np.asarray([_num(r.get(col, "")) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out[(m, metric)] = float(vals.mean()) if vals.size else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="Paths to per-seed metrics.csv files (wide format)")
    ap.add_argument("--ref-method", default="SR-VAE",
                    help="Reference method for paired Wilcoxon vs other methods")
    ap.add_argument("--paired-csv", default="",
                    help="CSV path whose per-sample rows are used for paired Wilcoxon. "
                         "Defaults to the first --csvs entry.")
    ap.add_argument("--out", default="", help="Optional path to write summary CSV")
    args = ap.parse_args()

    files = [p for p in args.csvs if os.path.isfile(p)]
    missing = [p for p in args.csvs if p not in files]
    for p in missing:
        print(f"[skip] missing: {p}", file=sys.stderr)
    if not files:
        raise SystemExit("no CSVs found")

    per_seed = []
    metrics_all: dict[str, list[str]] = {}
    for path in files:
        header, rows = read_csv(path)
        metrics = parse_columns(header)
        for m, methods in metrics.items():
            existing = set(metrics_all.get(m, []))
            metrics_all[m] = sorted(existing.union(methods))
        means = per_file_means(rows, metrics)
        per_seed.append({"path": path, "means": means})

    methods_all = sorted({m for methods in metrics_all.values() for m in methods},
                        key=lambda x: KNOWN_METHODS.index(x) if x in KNOWN_METHODS else 999)
    metric_names = sorted(metrics_all.keys())

    print(f"[aggregate] files={len(files)}  methods={methods_all}  metrics={metric_names}")

    summary = []
    for method in methods_all:
        entry = {"method": method, "n_seeds": len(files)}
        for metric in metric_names:
            vals = np.asarray([ps["means"].get((method, metric), float("nan")) for ps in per_seed],
                              dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                entry[f"{metric}_mean"] = float("nan")
                entry[f"{metric}_std"] = float("nan")
            else:
                entry[f"{metric}_mean"] = float(vals.mean())
                entry[f"{metric}_std"] = float(vals.std(ddof=1) if vals.size > 1 else 0.0)
        summary.append(entry)

    print("\n[summary mean +/- std across seeds]")
    header_cols = ["method"] + [f"{m}(mean±std)" for m in metric_names]
    print("  " + "  ".join(f"{h:>26s}" for h in header_cols))
    for e in summary:
        row = [e["method"]] + [f"{e[f'{m}_mean']:.4f}±{e[f'{m}_std']:.4f}" for m in metric_names]
        print("  " + "  ".join(f"{c:>26s}" for c in row))

    # Paired Wilcoxon on a single CSV's per-sample rows.
    paired_path = args.paired_csv or files[0]
    if wilcoxon is None:
        print("\n[note] scipy.stats.wilcoxon unavailable; skipping paired tests")
    elif not os.path.isfile(paired_path):
        print(f"\n[note] paired-csv not found: {paired_path}")
    elif args.ref_method not in methods_all:
        print(f"\n[note] ref-method {args.ref_method} not present; skipping paired tests")
    else:
        print(f"\n[paired Wilcoxon across tiles; ref={args.ref_method}; file={paired_path}]")
        header, rows = read_csv(paired_path)
        metrics = parse_columns(header)
        for metric, methods in metrics.items():
            if args.ref_method not in methods:
                continue
            ref_col = f"{args.ref_method}_{metric}"
            for m in methods:
                if m == args.ref_method:
                    continue
                col = f"{m}_{metric}"
                pairs = [(
                    _num(r.get(ref_col, "")), _num(r.get(col, ""))
                ) for r in rows]
                pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
                if len(pairs) < 3:
                    continue
                arr = np.asarray(pairs)
                if np.allclose(arr[:, 0], arr[:, 1]):
                    print(f"  {metric:10s}  {args.ref_method} vs {m:8s}: identical (n={len(pairs)})")
                    continue
                try:
                    stat, p = wilcoxon(arr[:, 0], arr[:, 1])
                    diff = float(np.median(arr[:, 0] - arr[:, 1]))
                    print(f"  {metric:10s}  {args.ref_method} vs {m:8s}: "
                          f"median_diff={diff:+.4f}  W={stat:.1f}  p={p:.4g}  (n={len(pairs)})")
                except ValueError as e:
                    print(f"  {metric:10s}  {args.ref_method} vs {m:8s}: wilcoxon failed ({e})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fields = ["method", "n_seeds"] + sum([[f"{m}_mean", f"{m}_std"] for m in metric_names], [])
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
