"""Biological validation via insulation score and TAD boundary calls.

Consumes mosaics saved by ``reconstruct_chromosome.py --save-npy`` and for
each method computes:
  - Insulation score profile (Crane et al., 2015)
  - Pearson correlation of the log2-normalized profile vs HR
  - TAD-boundary F1 vs HR (HR boundaries are ground truth)

Boundary calling uses the canonical delta-vector / zero-crossing approach:
for each bin i, delta(i) = mean(IS[i - delta_w : i]) - mean(IS[i + 1 : i + 1 + delta_w]).
Negative-to-positive zero crossings of delta, with the local IS below a
strength threshold, are called as boundaries.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from repro import write_run_artifacts


DEFAULT_METHODS = ["LR", "Bicubic", "Gaussian", "HiCPlus", "SR-VAE"]


def insulation_score(mat: np.ndarray, w: int = 20) -> np.ndarray:
    """Mean contact intensity in a w x w box above the diagonal at each bin.

    Returns the log2-normalized profile (log2(IS(i) / mean(IS))) with NaNs
    where the window falls off the edge or the mean is non-positive.
    """
    N = mat.shape[0]
    raw = np.full(N, np.nan, dtype=np.float64)
    for i in range(w, N - w):
        sub = mat[i - w:i, i + 1:i + w + 1]
        if sub.size == 0:
            continue
        raw[i] = float(sub.mean())

    ok = np.isfinite(raw) & (raw > 0)
    if not ok.any():
        return raw  # all-NaN, caller handles
    mean_is = float(raw[ok].mean())
    if mean_is <= 0:
        return raw
    out = np.full(N, np.nan, dtype=np.float64)
    out[ok] = np.log2(raw[ok] / mean_is + 1e-12)
    return out


def delta_vector(profile: np.ndarray, delta_w: int = 10) -> np.ndarray:
    N = profile.size
    out = np.full(N, np.nan, dtype=np.float64)
    for i in range(delta_w, N - delta_w):
        left = profile[i - delta_w:i]
        right = profile[i + 1:i + 1 + delta_w]
        if np.isfinite(left).all() and np.isfinite(right).all():
            out[i] = float(left.mean() - right.mean())
    return out


def call_boundaries(profile: np.ndarray, delta_w: int = 10, min_strength: float = 0.1) -> np.ndarray:
    """Zero crossings (neg -> pos) of the delta vector, requiring a local dip."""
    d = delta_vector(profile, delta_w=delta_w)
    N = profile.size
    boundaries = []
    for i in range(1, N - 1):
        if not (np.isfinite(d[i - 1]) and np.isfinite(d[i + 1])):
            continue
        if d[i - 1] < 0.0 <= d[i + 1]:
            # require a real dip in the IS profile at this bin
            if np.isfinite(profile[i]) and profile[i] < -abs(min_strength):
                boundaries.append(i)
    return np.asarray(boundaries, dtype=np.int64)


def boundary_f1(pred: np.ndarray, true: np.ndarray, tol: int = 5) -> tuple[float, float, float]:
    if true.size == 0:
        return float("nan"), float("nan"), float("nan")
    used_true = np.zeros(true.size, dtype=bool)
    tp = 0
    for p in pred:
        diffs = np.abs(true - p)
        diffs[used_true] = tol + 1
        k = int(np.argmin(diffs)) if diffs.size else -1
        if k >= 0 and diffs[k] <= tol:
            used_true[k] = True
            tp += 1
    precision = tp / max(1, pred.size)
    recall = tp / max(1, true.size)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def load_mosaic(mosaic_dir: str, base: str, method_key: str) -> np.ndarray | None:
    path = os.path.join(mosaic_dir, f"{base}_{method_key}.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path)


def method_key(name: str) -> str:
    return name.lower().replace("-", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaic-dir", required=True,
                    help="Directory containing {split}_chr{chrom}_*.npy from reconstruct_chromosome.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--chrom", default="19")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--window", type=int, default=20, help="Insulation window (bins, HR resolution)")
    ap.add_argument("--delta-window", type=int, default=10)
    ap.add_argument("--min-strength", type=float, default=0.1)
    ap.add_argument("--tol", type=int, default=5, help="Boundary match tolerance (bins)")
    ap.add_argument("--plot-range", nargs=2, type=int, default=None,
                    help="Optional (start, end) HR bin range for the profile figure")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    write_run_artifacts(args.outdir, script_name="scripts/insulation_validation.py",
                        args_dict=vars(args), cfg={}, extra={})

    chrom = str(args.chrom).replace("chr", "")
    base = f"{args.split}_chr{chrom}"

    hr = load_mosaic(args.mosaic_dir, base, method_key("HR"))
    if hr is None:
        raise SystemExit(f"HR mosaic not found. Did you pass --save-npy to reconstruct_chromosome.py? "
                         f"Expected: {os.path.join(args.mosaic_dir, base + '_hr.npy')}")

    # Mask: the reconstructed band. Only score bins whose insulation window is fully inside it.
    support = hr > 0
    diag_ok = np.zeros(hr.shape[0], dtype=bool)
    for i in range(args.window, hr.shape[0] - args.window):
        if support[i - args.window:i + args.window + 1, i - args.window:i + args.window + 1].all():
            diag_ok[i] = True

    hr_is = insulation_score(hr, w=args.window)
    hr_bounds = call_boundaries(hr_is, delta_w=args.delta_window, min_strength=args.min_strength)
    hr_bounds = hr_bounds[diag_ok[hr_bounds]] if hr_bounds.size else hr_bounds

    print(f"[HR] bins_with_valid_window={diag_ok.sum()}  boundaries={hr_bounds.size}")

    profiles = {"HR": hr_is}
    rows = []
    for name in DEFAULT_METHODS:
        m = load_mosaic(args.mosaic_dir, base, method_key(name))
        if m is None:
            print(f"[skip] {name}: mosaic not found")
            continue
        prof = insulation_score(m, w=args.window)
        profiles[name] = prof

        both_ok = diag_ok & np.isfinite(prof) & np.isfinite(hr_is)
        if both_ok.sum() >= 10:
            r, _ = pearsonr(prof[both_ok], hr_is[both_ok])
        else:
            r = float("nan")

        bounds = call_boundaries(prof, delta_w=args.delta_window, min_strength=args.min_strength)
        bounds = bounds[diag_ok[bounds]] if bounds.size else bounds
        p, rec, f1 = boundary_f1(bounds, hr_bounds, tol=args.tol)

        rows.append({
            "method": name,
            "pearson_is": r,
            "boundary_precision": p,
            "boundary_recall": rec,
            "boundary_f1": f1,
            "n_boundaries_pred": int(bounds.size),
            "n_boundaries_hr": int(hr_bounds.size),
        })

    csv_path = os.path.join(args.outdir, f"{base}_insulation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "pearson_is", "boundary_precision", "boundary_recall",
            "boundary_f1", "n_boundaries_pred", "n_boundaries_hr",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {csv_path}")

    lo, hi = args.plot_range if args.plot_range else (0, hr.shape[0])
    lo = max(0, lo); hi = min(hr.shape[0], hi)
    xs = np.arange(lo, hi)
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, prof in profiles.items():
        style = "-" if name == "HR" else "--"
        lw = 2.0 if name == "HR" else 1.2
        ax.plot(xs, prof[lo:hi], style, linewidth=lw, label=name, alpha=0.85)
    for b in hr_bounds:
        if lo <= b < hi:
            ax.axvline(b, color="black", alpha=0.15, linewidth=0.8)
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_xlabel("HR bin")
    ax.set_ylabel("log2 insulation score")
    ax.set_title(f"{base}  insulation score (window={args.window})")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig_path = os.path.join(args.outdir, f"{base}_insulation.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {fig_path}")

    print("\n[summary]")
    print(f"  {'method':10s}  Pearson   Prec    Recall   F1     (pred/hr bounds)")
    for r in rows:
        print(f"  {r['method']:10s}  {r['pearson_is']:+.3f}   "
              f"{r['boundary_precision']:.3f}  {r['boundary_recall']:.3f}   "
              f"{r['boundary_f1']:.3f}  ({r['n_boundaries_pred']}/{r['n_boundaries_hr']})")


if __name__ == "__main__":
    main()
