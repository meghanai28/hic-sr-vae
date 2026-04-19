"""Loop-level biological validation via donut-enrichment peak calling.

Consumes mosaics saved by ``reconstruct_chromosome.py --save-npy`` and, for
each method, calls loops using a HiCCUPS-inspired donut filter:
  - enrichment(i, j) = mat(i, j) / mean(mat over annular ring around (i, j))
  - a pixel is a loop candidate if it is a local maximum inside a k x k
    window AND its enrichment exceeds a threshold.
  - off-diagonal distance is constrained to ``[min_sep, max_sep]`` bins, to
    ignore the diagonal and the outside of the reconstruction band.

HR-called loops are the ground truth; we report Precision / Recall / F1
for each method and sweep the enrichment threshold to report AUPRC.

Unlike full HiCCUPS (Juicer) or chromosight, this is deliberately
self-contained: no Java, no extra pip install, identical code path for
every method. That's what makes it fair.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from repro import write_run_artifacts


DEFAULT_METHODS = ["LR", "Bicubic", "Gaussian", "HiCPlus", "SR-VAE"]


def method_key(name: str) -> str:
    return name.lower().replace("-", "")


def load_mosaic(mosaic_dir: str, base: str, key: str) -> np.ndarray | None:
    path = os.path.join(mosaic_dir, f"{base}_{key}.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path)


def donut_enrichment(mat: np.ndarray, inner: int = 1, outer: int = 5) -> np.ndarray:
    """Center / donut-mean. Donut = (2*outer+1)^2 ring minus (2*inner+1)^2 core.

    Uses uniform_filter twice (fast integral-image equivalent).
    """
    mat = mat.astype(np.float64)
    k_outer = 2 * outer + 1
    k_inner = 2 * inner + 1
    n_outer = k_outer * k_outer
    n_inner = k_inner * k_inner
    n_ring = n_outer - n_inner
    if n_ring <= 0:
        raise ValueError("outer must exceed inner")
    mean_outer = uniform_filter(mat, size=k_outer, mode="constant", cval=0.0)
    mean_inner = uniform_filter(mat, size=k_inner, mode="constant", cval=0.0)
    sum_ring = mean_outer * n_outer - mean_inner * n_inner
    donut_mean = sum_ring / n_ring
    return mat / (donut_mean + 1e-6)


def diag_band_mask(n: int, min_sep: int, max_sep: int) -> np.ndarray:
    """True on pixels (i, j) with min_sep <= j - i <= max_sep (upper triangle)."""
    ii, jj = np.indices((n, n))
    sep = jj - ii
    return (sep >= min_sep) & (sep <= max_sep)


def precompute_peak_map(
    mat: np.ndarray,
    support: np.ndarray,
    band: np.ndarray,
    inner: int = 1,
    outer: int = 5,
    peak_radius: int = 2,
) -> dict[str, np.ndarray]:
    """Run the expensive filters once. ``threshold_peak_map`` thresholds cheaply."""
    enr = donut_enrichment(mat, inner=inner, outer=outer)
    local_max = maximum_filter(mat, size=2 * peak_radius + 1, mode="constant", cval=0.0)
    is_peak = (mat == local_max) & (mat > 0.0) & band & support
    return {"enr": enr.astype(np.float32), "is_peak": is_peak, "mat": mat.astype(np.float32)}


def threshold_peak_map(pm: dict[str, np.ndarray], enr_thresh: float, abs_thresh: float) -> np.ndarray:
    mask = pm["is_peak"] & (pm["enr"] > enr_thresh) & (pm["mat"] > abs_thresh)
    rr, cc = np.where(mask)
    return np.stack([rr, cc], axis=1) if rr.size else np.zeros((0, 2), dtype=np.int64)


def call_loops(
    mat: np.ndarray,
    support: np.ndarray,
    min_sep: int,
    max_sep: int,
    enr_thresh: float,
    abs_thresh: float,
    peak_radius: int = 2,
    inner: int = 1,
    outer: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) of loop-called pixels in the upper triangle."""
    band = diag_band_mask(mat.shape[0], min_sep, max_sep)
    pm = precompute_peak_map(mat, support, band,
                             inner=inner, outer=outer, peak_radius=peak_radius)
    ij = threshold_peak_map(pm, enr_thresh, abs_thresh)
    if ij.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return ij[:, 0], ij[:, 1]


def f1_match(pred_ij: np.ndarray, true_ij: np.ndarray, tol: int) -> tuple[float, float, float, int]:
    """Greedy-match predicted to true loop pixels within Chebyshev tol."""
    if true_ij.size == 0 or pred_ij.size == 0:
        if true_ij.size == 0 and pred_ij.size == 0:
            return float("nan"), float("nan"), float("nan"), 0
        return 0.0, 0.0, 0.0, 0
    used = np.zeros(true_ij.shape[0], dtype=bool)
    tp = 0
    for p in pred_ij:
        d = np.max(np.abs(true_ij - p[None, :]), axis=1).astype(np.int64)
        d[used] = tol + 1
        k = int(np.argmin(d))
        if d[k] <= tol:
            used[k] = True
            tp += 1
    prec = tp / max(1, pred_ij.shape[0])
    rec = tp / max(1, true_ij.shape[0])
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(prec), float(rec), float(f1), int(tp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaic-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--chrom", default="19")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-sep", type=int, default=20,
                    help="Min off-diagonal separation in HR bins (~200 kb at 10 kb)")
    ap.add_argument("--max-sep", type=int, default=200,
                    help="Max off-diagonal separation in HR bins (~2 Mb at 10 kb)")
    ap.add_argument("--inner", type=int, default=1, help="Donut inner half-width")
    ap.add_argument("--outer", type=int, default=5, help="Donut outer half-width")
    ap.add_argument("--peak-radius", type=int, default=2,
                    help="Non-max-suppression radius (bins)")
    ap.add_argument("--enr-thresh", type=float, default=1.5,
                    help="Default enrichment threshold for headline F1")
    ap.add_argument("--abs-thresh", type=float, default=0.05,
                    help="Minimum absolute intensity for a loop call")
    ap.add_argument("--tol", type=int, default=3,
                    help="Match tolerance in bins for pred vs HR loops")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep enrichment threshold and emit F1/AUPRC curves")
    ap.add_argument("--sweep-min", type=float, default=1.05)
    ap.add_argument("--sweep-max", type=float, default=3.0)
    ap.add_argument("--sweep-steps", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    write_run_artifacts(args.outdir, script_name="scripts/loop_validation.py",
                        args_dict=vars(args), cfg={}, extra={})

    chrom = str(args.chrom).replace("chr", "")
    base = f"{args.split}_chr{chrom}"

    hr = load_mosaic(args.mosaic_dir, base, "hr")
    if hr is None:
        raise SystemExit(f"HR mosaic not found: {os.path.join(args.mosaic_dir, base + '_hr.npy')}")
    support = hr > 0
    band = diag_band_mask(hr.shape[0], args.min_sep, args.max_sep)
    print(f"[setup] mosaic={hr.shape} band={int(band.sum()):,} support={int(support.sum()):,}")

    # Precompute per-matrix enrichment + peak maps once; sweep is then cheap.
    peak_maps: dict[str, dict[str, np.ndarray]] = {}
    hr_pm = precompute_peak_map(hr, support, band,
                                inner=args.inner, outer=args.outer,
                                peak_radius=args.peak_radius)
    peak_maps["HR"] = hr_pm
    print(f"[HR] peak map ready")

    for name in DEFAULT_METHODS:
        m = load_mosaic(args.mosaic_dir, base, method_key(name))
        if m is None:
            print(f"[skip] {name}: mosaic not found")
            continue
        peak_maps[name] = precompute_peak_map(m, support, band,
                                              inner=args.inner, outer=args.outer,
                                              peak_radius=args.peak_radius)
        print(f"[{name}] peak map ready")

    # Ground-truth loops from HR at the default threshold.
    hr_ij = threshold_peak_map(hr_pm, args.enr_thresh, args.abs_thresh)
    print(f"[HR] loops called: {hr_ij.shape[0]} (enr>{args.enr_thresh}, abs>{args.abs_thresh})")

    rows = []
    per_method_calls: dict[str, np.ndarray] = {"HR": hr_ij}
    for name in DEFAULT_METHODS:
        if name not in peak_maps:
            continue
        pr_ij = threshold_peak_map(peak_maps[name], args.enr_thresh, args.abs_thresh)
        per_method_calls[name] = pr_ij
        p, r, f1, tp = f1_match(pr_ij, hr_ij, tol=args.tol)
        rows.append({
            "method": name,
            "precision": p, "recall": r, "f1": f1,
            "n_pred": int(pr_ij.shape[0]), "n_hr": int(hr_ij.shape[0]),
            "tp": tp,
        })
        print(f"  {name:10s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  ({pr_ij.shape[0]} pred / {hr_ij.shape[0]} HR)")

    csv_path = os.path.join(args.outdir, f"{base}_loops.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "precision", "recall", "f1",
                                          "n_pred", "n_hr", "tp"])
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {csv_path}")

    # Visualize HR loops overlaid on the HR mosaic.
    fig, axs = plt.subplots(1, min(3, len(per_method_calls)), figsize=(14, 4.8),
                            constrained_layout=True)
    if len(per_method_calls) == 1:
        axs = [axs]
    overlay_methods = ["HR"] + [m for m in ["SR-VAE", "HiCPlus"] if m in per_method_calls]
    for ax, name in zip(axs, overlay_methods[:len(axs)]):
        arr = load_mosaic(args.mosaic_dir, base, method_key(name))
        if arr is None:
            continue
        ij = per_method_calls[name]
        vmax = float(np.clip(arr.max(), 1e-6, 1.0))
        ax.imshow(arr, cmap="Reds", vmin=0.0, vmax=vmax, interpolation="nearest")
        if ij.size:
            ax.scatter(ij[:, 1], ij[:, 0], s=6, facecolors="none",
                       edgecolors="blue", linewidths=0.6)
        ax.set_title(f"{name}  ({ij.shape[0]} loops)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig_path = os.path.join(args.outdir, f"{base}_loops_overlay.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"[saved] {fig_path}")

    if args.sweep:
        thresholds = np.linspace(args.sweep_min, args.sweep_max, args.sweep_steps)
        sweep_rows = []
        curves = {}
        for name in DEFAULT_METHODS:
            if name not in peak_maps:
                continue
            precs, recs, f1s = [], [], []
            for t in thresholds:
                hr_ij_t = threshold_peak_map(hr_pm, float(t), args.abs_thresh)
                pr_ij_t = threshold_peak_map(peak_maps[name], float(t), args.abs_thresh)
                p, r, f1, _ = f1_match(pr_ij_t, hr_ij_t, tol=args.tol)
                precs.append(p); recs.append(r); f1s.append(f1)
                sweep_rows.append({
                    "method": name, "enr_thresh": float(t),
                    "precision": p, "recall": r, "f1": f1,
                    "n_pred": int(pr_ij_t.shape[0]), "n_hr": int(hr_ij_t.shape[0]),
                })
            curves[name] = (np.asarray(precs), np.asarray(recs), np.asarray(f1s))

        sweep_csv = os.path.join(args.outdir, f"{base}_loops_sweep.csv")
        with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["method", "enr_thresh", "precision",
                                              "recall", "f1", "n_pred", "n_hr"])
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"[saved] {sweep_csv}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        for name, (p, r, f1) in curves.items():
            ax1.plot(thresholds, f1, label=name, linewidth=1.4)
            ok = np.isfinite(p) & np.isfinite(r)
            ax2.plot(r[ok], p[ok], label=name, linewidth=1.4, marker=".", markersize=3)
        ax1.set_xlabel("enrichment threshold")
        ax1.set_ylabel("loop F1 vs HR")
        ax1.set_title("Loop F1 vs threshold")
        ax1.legend(fontsize=8); ax1.set_ylim(0, 1)
        ax2.set_xlabel("recall"); ax2.set_ylabel("precision")
        ax2.set_title("Loop precision-recall")
        ax2.legend(fontsize=8); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
        sweep_png = os.path.join(args.outdir, f"{base}_loops_sweep.png")
        fig.savefig(sweep_png, dpi=160)
        plt.close(fig)
        print(f"[saved] {sweep_png}")

        print("\n[sweep summary]")
        print(f"  {'method':10s}  best_F1  @threshold   AUPRC")
        for name, (p, r, f1) in curves.items():
            f1a = np.asarray(f1, dtype=np.float64)
            best_i = int(np.nanargmax(f1a)) if np.isfinite(f1a).any() else 0
            ok = np.isfinite(p) & np.isfinite(r)
            if ok.sum() > 1:
                order = np.argsort(r[ok])
                auprc = float(np.trapezoid(p[ok][order], r[ok][order]))
            else:
                auprc = float("nan")
            print(f"  {name:10s}  {f1a[best_i]:.3f}   {thresholds[best_i]:.3f}      {auprc:.3f}")


if __name__ == "__main__":
    main()
