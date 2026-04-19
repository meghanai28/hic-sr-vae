"""Tile-mosaic reconstruction of a held-out chromosome.

LR tiles (size H) -> SR predictions at HR resolution (size sH), stitched on a
chromosome-scale canvas with a 2D Hann blend window. Bicubic / Gaussian use the
same LR tiles upsampled to HR resolution. HiCPlus is included if a checkpoint
is given.
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter, zoom as ndi_zoom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_chrom_stats, parse_tile_name
from metrics import genomedisco_score, hicspector_score
from model import build_model
from repro import set_global_seed, write_run_artifacts
from utils import log1p_normalize


def hann1d(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float32)
    k = np.arange(n, dtype=np.float32)
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * k / float(n - 1)))).astype(np.float32)


def blend_window_2d(h: int, w: int) -> np.ndarray:
    return np.outer(hann1d(h), hann1d(w)).astype(np.float32)


def bicubic_upsample(arr_np, out_shape):
    src = torch.from_numpy(arr_np).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(src, size=out_shape, mode="bicubic", align_corners=False)
    return up[0, 0].clamp(0.0, 1.0).numpy()


def mse_np(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def masked_ssim(x, y, mask, k=11, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    x, y = x.astype(np.float32), y.astype(np.float32)
    mask = mask.astype(bool)
    if not mask.any():
        return float("nan")

    def box_mean(img, ksz):
        if ksz <= 1:
            return img.copy()
        pad = ksz // 2
        a = np.pad(img, pad, mode="edge")
        ii = np.cumsum(np.cumsum(a, 0), 1)
        return (ii[ksz:, ksz:] - ii[:-ksz, ksz:] - ii[ksz:, :-ksz] + ii[:-ksz, :-ksz]) / (ksz * ksz)

    mu_x, mu_y = box_mean(x, k), box_mean(y, k)
    sig_x2 = box_mean(x * x, k) - mu_x ** 2
    sig_y2 = box_mean(y * y, k) - mu_y ** 2
    sig_xy = box_mean(x * y, k) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2)
    sm = np.clip(num / (den + 1e-12), -1, 1)
    h = min(sm.shape[0], mask.shape[0])
    w = min(sm.shape[1], mask.shape[1])
    return float(np.mean(sm[:h, :w][mask[:h, :w]]))


def collect_pairs(cfg, split, chrom):
    data = cfg.get("data", {})
    lr_glob = data.get(f"{split}_lr") or f"tiles/lr/{split}/*.npy"
    hr_glob = data.get(f"{split}_hr") or f"tiles/hr/{split}/*.npy"
    lrs = sorted(glob.glob(lr_glob, recursive=True))
    hrs = {os.path.basename(p): p for p in sorted(glob.glob(hr_glob, recursive=True))}
    norm = lambda s: str(s).replace("chr", "")
    target = norm(chrom)
    out = []
    for lr_path in lrs:
        name = os.path.basename(lr_path)
        if name not in hrs:
            continue
        ch, i, j = parse_tile_name(name)
        if norm(ch) != target:
            continue
        out.append((i, j, lr_path, hrs[name]))
    return out


@torch.no_grad()
def reconstruct(model, hicplus, chrom_stats, pairs, scale, device):
    if not pairs:
        raise SystemExit("no tiles matched")

    hr_patch = np.load(pairs[0][3]).shape[0]
    canvas = max(max(i + hr_patch, j + hr_patch) for i, j, _, _ in pairs)
    print(f"[mosaic] tiles={len(pairs)} canvas={canvas}x{canvas}  hr_patch={hr_patch}")

    def acc():
        return np.zeros((canvas, canvas), dtype=np.float32)

    accs = {n: acc() for n in ("LR", "Bicubic", "Gaussian", "SR-VAE", "HR")}
    if hicplus is not None:
        accs["HiCPlus"] = acc()
    wmap = acc()

    for i, j, lr_path, hr_path in pairs:
        chrom, _, _ = parse_tile_name(lr_path)
        scale_norm = chrom_stats[chrom]
        lr_raw = np.load(lr_path).astype(np.float32)
        hr_raw = np.load(hr_path).astype(np.float32)

        lr_t = log1p_normalize(torch.from_numpy(lr_raw).unsqueeze(0).unsqueeze(0), scale_norm).to(device)
        hr_t = log1p_normalize(torch.from_numpy(hr_raw).unsqueeze(0).unsqueeze(0), scale_norm)

        sr_t, _, _ = model(lr_t, sample=False)
        sr_np = sr_t[0, 0].clamp(0.0, 1.0).cpu().numpy()
        hr_np = hr_t[0, 0].numpy()
        lr_np = lr_t[0, 0].cpu().numpy()
        out_shape = hr_np.shape

        bicubic_np = bicubic_upsample(lr_np, out_shape)
        smoothed = gaussian_filter(lr_np.astype(np.float64), sigma=1.0).astype(np.float32)
        gaussian_np = ndi_zoom(smoothed, (out_shape[0] / lr_np.shape[0], out_shape[1] / lr_np.shape[1]),
                               order=1).astype(np.float32)
        gaussian_np = np.clip(gaussian_np, 0.0, 1.0)
        lr_up_np = bicubic_upsample(lr_np, out_shape)

        h, w = out_shape
        wnd = blend_window_2d(h, w)
        accs["LR"][i:i + h, j:j + w]      += lr_up_np   * wnd
        accs["Bicubic"][i:i + h, j:j + w] += bicubic_np * wnd
        accs["Gaussian"][i:i + h, j:j + w] += gaussian_np * wnd
        accs["SR-VAE"][i:i + h, j:j + w]  += sr_np      * wnd
        accs["HR"][i:i + h, j:j + w]      += hr_np      * wnd
        if hicplus is not None:
            hp_np = hicplus(lr_t)[0, 0].clamp(0.0, 1.0).cpu().numpy()
            accs["HiCPlus"][i:i + h, j:j + w] += hp_np * wnd
        wmap[i:i + h, j:j + w] += wnd

        if i != j:
            accs["LR"][j:j + w, i:i + h]      += lr_up_np.T   * wnd.T
            accs["Bicubic"][j:j + w, i:i + h] += bicubic_np.T * wnd.T
            accs["Gaussian"][j:j + w, i:i + h] += gaussian_np.T * wnd.T
            accs["SR-VAE"][j:j + w, i:i + h]  += sr_np.T      * wnd.T
            accs["HR"][j:j + w, i:i + h]      += hr_np.T      * wnd.T
            if hicplus is not None:
                accs["HiCPlus"][j:j + w, i:i + h] += hp_np.T * wnd.T
            wmap[j:j + w, i:i + h] += wnd.T

    mask = wmap > 1e-8
    out = {}
    for name, a in accs.items():
        m = np.zeros_like(a)
        m[mask] = a[mask] / wmap[mask]
        out[name] = m
    return out, mask, len(pairs)


def plot_panels(panels, titles, outpath, suptitle=""):
    vmax = float(np.clip(np.max([p.max() for p in panels]), 1e-6, 1.0))
    fig, axs = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6),
                            constrained_layout=True)
    if len(panels) == 1:
        axs = [axs]
    for ax, arr, title in zip(axs, panels, titles):
        im = ax.imshow(arr, cmap="Reds", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hicplus-ckpt", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--chrom", default="19")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--save-npy", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed, deterministic=bool(cfg.get("deterministic", True)))
    write_run_artifacts(args.outdir, script_name="scripts/reconstruct_chromosome.py",
                        args_dict=vars(args), cfg=cfg, extra={"seed": seed})

    stats = load_chrom_stats(cfg.get("data", {}).get("stats", "tiles/hr/stats.json"))

    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck.get("model_name", "srvae"),
                        z_ch=ck.get("z_ch", 32), base_ch=ck.get("base_ch", 32),
                        scale_factor=ck.get("scale", 2)).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"[main] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    hicplus = None
    if args.hicplus_ckpt:
        hck = torch.load(args.hicplus_ckpt, map_location="cpu")
        hicplus = build_model("hicplus", scale_factor=hck.get("scale", 2)).to(device).eval()
        hicplus.load_state_dict(hck["model"])
        print(f"[baseline] loaded HiCPlus from {args.hicplus_ckpt}")

    chrom = str(args.chrom).replace("chr", "")
    pairs = collect_pairs(cfg, args.split, chrom)
    mosaics, mask, n = reconstruct(model, hicplus, stats, pairs,
                                   scale=ck.get("scale", 2), device=device)

    methods = ["LR", "Bicubic", "Gaussian", "SR-VAE"] + (["HiCPlus"] if hicplus is not None else [])
    rows = []
    for name in methods:
        rows.append({
            "method": name,
            "mse": mse_np(mosaics[name][mask], mosaics["HR"][mask]),
            "ssim": masked_ssim(mosaics[name], mosaics["HR"], mask, data_range=1.0),
            "genomedisco": genomedisco_score(np.where(mask, mosaics[name], 0.0),
                                              np.where(mask, mosaics["HR"], 0.0)),
            "hicspector": hicspector_score(np.where(mask, mosaics[name], 0.0),
                                            np.where(mask, mosaics["HR"], 0.0)),
            "coverage": float(mask.mean()),
            "n_tiles": n,
        })

    out_base = f"{args.split}_chr{chrom}"
    plot_panels(
        [mosaics[m] for m in methods] + [mosaics["HR"]],
        methods + ["HR"],
        outpath=os.path.join(args.outdir, f"{out_base}_mosaic.png"),
        suptitle=f"Tile-mosaic: split={args.split} chr{chrom} coverage={mask.mean()*100:.1f}%",
    )
    plot_panels([mask.astype(np.float32)], ["Support"],
                outpath=os.path.join(args.outdir, f"{out_base}_support.png"))

    csv_path = os.path.join(args.outdir, f"{out_base}_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "mse", "ssim", "genomedisco", "hicspector", "coverage", "n_tiles"])
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {csv_path}")

    if args.save_npy:
        for n_, arr in mosaics.items():
            np.save(os.path.join(args.outdir, f"{out_base}_{n_.lower().replace('-', '')}.npy"), arr)

    print("\n[summary]")
    for r in rows:
        print(f"  {r['method']:8s}  MSE={r['mse']:.4f}  SSIM={r['ssim']:.3f}  DISCO={r['genomedisco']:.3f}  HiC-Spec={r['hicspector']:.3f}")


if __name__ == "__main__":
    main()
