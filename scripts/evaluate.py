"""Evaluate SR-VAE (and optional HiCPlus) against bicubic and Gaussian baselines.

The bicubic and Gaussian baselines upsample LR (size H) to HR (size sH).
All metrics are reported in the normalized log1p / chrom-max space (range [0, 1]).
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter, zoom as ndi_zoom
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import make_loaders
from metrics import genomedisco_score, hicspector_score
from model import build_model
from repro import set_global_seed, write_run_artifacts
from utils import center_crop_to_match


def mse_np(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def ssim_np(x, y, k=11, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    x, y = x.astype(np.float32), y.astype(np.float32)

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
    return float(np.clip(np.mean(num / (den + 1e-12)), -1.0, 1.0))


def bicubic_upsample(lr_np: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    src = torch.from_numpy(lr_np).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(src, size=out_shape, mode="bicubic", align_corners=False)
    return up[0, 0].clamp(0.0, 1.0).numpy()


def gaussian_baseline(lr_np: np.ndarray, out_shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    smoothed = gaussian_filter(lr_np.astype(np.float64), sigma=sigma).astype(np.float32)
    sf_h = out_shape[0] / lr_np.shape[0]
    sf_w = out_shape[1] / lr_np.shape[1]
    if abs(sf_h - 1.0) > 1e-6 or abs(sf_w - 1.0) > 1e-6:
        smoothed = ndi_zoom(smoothed, (sf_h, sf_w), order=1).astype(np.float32)
    return np.clip(smoothed, 0.0, 1.0)


def load_optional_baseline(path: str | None, device: str):
    if not path:
        return None
    ck = torch.load(path, map_location="cpu")
    name = ck.get("model_name", "hicplus")
    model = build_model(name, z_ch=ck.get("z_ch", 32), base_ch=ck.get("base_ch", 32),
                        scale_factor=ck.get("scale", 2)).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"[baseline] loaded {name} from {path}")
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="SR-VAE checkpoint")
    ap.add_argument("--hicplus-ckpt", default="", help="Optional HiCPlus checkpoint for learned baseline")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-disco", action="store_true",
                    help="Skip GenomeDISCO/HiC-Spector (faster, lower RAM)")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)
    write_run_artifacts(args.outdir, script_name="scripts/evaluate.py",
                        args_dict=vars(args), cfg=cfg,
                        extra={"seed": seed, "deterministic": deterministic})

    _, val_ld, test_ld, _ = make_loaders(cfg, verbose=True)
    loader = test_ld or val_ld
    assert loader, "no val/test data"

    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck.get("model_name", "srvae"),
                        z_ch=ck.get("z_ch", 32), base_ch=ck.get("base_ch", 32),
                        scale_factor=ck.get("scale", 2)).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"[main] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")
    hicplus = load_optional_baseline(args.hicplus_ckpt or None, device)

    methods_base = ["LR", "Bicubic", "Gaussian", "SR-VAE"]
    if hicplus is not None:
        methods_base.append("HiCPlus")

    rows = []
    count, max_samples = 0, max(0, int(args.max_samples))
    for lr_b, hr_b in tqdm(loader, desc="eval"):
        lr_b, hr_b = lr_b.to(device), hr_b.to(device)
        sr_b, _, _ = model(lr_b, sample=False) if ck.get("model_name", "srvae") == "srvae" else (model(lr_b), None, None)
        hp_b = hicplus(lr_b) if hicplus is not None else None

        for b in range(lr_b.size(0)):
            lr_np = lr_b[b, 0].cpu().numpy()
            sr_np = sr_b[b, 0].clamp(0.0, 1.0).cpu().numpy()
            hr_np = hr_b[b, 0].cpu().numpy()
            sr_np, hr_np = (t.numpy() for t in center_crop_to_match(torch.from_numpy(sr_np), torch.from_numpy(hr_np)))
            hr_shape = hr_np.shape

            bicubic_np = bicubic_upsample(lr_np, hr_shape)
            gaussian_np = gaussian_baseline(lr_np, hr_shape, sigma=1.0)
            lr_up_np = bicubic_upsample(lr_np, hr_shape)  # LR shown at HR shape for fair scoring

            preds = {"LR": lr_up_np, "Bicubic": bicubic_np, "Gaussian": gaussian_np, "SR-VAE": sr_np}
            if hp_b is not None:
                preds["HiCPlus"] = hp_b[b, 0].clamp(0.0, 1.0).cpu().numpy()

            row = {"sample": count}
            for name in methods_base:
                p = preds[name]
                row[f"{name}_mse"] = mse_np(p, hr_np)
                row[f"{name}_ssim"] = ssim_np(p, hr_np, data_range=1.0)
                if not args.no_disco:
                    row[f"{name}_disco"] = genomedisco_score(p, hr_np)
                    row[f"{name}_hicspec"] = hicspector_score(p, hr_np)
            rows.append(row)

            if not args.no_plots and count < 32:
                panels = [preds[n] for n in methods_base] + [hr_np]
                titles = methods_base + ["HR"]
                vmax = float(np.clip(np.max([p.max() for p in panels]), 1e-6, 1.0))
                fig, axs = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6),
                                        constrained_layout=True)
                for ax, arr, title in zip(axs, panels, titles):
                    im = ax.imshow(arr, cmap="Reds", vmin=0.0, vmax=vmax, interpolation="nearest")
                    ax.set_title(title, fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.savefig(os.path.join(args.outdir, f"panel_{count:04d}.png"), dpi=args.dpi)
                plt.close(fig)
            count += 1
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break

    if not rows:
        print("[warn] no metrics computed")
        return

    fieldnames = ["sample"]
    suffixes = ["mse", "ssim"] + ([] if args.no_disco else ["disco", "hicspec"])
    for name in methods_base:
        for s in suffixes:
            fieldnames.append(f"{name}_{s}")

    csv_path = os.path.join(args.outdir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {csv_path}")

    print(f"\n[summary] N={count}")
    for name in methods_base:
        line = f"  {name:8s}  MSE={np.mean([r[f'{name}_mse'] for r in rows]):.4f}  SSIM={np.mean([r[f'{name}_ssim'] for r in rows]):.3f}"
        if not args.no_disco:
            line += f"  DISCO={np.mean([r[f'{name}_disco'] for r in rows]):.3f}  HiC-Spec={np.mean([r[f'{name}_hicspec'] for r in rows]):.3f}"
        print(line)


if __name__ == "__main__":
    main()
