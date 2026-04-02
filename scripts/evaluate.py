import os
import sys
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import make_loaders
from model import SRVAE
from utils import center_crop_to_match


def mse_np(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def ssim_np(x, y, k=11):
    C1, C2 = 0.01**2, 0.03**2
    x, y = x.astype(np.float32), y.astype(np.float32)

    def box_mean(img, k):
        if k <= 1:
            return img.copy()
        pad = k // 2
        a = np.pad(img, pad, mode="edge")
        ii = np.cumsum(np.cumsum(a, 0), 1)
        return (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / (k * k)

    mu_x, mu_y = box_mean(x, k), box_mean(y, k)
    sig_x2 = box_mean(x * x, k) - mu_x**2
    sig_y2 = box_mean(y * y, k) - mu_y**2
    sig_xy = box_mean(x * y, k) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x2 + sig_y2 + C2)
    return float(np.clip(np.mean(num / (den + 1e-12)), -1, 1))


def shared_range(arrays, pclip=1.0):
    flat = np.concatenate([a.ravel() for a in arrays])
    vmax = float(np.nanpercentile(flat, 100 - pclip))
    return 0.0, max(vmax, 1e-6)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="./runs/sr_vae/sr_vae_best.pt")
    ap.add_argument("--outdir", default="./runs/sr_vae/eval")
    ap.add_argument("--pclip", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_ld, test_ld = make_loaders(cfg, verbose=True)
    loader = test_ld or val_ld
    assert loader, "No val or test data"

    ck = torch.load(args.ckpt, map_location="cpu")
    z_ch      = int(ck.get("z_ch",      cfg.get("vae", {}).get("z_ch", 64)))
    scale     = int(ck.get("scale",     cfg.get("srvae", {}).get("scale", 1)))
    base_ch   = int(ck.get("base_ch",   cfg.get("vae", {}).get("base_ch", 48)))
    num_zooms = int(ck.get("num_zooms", cfg.get("num_zooms", 3)))

    model = SRVAE(z_ch=z_ch, scale_factor=scale, base_ch=base_ch, num_zooms=num_zooms).to(device).eval()
    model.load_state_dict(ck["model"], strict=False)
    print(f"[eval] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    metrics = []
    count = 0

    for lr_b, hr_b, zoom_idx_b in tqdm(loader, desc="eval"):
        lr_b, hr_b = lr_b.to(device), hr_b.to(device)
        zoom_idx_b = zoom_idx_b.to(device)
        pred, _, _ = model(lr_b, zoom_idx_b, sample=False)

        for b in range(lr_b.size(0)):
            lr_np = lr_b[b, 0].cpu().numpy()
            sr_np = pred[b, 0].cpu().numpy()
            hr_np = hr_b[b, 0].cpu().numpy()

            sr_c, hr_c = center_crop_to_match(
                torch.from_numpy(sr_np), torch.from_numpy(hr_np)
            )
            sr_c, hr_c = sr_c.numpy(), hr_c.numpy()
            lr_c, _ = center_crop_to_match(
                torch.from_numpy(lr_np), torch.from_numpy(sr_c)
            )
            lr_c = lr_c.numpy()

            m = mse_np(sr_c, hr_c)
            s = ssim_np(sr_c, hr_c)
            metrics.append(dict(mse=m, ssim=s))

            pos_panels = [np.clip(lr_c, 0, None), np.clip(sr_c, 0, None), np.clip(hr_c, 0, None)]
            titles = ["LR (input)", "SR-VAE (predicted)", "HR (ground truth)"]
            vmin, vmax = shared_range(pos_panels, pclip=args.pclip)
            norm = Normalize(vmin=vmin, vmax=vmax)

            fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.5), constrained_layout=True)
            for ax, arr, title in zip(axs, pos_panels, titles):
                im = ax.imshow(arr, cmap="Reds", norm=norm, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"MSE={m:.4f}  SSIM={s:.3f}", fontsize=10)
            fig.savefig(os.path.join(args.outdir, f"triptych_{count:04d}.png"), dpi=args.dpi)
            plt.close(fig)
            count += 1

    if metrics:
        print(f"\n[summary] {count} samples")
        print(f"  MSE:  {np.mean([m['mse'] for m in metrics]):.4f}")
        print(f"  SSIM: {np.mean([m['ssim'] for m in metrics]):.3f}")


if __name__ == "__main__":
    main()
