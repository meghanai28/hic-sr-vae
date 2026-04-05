import os
import sys
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
from utils import normalize
from datasets import _glob_sorted
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--outdir", default="./data_sanity_check")
    args = ap.parse_args()
 
    os.makedirs(args.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    data_cfg = cfg.get("data", {})
 
    lr_paths = _glob_sorted(data_cfg.get("val_lr"))
    hr_paths = _glob_sorted(data_cfg.get("val_hr"))
    norm_mode = data_cfg.get("norm", "oe")
 
    if not lr_paths or not hr_paths:
        print("[error] No val data found. Check config paths.")
        return
 
    n = min(args.n, len(lr_paths), len(hr_paths))
    print(f"[viz] {n} samples, norm={norm_mode}")
 
    for i in range(n):
        hr_raw = np.load(hr_paths[i]).astype(np.float32)
        lr_raw = np.load(lr_paths[i]).astype(np.float32)
        hr_oe = normalize(torch.from_numpy(hr_raw).unsqueeze(0), norm_mode)[0].numpy()
        lr_oe = normalize(torch.from_numpy(lr_raw).unsqueeze(0), norm_mode)[0].numpy()
 
        print(f"\n--- Sample {i} ({os.path.basename(hr_paths[i])}) ---")
        print(f"  HR raw: min={hr_raw.min():.1f} max={hr_raw.max():.1f} "
              f"zeros={100*(hr_raw==0).mean():.1f}%")
        print(f"  HR OE:  min={hr_oe.min():.3f} max={hr_oe.max():.3f} "
              f"mean={hr_oe.mean():.3f}")
        print(f"  LR OE:  min={lr_oe.min():.3f} max={lr_oe.max():.3f} "
              f"mean={lr_oe.mean():.3f}")
 
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

        im0 = axs[0].imshow(np.maximum(hr_raw, 0.5), cmap="YlOrRd",
                             norm=LogNorm(), interpolation="nearest")
        axs[0].set_title("HR raw (log scale)")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
 
        hr_pos = np.clip(hr_oe, 0, None)
        lr_pos = np.clip(lr_oe, 0, None)
        oe_all = np.concatenate([hr_pos.ravel(), lr_pos.ravel()])
        vmax = max(float(np.nanpercentile(oe_all, 99)), 0.1)
        oe_norm = Normalize(vmin=0, vmax=vmax)
 
        im1 = axs[1].imshow(hr_pos, cmap="Reds", norm=oe_norm, interpolation="nearest")
        axs[1].set_title("HR (OE, red/white)")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
 
        im2 = axs[2].imshow(lr_pos, cmap="Reds", norm=oe_norm, interpolation="nearest")
        axs[2].set_title("LR (OE, red/white)")
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
 
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
 
        fig.suptitle(os.path.basename(hr_paths[i]), fontsize=11)
        out_path = os.path.join(args.outdir, f"sanity_{i:03d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
 
    print(f"\n[done] {n} plots → {args.outdir}")
 
 
if __name__ == "__main__":
    main()
