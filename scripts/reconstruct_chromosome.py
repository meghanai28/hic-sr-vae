import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import SRVAE
from utils import normalize
from datasets import ZOOM_TO_IDX


def cosine_window_2d(size: int) -> np.ndarray:
    t = np.linspace(0, np.pi, size, dtype=np.float32)
    w1d = (1.0 - np.cos(t)) / 2.0
    return np.outer(w1d, w1d)


def oe_normalize_full(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    N = mat.shape[0]
    oe = np.zeros_like(mat, dtype=np.float32)
    for d in range(N):
        diag = np.diag(mat, k=d)
        if diag.size == 0:
            continue
        expected = max(float(np.mean(diag)), eps)
        idx = np.arange(N - d)
        oe[idx, idx + d] = mat[idx, idx + d] / expected - 1.0
        if d > 0:
            oe[idx + d, idx] = mat[idx + d, idx] / expected - 1.0
    return np.clip(oe, -2.0, 2.0)


def _downsample(arr, factor):
    H, W = arr.shape
    H2, W2 = H // factor, W // factor
    return arr[:H2 * factor, :W2 * factor].reshape(H2, factor, W2, factor).mean(axis=(1, 3))


@torch.no_grad()
def reconstruct_chromosome(model, lr_raw: np.ndarray, patch: int, stride: int,
                            zooms: list[int], device: str) -> np.ndarray:
    N = lr_raw.shape[0]
    result = np.zeros((N, N), dtype=np.float32)

    # coarse to fine: each zoom covers its distance band, finer overwrites coarser
    for zoom in sorted(zooms, reverse=True):
        zoom_idx_t = torch.tensor([ZOOM_TO_IDX[zoom]], dtype=torch.long, device=device)
        win  = patch * zoom
        step = win  # non-overlapping for speed; cosine window handles blending

        band_lo = patch * (zoom // 2) if zoom > 1 else 0
        band_hi = patch * zoom

        accumulator = np.zeros((N, N), dtype=np.float32)
        weight_map  = np.zeros((N, N), dtype=np.float32)
        window = cosine_window_2d(win)

        # tile all (i, j) in upper triangle where j-i falls in this zoom's band
        positions = []
        for i in range(0, N - win + 1, step):
            for dj in range(band_lo, band_hi, win):
                j = i + dj
                if j + win <= N:
                    positions.append((i, j))

        print(f"[zoom={zoom}x] {len(positions)} tiles  band=[{band_lo},{band_hi}) bins  win={win}")

        for (i, j) in tqdm(positions, desc=f"zoom={zoom}x", leave=False):
            raw_tile = lr_raw[i:i + win, j:j + win].copy()

            if zoom > 1:
                coarse = _downsample(raw_tile, zoom).astype(np.float32)
            else:
                coarse = raw_tile.astype(np.float32)

            tile_t = normalize(
                torch.from_numpy(coarse).unsqueeze(0).unsqueeze(0), "oe"
            ).to(device)

            sr_t, _, _ = model(tile_t, zoom_idx_t, sample=False)
            sr_np = sr_t[0, 0].cpu().numpy()

            if zoom > 1:
                sr_np = F.interpolate(
                    torch.from_numpy(sr_np).unsqueeze(0).unsqueeze(0).float(),
                    size=(win, win),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()

            h = min(sr_np.shape[0], win)
            w = min(sr_np.shape[1], win)
            sr_np = sr_np[:h, :w]
            wnd   = window[:h, :w]

            accumulator[i:i + h, j:j + w] += sr_np * wnd
            weight_map [i:i + h, j:j + w] += wnd

            # mirror to lower triangle
            if i != j:
                accumulator[j:j + w, i:i + h] += sr_np.T * wnd.T
                weight_map [j:j + w, i:i + h] += wnd.T

        mask = weight_map > 0
        result[mask] = accumulator[mask] / weight_map[mask]

    return result


def plot_comparison(lr_oe, sr_oe, hr_oe=None, chrom="", outdir=".", zoom_range=None, pclip=1.0):
    has_hr = hr_oe is not None
    ncol   = 3 if has_hr else 2

    panels = [np.clip(lr_oe, 0, None), np.clip(sr_oe, 0, None)]
    titles = [f"LR ({chrom})", f"SR-VAE ({chrom})"]
    if has_hr:
        panels.append(np.clip(hr_oe, 0, None))
        titles.append(f"HR ({chrom})")

    flat = np.concatenate([m.ravel() for m in panels])
    vmax = max(float(np.nanpercentile(flat[flat > 0], 100 - pclip)), 0.1)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axs = plt.subplots(1, ncol, figsize=(6 * ncol, 5.5), constrained_layout=True)
    for ax, arr, title in zip(axs, panels, titles):
        im = ax.imshow(arr, cmap="Reds", norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Full chromosome reconstruction: {chrom}", fontsize=13)
    path = os.path.join(outdir, f"{chrom}_full.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")

    if zoom_range is not None:
        s, e = zoom_range
        zpanels = [np.clip(lr_oe[s:e, s:e], 0, None), np.clip(sr_oe[s:e, s:e], 0, None)]
        ztitles = ["LR (zoomed)", "SR-VAE (zoomed)"]
        if has_hr:
            zpanels.append(np.clip(hr_oe[s:e, s:e], 0, None))
            ztitles.append("HR (zoomed)")

        zflat = np.concatenate([m.ravel() for m in zpanels])
        zvmax = max(float(np.nanpercentile(zflat[zflat > 0], 99)), 0.1)
        znorm = Normalize(vmin=0, vmax=zvmax)

        fig, axs = plt.subplots(1, ncol, figsize=(5 * ncol, 4.5), constrained_layout=True)
        for ax, arr, title in zip(axs, zpanels, ztitles):
            im = ax.imshow(arr, cmap="Reds", norm=znorm, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Zoomed: {chrom} bins {s}-{e}", fontsize=11)
        path = os.path.join(outdir, f"{chrom}_zoom_{s}_{e}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcool",    required=True)
    ap.add_argument("--lr-res",   type=int, required=True)
    ap.add_argument("--hr-res",   type=int, default=None)
    ap.add_argument("--chrom",    default="chr17")
    ap.add_argument("--ckpt",     required=True)
    ap.add_argument("--config",   required=True)
    ap.add_argument("--outdir",   default="./runs/sr_vae/reconstruction")
    ap.add_argument("--patch",    type=int, default=256)
    ap.add_argument("--stride",   type=int, default=64, help="Stride in patch units (applied per zoom)")
    ap.add_argument("--zooms",    default="1,2,4")
    ap.add_argument("--zoom-start", type=int, default=None)
    ap.add_argument("--zoom-end",   type=int, default=None)
    ap.add_argument("--save-npy", action="store_true")
    ap.add_argument("--hr-mcool", default=None)
    args = ap.parse_args()

    zooms = [int(z) for z in args.zooms.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    cfg    = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck        = torch.load(args.ckpt, map_location="cpu")
    z_ch      = int(ck.get("z_ch",      cfg.get("vae", {}).get("z_ch", 64)))
    scale     = int(ck.get("scale",     cfg.get("srvae", {}).get("scale", 1)))
    base_ch   = int(ck.get("base_ch",   cfg.get("vae", {}).get("base_ch", 48)))
    num_zooms = int(ck.get("num_zooms", cfg.get("num_zooms", 3)))

    model = SRVAE(z_ch=z_ch, scale_factor=scale, base_ch=base_ch, num_zooms=num_zooms).to(device).eval()
    model.load_state_dict(ck["model"], strict=False)
    print(f"[model] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    try:
        import cooler
    except ImportError:
        print("[error] Install cooler: pip install cooler")
        sys.exit(1)

    c     = cooler.Cooler(f"{args.mcool}::/resolutions/{args.lr_res}")
    chrom = args.chrom
    if chrom not in c.chromnames:
        alt = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        if alt in c.chromnames:
            chrom = alt
        else:
            print(f"[error] {args.chrom} not found. Available: {c.chromnames}")
            sys.exit(1)

    print(f"[data] loading {chrom} at {args.lr_res}bp")
    lr_raw = c.matrix(balance=False).fetch(chrom).astype(np.float32)
    lr_raw = 0.5 * (lr_raw + lr_raw.T)
    N = lr_raw.shape[0]
    print(f"[data] {N}x{N} bins")

    lr_oe = oe_normalize_full(lr_raw)
    sr = reconstruct_chromosome(model, lr_raw, patch=args.patch,
                                stride=args.stride, zooms=zooms, device=device)

    hr_oe = None
    hr_mcool_path = args.hr_mcool or args.mcool
    if args.hr_res is not None:
        c_hr = cooler.Cooler(f"{hr_mcool_path}::/resolutions/{args.hr_res}")
        hr_chrom = chrom
        if hr_chrom not in c_hr.chromnames:
            hr_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        hr_raw = c_hr.matrix(balance=False).fetch(hr_chrom).astype(np.float32)
        hr_raw = 0.5 * (hr_raw + hr_raw.T)
        hr_oe  = oe_normalize_full(hr_raw)
        if hr_oe.shape != sr.shape:
            print(f"[warning] HR shape {hr_oe.shape} != SR shape {sr.shape}, skipping HR comparison")
            hr_oe = None

    if args.save_npy:
        np.save(os.path.join(args.outdir, f"{chrom}_sr.npy"),     sr)
        np.save(os.path.join(args.outdir, f"{chrom}_lr_oe.npy"),  lr_oe)
        print(f"[saved] npy files to {args.outdir}")

    zoom_range = None
    if args.zoom_start is not None and args.zoom_end is not None:
        zoom_range = (args.zoom_start, args.zoom_end)
    elif N > 512:
        mid = N // 2
        zoom_range = (mid - 256, mid + 256)

    plot_comparison(lr_oe, sr, hr_oe=hr_oe, chrom=chrom,
                    outdir=args.outdir, zoom_range=zoom_range)

    print(f"\n[summary] {chrom}  {N}x{N} bins  ({N * args.lr_res / 1e6:.1f} Mb)")
    if hr_oe is not None:
        mse = float(np.mean((sr - hr_oe) ** 2))
        print(f"  MSE vs HR: {mse:.4f}")
    print("[done]")


if __name__ == "__main__":
    main()
