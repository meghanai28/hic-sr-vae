import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
from model import SRVAE
from utils import normalize

 
def cosine_window_2d(size: int) -> np.ndarray:
    # 1D cosine window: 0 at edges, 1 at center
    t = np.linspace(0, np.pi, size, dtype=np.float32)
    w1d = (1.0 - np.cos(t)) / 2.0  # Hann window
    # outer product for 2D
    return np.outer(w1d, w1d)

def oe_normalize_full(mat: np.ndarray, eps: float = 1e-8, clip_range: tuple = (-2.0, 2.0)) -> np.ndarray:
    N = mat.shape[0]
    oe = np.zeros_like(mat, dtype=np.float32)
 
    for d in range(N):
        diag = np.diag(mat, k=d)
        if diag.size == 0:
            continue
        expected = max(float(np.mean(diag)), eps)
 
        # both diags filled
        idx = np.arange(N - d)
        oe[idx, idx + d] = mat[idx, idx + d] / expected - 1.0
        if d > 0:
            oe[idx + d, idx] = mat[idx + d, idx] / expected - 1.0
 
    return np.clip(oe, clip_range[0], clip_range[1])

@torch.no_grad()
def reconstruct_chromosome(model, lr_oe: np.ndarray, patch_size: int = 256, stride: int = 64, max_diag_bins: int = 200, device: str = "cpu") -> np.ndarray:
    N = lr_oe.shape[0]
    window = cosine_window_2d(patch_size)
 
    accumulator = np.zeros((N, N), dtype=np.float64)
    weight_map = np.zeros((N, N), dtype=np.float64)
 
    positions = []
    for i in range(0, N - patch_size + 1, stride):
        for j in range(i, min(N - patch_size + 1, i + max_diag_bins), stride):
            positions.append((i, j))
 
    print(f"[reconstruct] N={N}, patch={patch_size}, stride={stride}, "
          f"max_diag={max_diag_bins}, tiles={len(positions)}")
 
    for (i, j) in tqdm(positions, desc="reconstructing"):
        # extract tile from full matrix
        tile = lr_oe[i:i + patch_size, j:j + patch_size].copy()
 
        # tensor [1, 1, H, W]
        tile_t = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)
 
        # forward pass
        sr_t, _, _ = model(tile_t, sample=False)
        sr_np = sr_t[0, 0].cpu().numpy()

        h_out, w_out = sr_np.shape
        h_tile = min(h_out, patch_size)
        w_tile = min(w_out, patch_size)
        sr_np = sr_np[:h_tile, :w_tile]
        w = window[:h_tile, :w_tile]
 
        accumulator[i:i + h_tile, j:j + w_tile] += sr_np * w
        weight_map[i:i + h_tile, j:j + w_tile] += w
 
        if i != j:
            accumulator[j:j + w_tile, i:i + h_tile] += sr_np.T * w.T
            weight_map[j:j + w_tile, i:i + h_tile] += w.T
 
    mask = weight_map > 0
    result = np.zeros_like(accumulator, dtype=np.float32)
    result[mask] = (accumulator[mask] / weight_map[mask]).astype(np.float32)
    result[~mask] = lr_oe[~mask]
    result = 0.5 * (result + result.T)
 
    return result


def plot_comparison(lr_oe, sr_oe, hr_oe=None, chrom="", outdir=".", zoom_range=None, pclip=1.0):
    has_hr = hr_oe is not None
    ncol = 3 if has_hr else 2
 
    all_mats = [np.clip(lr_oe, 0, None), np.clip(sr_oe, 0, None)]
    if has_hr:
        all_mats.append(np.clip(hr_oe, 0, None))
    flat = np.concatenate([m.ravel() for m in all_mats])
    vmax = max(float(np.nanpercentile(flat[flat > 0], 100 - pclip)), 0.1)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axs = plt.subplots(1, ncol, figsize=(6 * ncol, 5.5), constrained_layout=True)
    panels = [np.clip(lr_oe, 0, None), np.clip(sr_oe, 0, None)]
    titles = [f"LR ({chrom})", f"SR-VAE ({chrom})"]
    if has_hr:
        panels.append(np.clip(hr_oe, 0, None))
        titles.append(f"HR ({chrom})")
 
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
        fig, axs = plt.subplots(1, ncol, figsize=(5 * ncol, 4.5), constrained_layout=True)
        zoom_panels = [np.clip(lr_oe[s:e, s:e], 0, None),
                       np.clip(sr_oe[s:e, s:e], 0, None)]
        zoom_titles = ["LR (zoomed)", "SR-VAE (zoomed)"]
        if has_hr:
            zoom_panels.append(np.clip(hr_oe[s:e, s:e], 0, None))
            zoom_titles.append("HR (zoomed)")
 
        zflat = np.concatenate([m.ravel() for m in zoom_panels])
        zvmax = max(float(np.nanpercentile(zflat[zflat > 0], 99)), 0.1)
        znorm = Normalize(vmin=0, vmax=zvmax)
 
        for ax, arr, title in zip(axs, zoom_panels, zoom_titles):
            im = ax.imshow(arr, cmap="Reds", norm=znorm, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
        fig.suptitle(f"Zoomed: {chrom} bins {s}-{e} "
                     f"({s*10//1000}-{e*10//1000} kb)", fontsize=11)
        path = os.path.join(outdir, f"{chrom}_zoom_{s}_{e}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"[saved] {path}")

def main():
    ap = argparse.ArgumentParser(description="Reconstruct full chromosome from SR-VAE")
    ap.add_argument("--mcool", required=True, help="Path to .mcool file")
    ap.add_argument("--lr-res", type=int, required=True, help="Low-resolution in bp")
    ap.add_argument("--hr-res", type=int, default=None, help="High-resolution in bp for comparison")
    ap.add_argument("--chrom", default="chr17", help="Chromosome to reconstruct")
    ap.add_argument("--ckpt", required=True, help="SR-VAE checkpoint path")
    ap.add_argument("--config", required=True, help="Config YAML")
    ap.add_argument("--outdir", default="./runs/sr_vae/reconstruction")
    ap.add_argument("--stride", type=int, default=64, help="Inference stride (smaller=smoother, 32-128)")
    ap.add_argument("--max-diag", type=int, default=200, help="Max distance from diagonal in bins")
    ap.add_argument("--patch", type=int, default=256, help="Tile size")
    ap.add_argument("--zoom-start", type=int, default=None, help="Start bin for zoomed view")
    ap.add_argument("--zoom-end", type=int, default=None, help="End bin for zoomed view")
    ap.add_argument("--save-npy", action="store_true", help="Save reconstructed matrix as .npy")
    ap.add_argument("--hr-mcool", default=None, help="Optional HR .mcool for comparison")
    args = ap.parse_args()
    hr_mcool_path = args.hr_mcool if args.hr_mcool is not None else args.mcool

 
    os.makedirs(args.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu")
    z_ch = int(ck.get("z_ch", cfg.get("vae", {}).get("z_ch", 64)))
    scale = int(ck.get("scale", cfg.get("srvae", {}).get("scale", 1)))
    base_ch = int(ck.get("base_ch", cfg.get("vae", {}).get("base_ch", 48)))
 
    model = SRVAE(z_ch=z_ch, scale_factor=scale, base_ch=base_ch).to(device).eval()
    model.load_state_dict(ck["model"], strict=False)
    print(f"[model] loaded {args.ckpt} (epoch {ck.get('epoch', '?')}, "
          f"base_ch={base_ch}, z_ch={z_ch})")
 
    try:
        import cooler
    except ImportError:
        print("[error] Install cooler: pip install cooler")
        sys.exit(1)
 
    c = cooler.Cooler(f"{args.mcool}::/resolutions/{args.lr_res}")
    chrom = args.chrom

    if chrom not in c.chromnames:
        alt = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        if alt in c.chromnames:
            chrom = alt
        else:
            print(f"[error] {args.chrom} not found. Available: {c.chromnames}")
            sys.exit(1)
 
    print(f"[data] Loading LR {chrom} at {args.lr_res}bp from {args.mcool}")
    lr_raw = c.matrix(balance=False).fetch(chrom).astype(np.float32)
    lr_raw = 0.5 * (lr_raw + lr_raw.T)  # enforce symmetry
    print(f"[data] Matrix size: {lr_raw.shape[0]}x{lr_raw.shape[1]}")
 
    print("[norm] OE-normalizing full chromosome...")
    lr_oe = oe_normalize_full(lr_raw)
    print(f"[norm] OE range: [{lr_oe.min():.2f}, {lr_oe.max():.2f}], "
          f"mean={lr_oe.mean():.3f}")

    sr_oe = reconstruct_chromosome(model, lr_oe, patch_size=args.patch, stride=args.stride, max_diag_bins=args.max_diag, device=device)
 
    hr_oe = None
    if args.hr_res is not None:
        c_hr = cooler.Cooler(f"{hr_mcool_path}::/resolutions/{args.hr_res}")
        hr_chrom = chrom
        if hr_chrom not in c_hr.chromnames:
            hr_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        hr_raw = c_hr.matrix(balance=False).fetch(hr_chrom).astype(np.float32)
        hr_raw = 0.5 * (hr_raw + hr_raw.T)
        hr_oe = oe_normalize_full(hr_raw)
        print(f"[data] HR matrix loaded: {hr_raw.shape}")

    if args.save_npy:
        npy_path = os.path.join(args.outdir, f"{chrom}_sr.npy")
        np.save(npy_path, sr_oe)
        print(f"[saved] {npy_path}")
 
        npy_lr_path = os.path.join(args.outdir, f"{chrom}_lr_oe.npy")
        np.save(npy_lr_path, lr_oe)
        print(f"[saved] {npy_lr_path}")

    zoom = None
    if args.zoom_start is not None and args.zoom_end is not None:
        zoom = (args.zoom_start, args.zoom_end)
    elif lr_oe.shape[0] > 512:
        # Auto-zoom: pick a 512-bin region in the middle
        mid = lr_oe.shape[0] // 2
        zoom = (mid - 256, mid + 256)
        print(f"[viz] Auto-zoom: bins {zoom[0]}-{zoom[1]}")
 
    plot_comparison(lr_oe, sr_oe, hr_oe=hr_oe, chrom=chrom,
                    outdir=args.outdir, zoom_range=zoom)
 
    band_mask = np.zeros_like(lr_oe, dtype=bool)
    N = lr_oe.shape[0]
    for d in range(min(args.max_diag, N)):
        idx = np.arange(N - d)
        band_mask[idx, idx + d] = True
        band_mask[idx + d, idx] = True
 
    lr_band = lr_oe[band_mask]
    sr_band = sr_oe[band_mask]
 
    print(f"\n[summary] {chrom}")
    print(f"  Matrix size:  {N}x{N} ({N * args.lr_res / 1e6:.1f} Mb)")
    print(f"  Tiles used:   {len([(i,j) for i in range(0, N-args.patch+1, args.stride) for j in range(i, min(N-args.patch+1, i+args.max_diag), args.stride)])}")
    print(f"  LR band std:  {lr_band.std():.4f}")
    print(f"  SR band std:  {sr_band.std():.4f}")
 
    if hr_oe is not None and hr_oe.shape != sr_oe.shape:
        print(f"[warning] HR shape {hr_oe.shape} does not match SR shape {sr_oe.shape}. Skipping HR comparison.")
        hr_oe = None

    if hr_oe is not None:
        hr_band = hr_oe[band_mask]
        mse = float(np.mean((sr_band - hr_band) ** 2))
        # per-diagonal correlation
        corrs = []
        for d in range(min(50, N)):
            sr_diag = np.diag(sr_oe, k=d)
            hr_diag = np.diag(hr_oe, k=d)
            if len(sr_diag) < 3:
                continue
            r = np.corrcoef(sr_diag, hr_diag)[0, 1]
            if not np.isnan(r):
                corrs.append(r)
        print(f"  MSE vs HR:    {mse:.4f}")
        print(f"  DiagCorr(50): {np.mean(corrs):.3f}")
    print("\n[done]")
 
if __name__ == "__main__":
    main()