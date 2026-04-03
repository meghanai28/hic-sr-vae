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


def _positions(N, patch, stride):
    i_vals = sorted(set(range(0, N - patch + 1, stride)) | {max(0, N - patch)})
    positions = []
    for i in i_vals:
        j_hi = N - patch + 1
        j_vals = sorted(set(range(i, j_hi, stride)) | {max(i, j_hi - 1)} if j_hi > i else set())
        for j in j_vals:
            positions.append((i, j))
    return positions


@torch.no_grad()
def reconstruct(model, lr_raw, patch, stride, device):
    N = lr_raw.shape[0]
    accumulator = np.zeros((N, N), dtype=np.float32)
    weight_map  = np.zeros((N, N), dtype=np.float32)
    window = cosine_window_2d(patch)

    positions = _positions(N, patch, stride)
    print(f"[reconstruct] {len(positions)} tiles, {N}x{N} bins")

    for (i, j) in tqdm(positions, leave=False):
        tile = lr_raw[i:i + patch, j:j + patch].copy().astype(np.float32)
        tile_t = normalize(torch.from_numpy(tile).unsqueeze(0).unsqueeze(0), "oe").to(device)

        sr_t, _, _ = model(tile_t, sample=False)
        sr_np = sr_t[0, 0].cpu().numpy()

        h = min(sr_np.shape[0], patch)
        w = min(sr_np.shape[1], patch)
        sr_np = sr_np[:h, :w]
        wnd   = window[:h, :w]

        accumulator[i:i + h, j:j + w] += sr_np * wnd
        weight_map [i:i + h, j:j + w] += wnd

        if i != j:
            accumulator[j:j + w, i:i + h] += sr_np.T * wnd.T
            weight_map [j:j + w, i:i + h] += wnd.T

    mask = weight_map > 0
    result = np.zeros((N, N), dtype=np.float32)
    result[mask] = accumulator[mask] / weight_map[mask]
    return result


def plot_comparison(lr_oe, sr_oe, title, outpath, pclip=1.0):
    panels = [np.clip(lr_oe, 0, None), np.clip(sr_oe, 0, None)]
    titles = ["LR", "SR-VAE"]
    flat = np.concatenate([m.ravel() for m in panels])
    vmax = max(float(np.nanpercentile(flat[flat > 0], 100 - pclip)), 0.1)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, arr, t in zip(axs, panels, titles):
        im = ax.imshow(arr, cmap="Reds", norm=norm, interpolation="nearest")
        ax.set_title(t, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[saved] {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcool",    required=True)
    ap.add_argument("--res",      type=int, default=10000)
    ap.add_argument("--chrom",    default="chr17")
    ap.add_argument("--ckpt",     required=True)
    ap.add_argument("--config",   required=True)
    ap.add_argument("--outdir",   default="./runs/sr_vae/reconstruction")
    ap.add_argument("--patch",    type=int, default=256)
    ap.add_argument("--stride",   type=int, default=128)
    ap.add_argument("--save-npy", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg    = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck      = torch.load(args.ckpt, map_location="cpu")
    z_ch    = int(ck.get("z_ch",    cfg.get("vae", {}).get("z_ch", 64)))
    scale   = int(ck.get("scale",   cfg.get("srvae", {}).get("scale", 1)))
    base_ch = int(ck.get("base_ch", cfg.get("vae", {}).get("base_ch", 64)))

    model = SRVAE(z_ch=z_ch, scale_factor=scale, base_ch=base_ch).to(device).eval()
    model.load_state_dict(ck["model"], strict=False)
    print(f"[model] loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    try:
        import cooler
    except ImportError:
        print("[error] pip install cooler"); sys.exit(1)

    c = cooler.Cooler(f"{args.mcool}::/resolutions/{args.res}")
    chrom = args.chrom
    if chrom not in c.chromnames:
        alt = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        if alt in c.chromnames:
            chrom = alt
        else:
            print(f"[error] {args.chrom} not in {c.chromnames}"); sys.exit(1)

    lr_raw = c.matrix(balance=False).fetch(chrom).astype(np.float32)
    lr_raw = 0.5 * (lr_raw + lr_raw.T)
    N = lr_raw.shape[0]
    print(f"[data] {chrom} at {args.res}bp -> {N}x{N} bins")

    sr = reconstruct(model, lr_raw, args.patch, args.stride, device)

    lr_oe = oe_normalize_full(lr_raw)
    sr_oe = oe_normalize_full(sr)

    plot_comparison(lr_oe, sr_oe,
                    title=f"{args.chrom} @ {args.res // 1000}kb",
                    outpath=os.path.join(args.outdir, f"{args.chrom}_{args.res // 1000}kb.png"))

    if args.save_npy:
        np.save(os.path.join(args.outdir, f"{args.chrom}_sr.npy"), sr)
        print(f"[saved] npy to {args.outdir}")

    print(f"\n[done] {args.chrom}")


if __name__ == "__main__":
    main()
