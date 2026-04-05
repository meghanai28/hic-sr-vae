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
from utils import normalize, binomial_thin
from datasets import ZOOM_TO_IDX

BANDS = [
    {"band": (0, 256),     "zoom": 1,  "stride": 64},
    {"band": (256, 512),   "zoom": 1,  "stride": 128},
    {"band": (512, 1024),  "zoom": 2,  "stride": 512},
    {"band": (1024, 2048), "zoom": 4,  "stride": 1024},
    {"band": (2048, 4096), "zoom": 8,  "stride": 2048},
    {"band": (4096, 8192), "zoom": 16, "stride": 4096},
]


def cosine_window_2d(size: int) -> np.ndarray:
    t = np.linspace(0, np.pi, size, dtype=np.float32)
    w1d = (1.0 - np.cos(t)) / 2.0
    return np.outer(w1d, w1d)


def _downsample(mat, factor):
    h, w = mat.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return mat[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


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


@torch.no_grad()
def reconstruct(model, lr_raw, patch, device):
    N = lr_raw.shape[0]
    accumulator = np.zeros((N, N), dtype=np.float32)
    weight_map  = np.zeros((N, N), dtype=np.float32)
    window_small = cosine_window_2d(patch)

    all_positions = []
    for band_cfg in BANDS:
        blo, bhi = band_cfg["band"]
        zoom = band_cfg["zoom"]
        stride = max(band_cfg["stride"] // 2, patch)
        win = patch * zoom

        if blo >= N:
            continue

        zoom_idx_t = torch.tensor([ZOOM_TO_IDX[zoom]], dtype=torch.long, device=device)

        positions = []
        for i in range(0, N, stride):
            for dj in range(blo, min(bhi, N), stride):
                j = i + dj
                if j >= N:
                    break
                positions.append((i, j))

        print(f"[band {blo}-{bhi} z{zoom}] {len(positions)} tiles  stride={stride}")

        for (i, j) in tqdm(positions, desc=f"band{blo}-{bhi}", leave=False):
            ih = min(win, N - i)
            jw = min(win, N - j)
            tile = np.zeros((win, win), dtype=np.float32)
            tile[:ih, :jw] = lr_raw[i:i + ih, j:j + jw]

            if zoom > 1:
                coarse = _downsample(tile, zoom).astype(np.float32)
            else:
                coarse = tile.astype(np.float32)

            tile_t = normalize(
                torch.from_numpy(coarse).unsqueeze(0).unsqueeze(0), "oe"
            ).to(device)

            sr_t, _, _ = model(tile_t, zoom_idx_t, sample=False)
            sr_np = sr_t[0, 0].cpu().numpy()

            if zoom > 1:
                up_h = min(patch, (ih + zoom - 1) // zoom)
                up_w = min(patch, (jw + zoom - 1) // zoom)
                sr_np = F.interpolate(
                    torch.from_numpy(sr_np[:up_h, :up_w]).unsqueeze(0).unsqueeze(0).float(),
                    size=(up_h * zoom, up_w * zoom), mode="bilinear", align_corners=False,
                ).squeeze().numpy()
                wnd = cosine_window_2d(up_h * zoom) if up_h == up_w else np.outer(
                    (1.0 - np.cos(np.linspace(0, np.pi, up_h * zoom))) / 2.0,
                    (1.0 - np.cos(np.linspace(0, np.pi, up_w * zoom))) / 2.0,
                )
            else:
                wnd = window_small[:patch, :patch].copy()

            h = min(sr_np.shape[0], ih)
            w = min(sr_np.shape[1], jw)
            sr_np = sr_np[:h, :w]
            wnd = wnd[:h, :w]

            accumulator[i:i + h, j:j + w] += sr_np * wnd
            weight_map [i:i + h, j:j + w] += wnd

            if i != j:
                accumulator[j:j + w, i:i + h] += sr_np.T * wnd.T
                weight_map [j:j + w, i:i + h] += wnd.T

    mask = weight_map > 0
    result = np.zeros((N, N), dtype=np.float32)
    result[mask] = accumulator[mask] / weight_map[mask]
    return result


def plot_panels(panels, titles, suptitle, outpath, pclip=1.0):
    ncols = len(panels)
    figw = 5.5 * ncols + 1
    flat = np.concatenate([m.ravel() for m in panels])
    vmax = max(float(np.nanpercentile(flat[flat > 0], 100 - pclip)), 0.1)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axs = plt.subplots(1, ncols, figsize=(figw, 5.5), constrained_layout=True)
    if ncols == 1:
        axs = [axs]
    for ax, arr, t in zip(axs, panels, titles):
        im = ax.imshow(arr, cmap="Reds", norm=norm, interpolation="nearest")
        ax.set_title(t, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[saved] {outpath}")


def hicrep_scc(mat_a, mat_b, max_dist_bp=500000, res=10000, smooth=5):
    import scipy.sparse as sp
    from scipy.ndimage import uniform_filter
    a = uniform_filter(mat_a.astype(np.float64), size=2*smooth+1)
    b = uniform_filter(mat_b.astype(np.float64), size=2*smooth+1)
    n_diags = max_dist_bp // res
    m1 = sp.coo_matrix(a)
    m2 = sp.coo_matrix(b)
    try:
        from hicrep.hicrep import sccByDiag
        return float(sccByDiag(m1, m2, n_diags))
    except Exception:
        N = a.shape[0]
        scores, weights = [], []
        for d in range(min(n_diags, N)):
            da = np.diag(a, k=d)
            db = np.diag(b, k=d)
            if len(da) < 3:
                break
            va, vb = np.var(da), np.var(db)
            if va < 1e-12 or vb < 1e-12:
                continue
            r = float(np.corrcoef(da, db)[0, 1])
            if np.isnan(r):
                continue
            scores.append(r)
            weights.append(len(da) * np.sqrt(va * vb))
        if not scores:
            return 0.0
        return float(np.average(np.array(scores), weights=np.array(weights)))


def _row_normalize(mat):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def genomedisco(mat_a, mat_b, t_steps=3, subsample=2000):
    a = np.clip(mat_a, 0, None).astype(np.float64)
    b = np.clip(mat_b, 0, None).astype(np.float64)
    N = a.shape[0]
    if N > subsample:
        step = N // subsample
        idx = np.arange(0, N, step)[:subsample]
        a = a[np.ix_(idx, idx)]
        b = b[np.ix_(idx, idx)]
    Ta = _row_normalize(a)
    Tb = _row_normalize(b)
    scores = []
    for _ in range(t_steps):
        Ta = Ta @ Ta
        Tb = Tb @ Tb
        diff = np.sum(np.abs(Ta - Tb)) / (2.0 * Ta.shape[0])
        scores.append(1.0 - float(diff))
    return float(np.mean(scores))


def per_distance_mse(mat_a, mat_b, max_dist=100):
    N = mat_a.shape[0]
    results = []
    for d in range(min(max_dist, N)):
        a = np.diag(mat_a, k=d)
        b = np.diag(mat_b, k=d)
        if len(a) < 1:
            break
        results.append(float(np.mean((a - b) ** 2)))
    return results


def ssim_full(x, y, k=11):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcool",    required=True)
    ap.add_argument("--res",      type=int, default=10000)
    ap.add_argument("--chrom",    default="chr17")
    ap.add_argument("--ckpt",     required=True)
    ap.add_argument("--config",   required=True)
    ap.add_argument("--outdir",   default="./runs/sr_vae/reconstruction")
    ap.add_argument("--patch",    type=int, default=256)
    ap.add_argument("--save-npy", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg    = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck      = torch.load(args.ckpt, map_location="cpu")
    z_ch    = int(ck.get("z_ch",    cfg.get("vae", {}).get("z_ch", 64)))
    scale   = int(ck.get("scale",   cfg.get("srvae", {}).get("scale", 1)))
    base_ch = int(ck.get("base_ch", cfg.get("vae", {}).get("base_ch", 64)))
    num_zooms = int(ck.get("num_zooms", cfg.get("num_zooms", 6)))

    model = SRVAE(z_ch=z_ch, scale_factor=scale, base_ch=base_ch, num_zooms=num_zooms).to(device).eval()
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

    hr_raw = c.matrix(balance=False).fetch(chrom).astype(np.float32)
    hr_raw = 0.5 * (hr_raw + hr_raw.T)
    N = hr_raw.shape[0]
    print(f"[data] {chrom} at {args.res}bp -> {N}x{N} bins")

    lr_raw = binomial_thin(hr_raw, frac=1/16)
    print(f"[thinned] 1/16 fraction -> LR input")

    sr = reconstruct(model, lr_raw, args.patch, device)

    lr_oe = oe_normalize_full(lr_raw)
    sr_oe = sr
    hr_oe = oe_normalize_full(hr_raw)

    plot_panels(
        [np.clip(lr_oe, 0, None), np.clip(sr_oe, 0, None), np.clip(hr_oe, 0, None)],
        [f"LR (1/16 thinned)", "SR-VAE", "HR (ground truth)"],
        suptitle=f"{args.chrom} @ {args.res // 1000}kb — OE normalized",
        outpath=os.path.join(args.outdir, f"{args.chrom}_oe.png"),
    )

    plot_panels(
        [np.log1p(lr_raw), np.log1p(hr_raw)],
        ["LR (1/16 thinned)", "HR (ground truth)"],
        suptitle=f"{args.chrom} @ {args.res // 1000}kb — log1p raw counts (depth comparison)",
        outpath=os.path.join(args.outdir, f"{args.chrom}_raw.png"),
    )

    mse_val = float(np.mean((sr_oe - hr_oe) ** 2))
    ssim_val = ssim_full(sr_oe, hr_oe)
    scc_sr = hicrep_scc(sr_oe, hr_oe)
    scc_lr = hicrep_scc(lr_oe, hr_oe)
    disco_sr = genomedisco(sr_oe, hr_oe)
    disco_lr = genomedisco(lr_oe, hr_oe)

    print(f"\n[metrics] SR-VAE vs HR:")
    print(f"  MSE:          {mse_val:.4f}")
    print(f"  SSIM:         {ssim_val:.3f}")
    print(f"  HiCRep SCC:   {scc_sr:.3f}")
    print(f"  GenomeDISCO:  {disco_sr:.3f}")
    print(f"\n[metrics] LR vs HR (baseline):")
    print(f"  HiCRep SCC:   {scc_lr:.3f}")
    print(f"  GenomeDISCO:  {disco_lr:.3f}")

    dist_mse = per_distance_mse(sr_oe, hr_oe, max_dist=100)
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.plot(dist_mse, linewidth=1.2)
    ax.set_xlabel("Genomic distance (bins)")
    ax.set_ylabel("MSE")
    ax.set_title(f"{args.chrom} — per-distance MSE (SR-VAE vs HR)")
    fig.savefig(os.path.join(args.outdir, f"{args.chrom}_distance_mse.png"), dpi=150)
    plt.close(fig)
    print(f"[saved] {args.chrom}_distance_mse.png")

    if args.save_npy:
        np.save(os.path.join(args.outdir, f"{args.chrom}_sr.npy"), sr)
        print(f"[saved] npy to {args.outdir}")

    print(f"\n[done] {args.chrom}")


if __name__ == "__main__":
    main()
