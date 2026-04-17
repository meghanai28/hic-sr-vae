"""Generate LR tiles by binomial-thinning + spatial 2x downsampling of HR tiles.

LR tile = avg_pool2x( binomial_thin(HR, frac) ).
Filename and chromosome stay the same as the HR tile (different output dir).
"""

import argparse
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import avg_pool2d_np, binomial_thin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-glob", required=True, help="Glob for HR tiles, e.g. tiles/hr/train/*.npy")
    ap.add_argument("--out", required=True, help="Output dir for LR tiles")
    ap.add_argument("--frac", type=float, default=0.0625, help="Read-keep fraction (default 1/16)")
    ap.add_argument("--scale", type=int, default=2, help="Spatial downsample factor for LR (default 2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    paths = sorted(glob.glob(args.hr_glob))
    if not paths:
        raise SystemExit(f"glob matched 0 files: {args.hr_glob}")

    os.makedirs(args.out, exist_ok=True)
    print(f"[info] {len(paths)} HR tiles, frac={args.frac}, scale={args.scale}, seed={args.seed}")

    for p in tqdm(paths, desc="thin+downsample"):
        hr = np.load(p).astype(np.float32)
        thin = binomial_thin(hr, args.frac)
        lr = avg_pool2d_np(thin, args.scale)
        np.save(os.path.join(args.out, os.path.basename(p)), lr.astype(np.float32))

    print(f"[done] {len(paths)} LR tiles -> {args.out}")


if __name__ == "__main__":
    main()
