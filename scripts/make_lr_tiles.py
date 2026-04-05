import os
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import binomial_thin


def main():
    ap = argparse.ArgumentParser(description="Generate LR tiles via binomial thinning")
    ap.add_argument("--hr-glob", required=True, help="Glob pattern for HR tiles (e.g. 'tiles/hr/train/*.npy')")
    ap.add_argument("--out", required=True, help="Output directory for LR tiles")
    ap.add_argument("--frac", type=float, default=0.0625, help="Fraction of reads to keep (default: 1/16)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    np.random.seed(args.seed)

    paths = sorted(glob.glob(args.hr_glob))
    if not paths:
        print(f"[error] glob matched 0 files: {args.hr_glob}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print(f"[info] {len(paths)} HR tiles, frac={args.frac}, seed={args.seed}")

    for p in tqdm(paths, desc="thinning"):
        hr = np.load(p).astype(np.float32)
        lr = binomial_thin(hr, args.frac)
        out_path = os.path.join(args.out, os.path.basename(p))
        np.save(out_path, lr)

    print(f"[done] saved {len(paths)} LR tiles to {args.out}")

if __name__ == "__main__":
    main()