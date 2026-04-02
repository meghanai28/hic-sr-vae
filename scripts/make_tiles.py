import os
import argparse
import numpy as np

try:
    import cooler
except ImportError:
    raise ImportError("Install cooler: pip install cooler")

from tqdm import tqdm


def _downsample(arr, factor):
    H, W = arr.shape
    H2, W2 = H // factor, W // factor
    return arr[:H2 * factor, :W2 * factor].reshape(H2, factor, W2, factor).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser(description="Extract multi-scale diagonal HR tiles from .mcool")
    ap.add_argument("--mcool", required=True)
    ap.add_argument("--res", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=256, help="Output tile size in pixels")
    ap.add_argument("--zooms", default="1,2,4", help="Comma-separated zoom factors")
    args = ap.parse_args()

    zooms = [int(z) for z in args.zooms.split(",")]

    c = cooler.Cooler(f"{args.mcool}::/resolutions/{args.res}")
    chroms = list(c.chromnames)
    has_chr_prefix = chroms[0].startswith("chr")

    def to_cooler_name(name):
        if has_chr_prefix:
            return name
        return name.replace("chr", "")

    train_chroms = [f"chr{i}" for i in range(1, 17)]
    val_chroms   = [f"chr{i}" for i in [17, 18]]
    test_chroms  = [f"chr{i}" for i in [19, 20, 21, 22]]

    splits = {
        "train": [to_cooler_name(n) for n in train_chroms if to_cooler_name(n) in chroms],
        "val":   [to_cooler_name(n) for n in val_chroms   if to_cooler_name(n) in chroms],
        "test":  [to_cooler_name(n) for n in test_chroms  if to_cooler_name(n) in chroms],
    }

    mat = c.matrix(balance=False)
    total_tiles = 0

    for split, chrom_list in splits.items():
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)

        for chrom in chrom_list:
            bins_df = c.bins().fetch(chrom)
            starts = bins_df["start"].to_numpy()
            ends   = bins_df["end"].to_numpy()
            n_bins = len(starts)
            chrom_len = int(c.chromsizes[chrom])

            for zoom in zooms:
                win    = args.patch * zoom
                stride = win

                pbar = tqdm(
                    range(0, n_bins - win + 1, stride),
                    desc=f"{split}/{chrom}/z{zoom}",
                    leave=False,
                )

                for i in pbar:
                    si = int(starts[i])
                    ei = int(ends[i + win - 1])

                    if not (0 <= si < ei <= chrom_len):
                        continue

                    M = np.asarray(
                        mat.fetch((chrom, si, ei), (chrom, si, ei)),
                        dtype=np.float32,
                    )
                    M = 0.5 * (M + M.T)

                    if zoom > 1:
                        M = _downsample(M, zoom).astype(np.float32)

                    fname = f"{chrom}_{i}_{zoom}.npy"
                    np.save(os.path.join(out_dir, fname), M)
                    total_tiles += 1

    print(f"[done] saved {total_tiles} tiles to {args.out}")


if __name__ == "__main__":
    main()
