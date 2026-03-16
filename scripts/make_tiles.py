import os 
import argparse
import numpy as np

try:
    import cooler
except ImportError:
    raise ImportError("Install cooler: pip install cooler")

from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser(description="Extract HR tiles from .mcool")
    ap.add_argument("--mcool", required=True, help="Path to .mcool file")
    ap.add_argument("--res", type=int, default=10000, help="Resolution in bp")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--patch", type=int, default=256, help="Tile size in bins")
    ap.add_argument("--stride", type=int, default=128, help="Step between tiles")
    ap.add_argument("--max-diag-bins", type=int, default=200, help="Max distance from diagonal in bins")
    args = ap.parse_args()

    c = cooler.Cooler(f"{args.mcool}::/resolutions/{args.res}")
    chroms = list(c.chromnames)
    has_chr_prefix = chroms[0].startswith("chr")

    def to_cooler_name(name: str) -> str:
        if has_chr_prefix:
            return name
        return name.replace("chr", "")
    
    train_chroms = [f"chr{i}" for i in range(1, 17)]
    val_chroms = [f"chr{i}" for i in [17, 18]]
    test_chroms = [f"chr{i}" for i in [19, 20, 21, 22]]    

    splits = {
        "train": [to_cooler_name(n) for n in train_chroms if to_cooler_name(n) in chroms],
        "val":   [to_cooler_name(n) for n in val_chroms   if to_cooler_name(n) in chroms],
        "test":  [to_cooler_name(n) for n in test_chroms  if to_cooler_name(n) in chroms],
    }

    mat = c.matrix(balance=False)

    # square patches from each chromosome
    total_tiles = 0
    for split, chrom_list in splits.items():
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)

        for chrom in chrom_list:
            bins_df = c.bins().fetch(chrom)
            starts = bins_df["start"].to_numpy()
            ends = bins_df["end"].to_numpy()
            n_bins = len(starts)

            pbar = tqdm(
                range(0, n_bins - args.patch + 1, args.stride),
                desc=f"{split}/{chrom}",
                leave=False,
            )

            for i in pbar:
                j_lo = max(i, 0)  
                j_hi = min(n_bins - args.patch, i + args.max_diag_bins)

                for j in range(j_lo, j_hi + 1, args.stride):
                    si = int(starts[i])
                    ei = int(ends[i + args.patch - 1])
                    sj = int(starts[j])
                    ej = int(ends[j + args.patch - 1])

                    chrom_len = int(c.chromsizes[chrom])
                    if not (0 <= si < ei <= chrom_len and 0 <= sj < ej <= chrom_len):
                        continue

                    M = np.asarray(mat.fetch((chrom, si, ei), (chrom, sj, ej)),
                                   dtype=np.float32)
                    M = 0.5 * (M + M.T)

                    fname = f"{chrom}_{i}_{j}.npy"
                    np.save(os.path.join(out_dir, fname), M)
                    total_tiles += 1
            
    print(f"[done] saved {total_tiles} tiles to {args.out}")

if __name__ == "__main__":
    main()