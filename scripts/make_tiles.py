import os
import argparse
import numpy as np

try:
    import cooler
except ImportError:
    raise ImportError("Install cooler: pip install cooler")

from tqdm import tqdm


def _get_tile(mat, chrom, starts, ends, i, j, patch):
    si = int(starts[i])
    ei = int(ends[i + patch - 1])
    sj = int(starts[j])
    ej = int(ends[j + patch - 1])
    M = np.asarray(mat.fetch((chrom, si, ei), (chrom, sj, ej)), dtype=np.float32)
    if i == j:
        M = 0.5 * (M + M.T)
    return M


def _positions(N, patch, stride):
    i_vals = sorted(set(range(0, N - patch + 1, stride)) | {max(0, N - patch)})
    positions = []
    for i in i_vals:
        j_hi = N - patch + 1
        j_vals = sorted(set(range(i, j_hi, stride)) | {max(i, j_hi - 1)} if j_hi > i else set())
        for j in j_vals:
            positions.append((i, j))
    return positions


def extract_split(c, chrom_list, patch, stride, out_base, split):
    out_dir = os.path.join(out_base, split)
    os.makedirs(out_dir, exist_ok=True)
    mat = c.matrix(balance=False)
    count = 0

    for chrom in chrom_list:
        if chrom not in c.chromnames:
            continue
        bins_df = c.bins().fetch(chrom)
        starts = bins_df["start"].to_numpy()
        ends = bins_df["end"].to_numpy()
        n_bins = len(starts)

        if n_bins < patch:
            print(f"[skip] {chrom}: {n_bins} bins < patch {patch}")
            continue

        positions = _positions(n_bins, patch, stride)
        for i, j in tqdm(positions, desc=f"{split}/{chrom}", leave=False):
            M = _get_tile(mat, chrom, starts, ends, i, j, patch)
            np.save(os.path.join(out_dir, f"{chrom}_{i}_{j}.npy"), M)
            count += 1

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcool",  required=True)
    ap.add_argument("--res",    type=int, required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--patch",  type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()

    c = cooler.Cooler(f"{args.mcool}::/resolutions/{args.res}")
    all_chroms = list(c.chromnames)
    has_chr = all_chroms[0].startswith("chr")
    def fix(n): return n if has_chr else n.replace("chr", "")

    train_chroms = [fix(f"chr{i}") for i in range(1, 17)]
    val_chroms   = [fix(f"chr{i}") for i in [17, 18]]
    test_chroms  = [fix(f"chr{i}") for i in [19, 20, 21, 22]]

    splits = {
        "train": [ch for ch in train_chroms if ch in all_chroms],
        "val":   [ch for ch in val_chroms   if ch in all_chroms],
        "test":  [ch for ch in test_chroms  if ch in all_chroms],
    }

    total = 0
    for split, chrom_list in splits.items():
        n = extract_split(c, chrom_list, args.patch, args.stride, args.out, split)
        print(f"[{split}] {n} tiles")
        total += n

    print(f"[done] {total} tiles -> {args.out}")


if __name__ == "__main__":
    main()
