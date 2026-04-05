import os
import argparse
import numpy as np

try:
    import cooler
except ImportError:
    raise ImportError("Install cooler: pip install cooler")

BANDS = [
    {"band": (0, 256),     "zoom": 1,  "stride": 64},
    {"band": (256, 512),   "zoom": 1,  "stride": 128},
    {"band": (512, 1024),  "zoom": 2,  "stride": 512},
    {"band": (1024, 2048), "zoom": 4,  "stride": 1024},
    {"band": (2048, 4096), "zoom": 8,  "stride": 2048},
    {"band": (4096, 8192), "zoom": 16, "stride": 4096},
]

ZOOM_TO_IDX = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}


def _downsample(mat, factor):
    h, w = mat.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return mat[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def extract_split(c, chrom_list, patch, out_base, split):
    out_dir = os.path.join(out_base, split)
    os.makedirs(out_dir, exist_ok=True)
    mat = c.matrix(balance=False)
    count = 0

    for chrom in chrom_list:
        if chrom not in c.chromnames:
            continue
        raw = mat.fetch(chrom).astype(np.float32)
        raw = 0.5 * (raw + raw.T)
        N = raw.shape[0]

        if N < patch:
            print(f"[skip] {chrom}: {N} bins < patch {patch}")
            continue

        for band_cfg in BANDS:
            blo, bhi = band_cfg["band"]
            zoom = band_cfg["zoom"]
            stride = band_cfg["stride"]
            win = patch * zoom

            if blo >= N:
                continue

            skipped = 0
            for i in range(0, N, stride):
                for dj in range(blo, min(bhi, N), stride):
                    j = i + dj
                    if j >= N:
                        break

                    ih = min(win, N - i)
                    jw = min(win, N - j)
                    tile = np.zeros((win, win), dtype=np.float32)
                    tile[:ih, :jw] = raw[i:i + ih, j:j + jw]

                    if zoom > 1:
                        tile = _downsample(tile, zoom)

                    nonzero_frac = np.count_nonzero(tile) / tile.size
                    if nonzero_frac < 0.01:
                        skipped += 1
                        continue

                    np.save(os.path.join(out_dir, f"{chrom}_{i}_{j}_{zoom}.npy"), tile)
                    count += 1

            if skipped:
                print(f"  [{chrom}/band{blo}-{bhi}/z{zoom}] skipped {skipped} near-empty")

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcool",  required=True)
    ap.add_argument("--res",    type=int, required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--patch",  type=int, default=256)
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
        n = extract_split(c, chrom_list, args.patch, args.out, split)
        print(f"[{split}] {n} tiles")
        total += n

    print(f"[done] {total} tiles -> {args.out}")


if __name__ == "__main__":
    main()
