"""Extract HR Hi-C tiles + per-chromosome log1p-max stats.

Tiles are stored as raw counts (no normalization). The accompanying
`stats.json` holds `log1p(max(chrom))` for each chromosome, which the
DataLoader uses to map both LR and HR into [0, 1] with a *single, shared*
per-chromosome scale.

Tile filename:  {chrom}_{i}_{j}.npy   where (i, j) are HR-bin coordinates.
Coverage:       diagonal band, stride S, off-diagonal offset up to OFFSET.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import cooler
except ImportError as exc:
    raise SystemExit("Install cooler: pip install cooler") from exc


TRAIN_CHROMS = list(range(1, 17))
VAL_CHROMS   = [17, 18]
TEST_CHROMS  = [19, 20, 21, 22]


def normalize_chrom_name(chrom: str, has_chr_prefix: bool) -> str:
    chrom = str(chrom)
    if has_chr_prefix:
        return chrom if chrom.startswith("chr") else f"chr{chrom}"
    return chrom.replace("chr", "")


def extract_split(c, chrom_list, patch, stride, offset_max, out_base, split, stats):
    out_dir = os.path.join(out_base, split)
    os.makedirs(out_dir, exist_ok=True)
    mat = c.matrix(balance=False)
    count = 0

    for chrom in chrom_list:
        if chrom not in c.chromnames:
            continue

        raw = mat.fetch(chrom).astype(np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.maximum(raw, 0.0)
        raw = 0.5 * (raw + raw.T)
        N = raw.shape[0]

        if N < patch:
            print(f"[skip] {chrom}: {N} bins < patch {patch}")
            continue

        chrom_max = float(raw.max())
        stats[chrom] = float(np.log1p(chrom_max))

        kept = 0
        skipped_empty = 0
        for i in range(0, N - patch + 1, stride):
            for dj in range(0, offset_max + 1, stride):
                j = i + dj
                if j + patch > N:
                    break
                tile = raw[i:i + patch, j:j + patch]
                if np.count_nonzero(tile) / tile.size < 0.01:
                    skipped_empty += 1
                    continue
                np.save(os.path.join(out_dir, f"{chrom}_{i}_{j}.npy"), tile.astype(np.float32))
                kept += 1
        count += kept
        print(f"  [{chrom}] kept={kept} skipped_empty={skipped_empty} log1p_max={stats[chrom]:.3f}")

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcool", required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--offset-max", type=int, default=256,
                    help="Max off-diagonal offset (in HR bins) for tile origin j relative to i.")
    ap.add_argument("--splits", nargs="*", choices=["train", "val", "test"], default=None)
    args = ap.parse_args()

    c = cooler.Cooler(f"{args.mcool}::/resolutions/{args.res}")
    has_chr = c.chromnames[0].startswith("chr")
    splits = {
        "train": [normalize_chrom_name(f"chr{i}", has_chr) for i in TRAIN_CHROMS],
        "val":   [normalize_chrom_name(f"chr{i}", has_chr) for i in VAL_CHROMS],
        "test":  [normalize_chrom_name(f"chr{i}", has_chr) for i in TEST_CHROMS],
    }
    splits = {k: [ch for ch in v if ch in c.chromnames] for k, v in splits.items()}
    if args.splits:
        splits = {k: splits[k] for k in args.splits}

    print(f"[cfg] patch={args.patch} stride={args.stride} offset_max={args.offset_max}")
    stats: dict[str, float] = {}
    total = 0
    for split, chrom_list in splits.items():
        n = extract_split(
            c, chrom_list, args.patch, args.stride, args.offset_max,
            out_base=args.out, split=split, stats=stats,
        )
        print(f"[{split}] {n} tiles")
        total += n

    stats_path = os.path.join(args.out, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"[done] {total} tiles -> {args.out}")
    print(f"[done] stats   -> {stats_path}")


if __name__ == "__main__":
    main()
