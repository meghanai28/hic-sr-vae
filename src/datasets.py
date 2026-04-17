import glob
import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import log1p_normalize


def parse_tile_name(path: str) -> tuple[str, int, int]:
    """Tile filename format: {chrom}_{i}_{j}.npy where (i, j) are HR coords."""
    base = os.path.splitext(os.path.basename(path))[0]
    chrom, i, j = base.rsplit("_", 2)
    return chrom, int(i), int(j)


def _glob_sorted(pattern: Optional[str]) -> list[str]:
    if not pattern:
        return []
    return sorted(glob.glob(pattern, recursive=True))


def _limit_pairs(lr: list[str], hr: list[str], limit: Optional[int], seed: int) -> tuple[list[str], list[str]]:
    if limit is None or limit <= 0 or len(lr) <= limit:
        return lr, hr
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(lr), size=limit, replace=False))
    return [lr[i] for i in idx], [hr[i] for i in idx]


def load_chrom_stats(stats_path: str) -> dict[str, float]:
    """Load per-chromosome log1p-max stats produced by make_tiles.py."""
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(
            f"Missing per-chromosome stats file: {stats_path}. "
            "Re-run scripts/make_tiles.py to regenerate it."
        )
    with open(stats_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


class PairedHiC(Dataset):
    """LR/HR tile pairs at different spatial resolutions.

    LR = avg-pooled, binomial-thinned HR (size HR/scale).
    Both are normalized as log1p(raw) / chrom_log1p_max  -> [0, 1].
    The same chromosome-level scale is used for LR and HR so they're directly
    comparable in the model's input/output space.
    """

    def __init__(
        self,
        lr_paths: list[str],
        hr_paths: list[str],
        chrom_stats: dict[str, float],
        augment: bool = False,
    ):
        assert len(lr_paths) == len(hr_paths) > 0, \
            f"LR/HR count mismatch: {len(lr_paths)} vs {len(hr_paths)}"
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.stats = chrom_stats
        self.augment = augment

    def __len__(self) -> int:
        return len(self.lr_paths)

    def _scale_for(self, hr_path: str) -> float:
        chrom, _, _ = parse_tile_name(hr_path)
        if chrom not in self.stats:
            raise KeyError(f"No stats entry for chrom={chrom}; regenerate tiles.")
        return self.stats[chrom]

    def __getitem__(self, idx: int):
        lr = np.load(self.lr_paths[idx]).astype(np.float32)
        hr = np.load(self.hr_paths[idx]).astype(np.float32)
        scale = self._scale_for(self.hr_paths[idx])

        lr_t = log1p_normalize(torch.from_numpy(lr).unsqueeze(0), scale)
        hr_t = log1p_normalize(torch.from_numpy(hr).unsqueeze(0), scale)

        if self.augment and torch.rand(1).item() > 0.5:
            lr_t = lr_t.flip(-1).flip(-2)
            hr_t = hr_t.flip(-1).flip(-2)

        # Hi-C contact maps are symmetric M = M.T -> transpose is a valid aug.
        if self.augment and torch.rand(1).item() > 0.5:
            lr_t = lr_t.transpose(-1, -2)
            hr_t = hr_t.transpose(-1, -2)

        return lr_t, hr_t


def make_loaders(cfg: dict, verbose: bool = False):
    data_cfg = cfg.get("data", {})
    bs = int(cfg.get("vae", {}).get("batch_size", 4))
    nw = int(cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 42))

    stats_path = data_cfg.get("stats", "tiles/hr/stats.json")
    chrom_stats = load_chrom_stats(stats_path)

    tr_lr = _glob_sorted(data_cfg.get("train_lr"))
    tr_hr = _glob_sorted(data_cfg.get("train_hr"))
    va_lr = _glob_sorted(data_cfg.get("val_lr"))
    va_hr = _glob_sorted(data_cfg.get("val_hr"))
    te_lr = _glob_sorted(data_cfg.get("test_lr"))
    te_hr = _glob_sorted(data_cfg.get("test_hr"))

    tr_lr, tr_hr = _limit_pairs(tr_lr, tr_hr, data_cfg.get("train_limit"), seed=seed)
    va_lr, va_hr = _limit_pairs(va_lr, va_hr, data_cfg.get("val_limit"),   seed=seed + 1)
    te_lr, te_hr = _limit_pairs(te_lr, te_hr, data_cfg.get("test_limit"),  seed=seed + 2)

    dl_kwargs = dict(
        batch_size=bs,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=bool(nw > 0),
    )

    train_ld = None
    if tr_lr and tr_hr:
        assert len(tr_lr) == len(tr_hr), \
            f"Train LR/HR count mismatch: {len(tr_lr)} vs {len(tr_hr)}"
        ds = PairedHiC(tr_lr, tr_hr, chrom_stats=chrom_stats, augment=True)
        train_ld = DataLoader(ds, shuffle=True, drop_last=True, **dl_kwargs)

    val_ld = None
    if va_lr and va_hr:
        ds = PairedHiC(va_lr, va_hr, chrom_stats=chrom_stats, augment=False)
        val_ld = DataLoader(ds, shuffle=False, drop_last=False, **dl_kwargs)

    test_ld = None
    if te_lr and te_hr:
        ds = PairedHiC(te_lr, te_hr, chrom_stats=chrom_stats, augment=False)
        test_ld = DataLoader(ds, shuffle=False, drop_last=False, **dl_kwargs)

    if verbose:
        print(f"[data] bs={bs} workers={nw} seed={seed}  stats={stats_path}")
        print(f"  train: {len(tr_lr)} pairs")
        print(f"  val:   {len(va_lr)} pairs")
        print(f"  test:  {len(te_lr)} pairs")

    return train_ld, val_ld, test_ld, chrom_stats
