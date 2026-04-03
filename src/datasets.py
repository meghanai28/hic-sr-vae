import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import normalize


def _glob_sorted(pattern: str | None) -> list[str]:
    if not pattern:
        return []
    return sorted(glob.glob(pattern, recursive=True))


class PairedHiC(Dataset):
    def __init__(self, lr_paths: list[str], hr_paths: list[str], norm: str | None = "oe", augment: bool = False):
        assert len(lr_paths) == len(hr_paths) > 0, \
            f"LR/HR count mismatch: {len(lr_paths)} vs {len(hr_paths)}"
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.norm = norm
        self.augment = augment

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr = np.load(self.lr_paths[idx]).astype(np.float32)
        hr = np.load(self.hr_paths[idx]).astype(np.float32)

        lr_t = normalize(torch.from_numpy(lr).unsqueeze(0), self.norm)
        hr_t = normalize(torch.from_numpy(hr).unsqueeze(0), self.norm)

        if self.augment and torch.rand(1).item() > 0.5:
            lr_t = lr_t.flip(-1).flip(-2)
            hr_t = hr_t.flip(-1).flip(-2)

        return lr_t, hr_t


class BlindHiC(Dataset):
    def __init__(self, lr_paths: list[str], norm: str | None = "oe"):
        assert len(lr_paths) > 0, "No LR files found"
        self.paths = lr_paths
        self.norm = norm

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        lr = np.load(self.paths[idx]).astype(np.float32)
        return normalize(torch.from_numpy(lr).unsqueeze(0), self.norm)


def make_loaders(cfg: dict, verbose: bool = False):
    data_cfg = cfg.get("data", {})
    norm = data_cfg.get("norm", cfg.get("norm", "oe"))
    bs = int(cfg.get("vae", {}).get("batch_size", 4))
    nw = int(cfg.get("num_workers", 0))

    tr_lr = _glob_sorted(data_cfg.get("train_lr"))
    tr_hr = _glob_sorted(data_cfg.get("train_hr"))
    va_lr = _glob_sorted(data_cfg.get("val_lr"))
    va_hr = _glob_sorted(data_cfg.get("val_hr"))
    te_lr = _glob_sorted(data_cfg.get("test_lr"))
    te_hr = _glob_sorted(data_cfg.get("test_hr"))

    train_ld = None
    if tr_lr and tr_hr:
        assert len(tr_lr) == len(tr_hr), \
            f"Train LR/HR count mismatch: {len(tr_lr)} vs {len(tr_hr)}"
        ds = PairedHiC(tr_lr, tr_hr, norm=norm, augment=True)
        train_ld = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw, pin_memory=True)

    val_ld = None
    if va_lr and va_hr:
        ds = PairedHiC(va_lr, va_hr, norm=norm, augment=False)
        val_ld = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=True)

    test_ld = None
    if te_lr and te_hr:
        ds = PairedHiC(te_lr, te_hr, norm=norm, augment=False)
        test_ld = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=True)

    if verbose:
        print(f"[data] norm={norm} bs={bs}")
        print(f"  train: {len(tr_lr)} pairs")
        print(f"  val:   {len(va_lr)} pairs")
        print(f"  test:  {len(te_lr)} pairs")

    return train_ld, val_ld, test_ld
