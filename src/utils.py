import numpy as np
import torch
import torch.nn.functional as F


def binomial_thin(counts: np.ndarray, frac: float) -> np.ndarray:
    arr = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.maximum(arr, 0).astype(np.int64)
    thinned = np.random.binomial(clean, frac)
    return thinned.astype(np.float32)


def avg_pool2d_np(mat: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mat.astype(np.float32, copy=False)
    h, w = mat.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return (
        mat[:h2, :w2]
        .reshape(h2 // factor, factor, w2 // factor, factor)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def log1p_normalize(x: torch.Tensor, scale: float) -> torch.Tensor:
    """log1p(x) / scale, clipped to [0, 1]. Same scale used for LR and HR."""
    x = x.float().clamp_min(0.0)
    out = torch.log1p(x) / max(scale, 1e-6)
    return out.clamp(0.0, 1.0)


def denormalize_log1p(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.expm1((x.float().clamp(0.0, 1.0)) * scale)


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=0)
    mu_t = F.avg_pool2d(target, window_size, stride=1, padding=0)

    sigma_pp = F.avg_pool2d(pred * pred, window_size, stride=1, padding=0) - mu_p * mu_p
    sigma_tt = F.avg_pool2d(target * target, window_size, stride=1, padding=0) - mu_t * mu_t
    sigma_pt = F.avg_pool2d(pred * target, window_size, stride=1, padding=0) - mu_p * mu_t

    num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_pp + sigma_tt + C2)
    return 1.0 - (num / (den + 1e-12)).mean()


def sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)

    pred_gx = F.conv2d(pred, kx, padding=1)
    pred_gy = F.conv2d(pred, ky, padding=1)
    tgt_gx = F.conv2d(target, kx, padding=1)
    tgt_gy = F.conv2d(target, ky, padding=1)

    return F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)


def center_crop_to_match(a: torch.Tensor, b: torch.Tensor):
    H = min(a.shape[-2], b.shape[-2])
    W = min(a.shape[-1], b.shape[-1])

    def _crop(x):
        dh = (x.shape[-2] - H) // 2
        dw = (x.shape[-1] - W) // 2
        return x[..., dh:dh + H, dw:dw + W]

    return _crop(a), _crop(b)
