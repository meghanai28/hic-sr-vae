import torch
import torch.nn.functional as F
import numpy as np

def _diagonal_expectations(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:

    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], "OE requires square matrix"
    N = mat.shape[0]
    E = np.zeros((N, N), dtype=np.float32)

    for d in range(N):
        # all entries at diagonal offset d
        diag_vals = np.diag(mat, k=d)
        mean_val = float(np.mean(diag_vals)) if diag_vals.size > 0 else 0.0
        mean_val = max(mean_val, eps)

        # fill both upper and lower diagonals
        idx = np.arange(N - d)
        E[idx, idx + d] = mean_val
        if d > 0:
            E[idx + d, idx] = mean_val

    return E

def normalize(x: torch.Tensor, mode: str | None) -> torch.Tensor:
    if mode is None or str(mode).lower() == "none":
        return torch.as_tensor(x, dtype=torch.float32)

    t = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
    mode_lower = str(mode).lower()

    if mode_lower == "oe":
        return _apply_oe(t)
    elif mode_lower == "zscore":
        mu = t.mean(dim=(-2, -1), keepdim=True)
        std = t.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (t - mu) / std
    elif mode_lower == "minmax":
        lo = t.amin(dim=(-2, -1), keepdim=True)
        hi = t.amax(dim=(-2, -1), keepdim=True)
        return (t - lo) / (hi - lo + 1e-6)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

def _apply_oe(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        # Single matrix [H, W]
        E = torch.from_numpy(_diagonal_expectations(t.cpu().numpy())).to(t.device)
        return torch.clamp(t / (E + 1e-8) - 1.0, -2.0, 2.0)

    elif t.dim() == 3:
        # [C, H, W]
        E = torch.from_numpy(_diagonal_expectations(t[0].cpu().numpy())).to(t.device)
        out = torch.clamp(t[0] / (E + 1e-8) - 1.0, -2.0, 2.0)
        return out.unsqueeze(0)

    elif t.dim() == 4:
        # [B, 1, H, W]
        B = t.size(0)
        out = torch.empty_like(t)
        for b in range(B):
            E = torch.from_numpy(
                _diagonal_expectations(t[b, 0].cpu().numpy())
            ).to(t.device)
            out[b, 0] = torch.clamp(t[b, 0] / (E + 1e-8) - 1.0, -2.0, 2.0)
        return out

    raise ValueError(f"OE norm expects 2D/3D/4D tensor, got {t.dim()}D")

def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=0)
    mu_t = F.avg_pool2d(target, window_size, stride=1, padding=0)

    sigma_pp = F.avg_pool2d(pred * pred, window_size, stride=1, padding=0) - mu_p * mu_p
    sigma_tt = F.avg_pool2d(target * target, window_size, stride=1, padding=0) - mu_t * mu_t
    sigma_pt = F.avg_pool2d(pred * target, window_size, stride=1, padding=0) - mu_p * mu_t

    numerator = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
    denominator = (mu_p**2 + mu_t**2 + C1) * (sigma_pp + sigma_tt + C2)
    ssim_map = numerator / (denominator + 1e-12)

    return 1.0 - ssim_map.mean()


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


def distance_weight_map(H: int, W: int, alpha: float = 1.0, device: torch.device | str = "cpu") -> torch.Tensor:
    i = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)
    j = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0)
    dist = (i - j).abs()
    max_dist = dist.max().clamp_min(1.0)
    weights = 1.0 + alpha * (dist / max_dist)
    return weights.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    return (weight_map * (pred - target).abs()).mean()

def center_crop_to_match(a: torch.Tensor, b: torch.Tensor):
    H = min(a.shape[-2], b.shape[-2])
    W = min(a.shape[-1], b.shape[-1])

    def _crop(x):
        dh = (x.shape[-2] - H) // 2
        dw = (x.shape[-1] - W) // 2
        return x[..., dh:dh + H, dw:dw + W]

    return _crop(a), _crop(b)


