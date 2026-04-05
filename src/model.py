import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.gn2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(x))
        h = self.conv1(h)
        h = self.act(self.gn2(h))
        h = self.conv2(h)
        return x + h

class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid),
            nn.GELU(),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w)
        return x * w.view(B, C, 1, 1)

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale ** 2), 3, 1, 1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ps(self.conv(x)))


class SRVAE(nn.Module):
    def __init__(self, z_ch: int = 64, scale_factor: int = 1, base_ch: int = 48, num_zooms: int = 6):
        super().__init__()
        self.z_ch = z_ch
        self.scale = scale_factor
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.enc0 = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1, 1), nn.GELU(), ResBlock(c1),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1), nn.GELU(), ResBlock(c2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1), nn.GELU(), ResBlock(c3), ResBlock(c3),
        )

        self.to_mu     = nn.Conv2d(c3, z_ch, 3, 1, 1)
        self.to_logvar = nn.Conv2d(c3, z_ch, 3, 1, 1)

        self.zoom_emb = nn.Embedding(num_zooms, z_ch)

        self.dec_in = nn.Sequential(
            nn.Conv2d(z_ch, c3, 3, 1, 1), nn.GELU(),
            ResBlock(c3), ResBlock(c3),
        )

        self.up1   = UpsampleBlock(c3, c3, scale=2)
        self.se1   = SEBlock(c2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, 3, 1, 1), nn.GELU(),
            ResBlock(c2), ResBlock(c2),
        )

        self.up0   = UpsampleBlock(c2, c2, scale=2)
        self.se0   = SEBlock(c1)
        self.fuse0 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, 3, 1, 1), nn.GELU(),
            ResBlock(c1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c1, 1, 3, 1, 1),
        )

        self.res_gain = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        h  = self.enc2(s1)
        return self.to_mu(h), self.to_logvar(h), s0, s1

    def decode(self, z: torch.Tensor, skip0: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z)
        h = self.up1(h)
        h = torch.cat([h, self.se1(skip1)], dim=1)
        h = self.fuse1(h)
        h = self.up0(h)
        h = torch.cat([h, self.se0(skip0)], dim=1)
        h = self.fuse0(h)
        return self.head(h)

    def forward(self, x: torch.Tensor, zoom_idx: torch.Tensor, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, skip0, skip1 = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu

        emb = self.zoom_emb(zoom_idx.to(x.device)).view(-1, self.z_ch, 1, 1)
        z = z + emb

        residual = self.decode(z, skip0, skip1)

        if self.scale == 1:
            skip = x
        else:
            out_h = int(round(x.shape[-2] * self.scale))
            out_w = int(round(x.shape[-1] * self.scale))
            skip = F.interpolate(x, size=(out_h, out_w), mode="bicubic", align_corners=False)

        if residual.shape[-2:] != skip.shape[-2:]:
            residual = F.interpolate(residual, size=skip.shape[-2:], mode="bicubic", align_corners=False)

        return skip + self.res_gain * residual, mu, logvar
