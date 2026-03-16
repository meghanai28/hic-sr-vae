import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    # feature refinement
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
        return x + h  # residual connection

class SEBlock(nn.Module):
    # learn which channels matter
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        mid = max(ch // reduction, 4)  # at least 4 hidden units
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid),
            nn.GELU(),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        # each channel one summary number
        w = self.pool(x).view(B, C)
        # channel weights
        w = self.fc(w)
        # reweight channels
        return x * w.view(B, C, 1, 1)

class UpsampleBlock(nn.Module):
   # PuxelShuffle Upsample: https://juliusruseckas.github.io/ml/pixel_shuffle.html
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale ** 2), 3, 1, 1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ps(self.conv(x)))


class SRVAE(nn.Module):
    def __init__(self, z_ch: int = 64, scale_factor: int = 1, base_ch: int = 48):
        super().__init__()
        self.z_ch = z_ch
        self.scale = scale_factor
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4  

        # ENCODER
        # Level 0: full resolution - captures fine details
        self.enc0 = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1, 1), nn.GELU(), ResBlock(c1),
        )
        # Level 1: half resolution - captures mid-scale patterns 
        self.enc1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1), nn.GELU(), ResBlock(c2),
        )
        # Level 2: quarter resolution - captures large-scale structure like da compartments
        self.enc2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1), nn.GELU(), ResBlock(c3), ResBlock(c3),
        )

        # VAE
        self.to_mu = nn.Conv2d(c3, z_ch, 3, 1, 1)
        self.to_logvar = nn.Conv2d(c3, z_ch, 3, 1, 1)

        # DECODER
        self.dec_in = nn.Sequential(
            nn.Conv2d(z_ch, c3, 3, 1, 1), nn.GELU(),
            ResBlock(c3), ResBlock(c3),
        )

        # Upsample level 1: H/4 to H/2, then fuse with skip1
        self.up1 = UpsampleBlock(c3, c3, scale=2)
        self.se1 = SEBlock(c2) 
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, 3, 1, 1), nn.GELU(),
            ResBlock(c2), ResBlock(c2),
        )

        # Upsample level 0: H/2 to H, then fuse with skip0
        self.up0 = UpsampleBlock(c2, c2, scale=2)
        self.se0 = SEBlock(c1)
        self.fuse0 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, 3, 1, 1), nn.GELU(),
            ResBlock(c1),
        )

        # Final 1-channel output - residual detail
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c1, 1, 3, 1, 1),
        )

        # Learnable scalar to constrol residual strength.
        self.res_gain = nn.Parameter(torch.tensor(1.0))

    # VAE details
    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ENCODE & DECODE

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        s0 = self.enc0(x)      # [B, 48, H, W]
        s1 = self.enc1(s0)     # [B, 96, H/2, W/2]
        h = self.enc2(s1)      # [B, 192, H/4, W/4]

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar, s0, s1

    def decode(self, z: torch.Tensor, skip0: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:

        h = self.dec_in(z)               

        # Upsample to H/2
        h = self.up1(h)                  
        s1 = self.se1(skip1)              
        h = torch.cat([h, s1], dim=1)     
        h = self.fuse1(h)                 

        # Upsample to H
        h = self.up0(h)                  
        s0 = self.se0(skip0)              
        h = torch.cat([h, s0], dim=1)     
        h = self.fuse0(h)                 

        return self.head(h)              

    def forward(self, x: torch.Tensor, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 
        mu, logvar, skip0, skip1 = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        residual = self.decode(z, skip0, skip1)

        # Bicubic skip
        if self.scale == 1:
            skip = x
        else:
            out_h = int(round(x.shape[-2] * self.scale))
            out_w = int(round(x.shape[-1] * self.scale))
            skip = F.interpolate(x, size=(out_h, out_w), mode="bicubic", align_corners=False)

    
        if residual.shape[-2:] != skip.shape[-2:]:
            residual = F.interpolate(residual, size=skip.shape[-2:], mode="bicubic", align_corners=False)

        # Final output: bicubic baseline + learned residual
        y = skip + self.res_gain * residual
        return y, mu, logvar

