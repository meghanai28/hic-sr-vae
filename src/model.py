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

    def forward(self, x):
        h = self.act(self.gn1(x))
        h = self.conv1(h)
        h = self.act(self.gn2(h))
        h = self.conv2(h)
        return x + h


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale ** 2), 3, 1, 1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class SRVAE(nn.Module):
    """
    Hi-C super-resolution VAE.

    Encoder downsamples LR by 4x (two stride-2 convs) into a stochastic latent.
    Decoder upsamples back to LR resolution with skip fusion, then a final
    PixelShuffle stage upsamples to HR resolution. The final output is

        out = bicubic(LR, scale) + res_gain * decoder(z)

    so the network is forced to learn the residual on top of the bicubic
    baseline, which is also what gets reported as the bicubic comparison.
    """

    def __init__(self, z_ch: int = 32, scale_factor: int = 2, base_ch: int = 32):
        super().__init__()
        assert scale_factor in (1, 2, 4), "scale_factor must be 1, 2, or 4"
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

        self.dec_in = nn.Sequential(
            nn.Conv2d(z_ch, c3, 3, 1, 1), nn.GELU(),
            ResBlock(c3), ResBlock(c3),
        )
        self.up1 = UpsampleBlock(c3, c3, scale=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, 3, 1, 1), nn.GELU(),
            ResBlock(c2), ResBlock(c2),
        )
        self.up0 = UpsampleBlock(c2, c2, scale=2)
        self.fuse0 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, 3, 1, 1), nn.GELU(),
            ResBlock(c1),
        )

        if scale_factor > 1:
            self.up_sr = UpsampleBlock(c1, c1, scale=scale_factor)
        else:
            self.up_sr = nn.Identity()

        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c1, 1, 3, 1, 1),
        )

        self.res_gain = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits_per_dim: float = 0.0) -> torch.Tensor:
        """KL(q(z|x) || N(0, I)), summed over latent dims, averaged over batch.

        Free-bits per latent dim prevents posterior collapse: each dim is
        guaranteed at least `free_bits_per_dim` nats of KL.
        """
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        if free_bits_per_dim > 0.0:
            kl = torch.clamp(kl, min=free_bits_per_dim)
        return kl.flatten(1).sum(dim=1).mean()

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def encode(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        h = self.enc2(s1)
        return self.to_mu(h), self.to_logvar(h), s0, s1

    def decode(self, z, s0, s1):
        h = self.dec_in(z)
        h = self.up1(h)
        h = self.fuse1(torch.cat([h, s1], dim=1))
        h = self.up0(h)
        h = self.fuse0(torch.cat([h, s0], dim=1))
        h = self.up_sr(h)
        return self.head(h)

    def forward(self, x, sample: bool = True):
        mu, logvar, s0, s1 = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        residual = self.decode(z, s0, s1)

        if self.scale > 1:
            skip = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        else:
            skip = x

        if residual.shape[-2:] != skip.shape[-2:]:
            residual = F.interpolate(residual, size=skip.shape[-2:], mode="bicubic", align_corners=False)

        return skip + self.res_gain * residual, mu, logvar


class HiCPlus(nn.Module):
    """HiCPlus baseline (Zhang et al., Nat. Commun. 2018).

    Bicubic upsample -> 3-conv refine. Trained with the same data and L1 + SSIM
    losses as SRVAE so the comparison is apples-to-apples.
    """

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale = scale_factor
        self.conv1 = nn.Conv2d(1, 8, 9, padding=4)
        self.conv2 = nn.Conv2d(8, 8, 1)
        self.conv3 = nn.Conv2d(8, 1, 5, padding=2)

    def forward(self, x):
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return self.conv3(h)


def build_model(name: str, *, z_ch: int = 32, base_ch: int = 32, scale_factor: int = 2):
    name = name.lower()
    if name == "srvae":
        return SRVAE(z_ch=z_ch, scale_factor=scale_factor, base_ch=base_ch)
    if name == "hicplus":
        return HiCPlus(scale_factor=scale_factor)
    raise ValueError(f"Unknown model: {name}")
