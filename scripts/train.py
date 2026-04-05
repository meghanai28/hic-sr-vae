import os
import sys
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import make_loaders
from model import SRVAE
from utils import ssim_loss, sobel_edge_loss, center_crop_to_match


def kl_beta(epoch, total_epochs, warmup, beta_start, beta_end):
    if epoch <= warmup:
        return 0.0
    t = min(1.0, max(0.0, (epoch - warmup) / max(1, total_epochs - warmup)))
    return beta_start + (beta_end - beta_start) * t


def train_one_epoch(model, optimizer, loader, device, loss_cfg):
    model.train()
    rec_w, ssim_w, grad_w = loss_cfg["rec_w"], loss_cfg["ssim_w"], loss_cfg["grad_w"]
    beta = loss_cfg["beta"]

    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)

    for lr_px, hr_px, zoom_idx in pbar:
        lr_px, hr_px = lr_px.to(device), hr_px.to(device)
        zoom_idx = zoom_idx.to(device)

        pred, mu, logvar = model(lr_px, zoom_idx, sample=True)
        pred, hr_px = center_crop_to_match(pred, hr_px)

        L_rec  = F.l1_loss(pred, hr_px)
        L_ssim = ssim_loss(pred, hr_px)
        L_grad = sobel_edge_loss(pred, hr_px)
        L_kl   = SRVAE.kl_divergence(mu, logvar) * beta
        loss   = rec_w * L_rec + ssim_w * L_ssim + grad_w * L_grad + L_kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = lr_px.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{SRVAE.kl_divergence(mu, logvar).item():.2e}")

    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, device, loss_cfg):
    model.eval()
    rec_w, ssim_w, grad_w = loss_cfg["rec_w"], loss_cfg["ssim_w"], loss_cfg["grad_w"]
    beta = loss_cfg["beta"]

    total, n = 0.0, 0
    for lr_px, hr_px, zoom_idx in tqdm(loader, desc="val", leave=False):
        lr_px, hr_px = lr_px.to(device), hr_px.to(device)
        zoom_idx = zoom_idx.to(device)

        pred, mu, logvar = model(lr_px, zoom_idx, sample=False)
        pred, hr_px = center_crop_to_match(pred, hr_px)

        L_rec  = F.l1_loss(pred, hr_px)
        L_ssim = ssim_loss(pred, hr_px)
        L_grad = sobel_edge_loss(pred, hr_px)
        L_kl   = SRVAE.kl_divergence(mu, logvar) * beta
        loss   = rec_w * L_rec + ssim_w * L_ssim + grad_w * L_grad + L_kl

        total += loss.item() * lr_px.size(0)
        n += lr_px.size(0)

    return total / max(1, n)


def train(cfg, resume_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    vae_cfg  = cfg.get("vae", {})
    loss_raw = cfg.get("loss", {})
    srvae_cfg = cfg.get("srvae", {})

    z_ch     = int(vae_cfg.get("z_ch", 64))
    base_ch  = int(vae_cfg.get("base_ch", 48))
    scale    = int(srvae_cfg.get("scale", 1))
    num_zooms = int(cfg.get("num_zooms", 6))
    epochs   = int(vae_cfg.get("epochs", 100))
    lr       = float(vae_cfg.get("lr", 2e-4))
    save_dir = vae_cfg.get("save_dir", "./runs/sr_vae")

    rec_w      = float(loss_raw.get("rec_w", 0.5))
    ssim_w     = float(loss_raw.get("ssim_w", 0.25))
    grad_w     = float(loss_raw.get("grad_w", 0.25))
    beta_start = float(loss_raw.get("beta_start", 0.0))
    beta_end   = float(loss_raw.get("beta_end", 1e-6))
    warmup     = int(loss_raw.get("kl_warmup_epochs", max(10, epochs // 2)))

    train_ld, val_ld, _ = make_loaders(cfg, verbose=True)
    assert train_ld and val_ld, "Need both train and val data"

    model = SRVAE(z_ch=z_ch, scale_factor=scale, base_ch=base_ch, num_zooms=num_zooms).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] z_ch={z_ch} scale={scale} base_ch={base_ch} num_zooms={num_zooms} params={params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    os.makedirs(save_dir, exist_ok=True)

    best_val, start_ep = float("inf"), 1
    if resume_path and os.path.isfile(resume_path):
        print(f"[resume] {resume_path}")
        ck = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        if "opt" in ck:
            try:
                opt.load_state_dict(ck["opt"])
            except Exception:
                pass
        best_val = ck.get("best_val", best_val)
        start_ep = ck.get("epoch", 0) + 1

    best_path = os.path.join(save_dir, "sr_vae_best.pt")
    last_path = os.path.join(save_dir, "sr_vae_last.pt")

    for ep in range(start_ep, epochs + 1):
        beta = kl_beta(ep, epochs, warmup, beta_start, beta_end)
        lcfg = dict(rec_w=rec_w, ssim_w=ssim_w, grad_w=grad_w, beta=beta)

        tl = train_one_epoch(model, opt, train_ld, device, lcfg)
        print(f"[train] ep{ep:03d}  lr={opt.param_groups[0]['lr']:.2e}  loss={tl:.4f}")

        vl = validate(model, val_ld, device, lcfg)
        print(f"[val]   ep{ep:03d}  loss={vl:.4f}")

        sched.step()

        ckpt = dict(model=model.state_dict(), opt=opt.state_dict(),
                    epoch=ep, best_val=min(best_val, vl),
                    z_ch=z_ch, scale=scale, base_ch=base_ch,
                    num_zooms=num_zooms, cfg=cfg)
        torch.save(ckpt, last_path)

        if vl < best_val:
            best_val = vl
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)
            print(f"[save] new best -> {best_path} ({best_val:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg, resume_path=args.resume or None)


if __name__ == "__main__":
    main()
