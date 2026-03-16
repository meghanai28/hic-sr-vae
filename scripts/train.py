import os
import sys
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import make_loaders
from model import SRVAE
from utils import (ssim_loss, sobel_edge_loss, weighted_l1_loss, distance_weight_map, center_crop_to_match)


def kl_beta(epoch, total_epochs, warmup, beta_start, beta_end):
    # linear ramp
    if epoch <= warmup:
        return 0.0
    t = min(1.0, max(0.0, (epoch - warmup) / max(1, total_epochs - warmup)))
    return beta_start + (beta_end - beta_start) * t

def train_one_epoch(model, optimizer, loader, device, loss_cfg, dw_cache):
    model.train()
    rec_w, ssim_w, grad_w = loss_cfg["rec_w"], loss_cfg["ssim_w"], loss_cfg["grad_w"]
    beta, dist_alpha = loss_cfg["beta"], loss_cfg.get("dist_alpha", 1.0)

    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)

    for lr_px, hr_px in pbar:
        lr_px, hr_px = lr_px.to(device), hr_px.to(device)
        pred, mu, logvar = model(lr_px, sample=False)
        pred, hr_px = center_crop_to_match(pred, hr_px)

        H, W = pred.shape[-2:]
        if (H, W) not in dw_cache:
            dw_cache[(H, W)] = distance_weight_map(H, W, alpha=dist_alpha, device=device)

        L_rec = weighted_l1_loss(pred, hr_px, dw_cache[(H, W)])
        L_ssim = ssim_loss(pred, hr_px)
        L_grad = sobel_edge_loss(pred, hr_px)
        L_kl = SRVAE.kl_divergence(mu, logvar) * beta
        loss = rec_w * L_rec + ssim_w * L_ssim + grad_w * L_grad + L_kl

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
def validate(model, loader, device, loss_cfg, dw_cache):
    model.eval()
    rec_w, ssim_w, grad_w = loss_cfg["rec_w"], loss_cfg["ssim_w"], loss_cfg["grad_w"]
    beta, dist_alpha = loss_cfg["beta"], loss_cfg.get("dist_alpha", 1.0)

    total, n = 0.0, 0
    for lr_px, hr_px in tqdm(loader, desc="val", leave=False):
        lr_px, hr_px = lr_px.to(device), hr_px.to(device)
        pred, mu, logvar = model(lr_px, sample=False)
        pred, hr_px = center_crop_to_match(pred, hr_px)

        H, W = pred.shape[-2:]
        if (H, W) not in dw_cache:
            dw_cache[(H, W)] = distance_weight_map(H, W, alpha=dist_alpha, device=device)

        L_rec = weighted_l1_loss(pred, hr_px, dw_cache[(H, W)])
        L_ssim = ssim_loss(pred, hr_px)
        L_grad = sobel_edge_loss(pred, hr_px)
        L_kl = SRVAE.kl_divergence(mu, logvar) * beta
        loss = rec_w * L_rec + ssim_w * L_ssim + grad_w * L_grad + L_kl

        total += loss.item() * lr_px.size(0)
        n += lr_px.size(0)

    return total / max(1, n)

def train(cfg, resume_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    vae_cfg = cfg.get("vae", {})
    loss_raw = cfg.get("loss", {})
    srvae_cfg = cfg.get("srvae", {})

    z_ch = int(vae_cfg.get("z_ch", 64))
    base_ch = int(vae_cfg.get("base_ch", 48))
    scale = int(srvae_cfg.get("scale", 1))
    epochs = int(vae_cfg.get("epochs", 100))
    lr = float(vae_cfg.get("lr", 2e-4))
    save_dir = vae_cfg.get("save_dir", "./runs/sr_vae")

    rec_w = float(loss_raw.get("rec_w", 0.5))
    ssim_w = float(loss_raw.get("ssim_w", 0.25))
    grad_w = float(loss_raw.get("grad_w", 0.25))
    beta_start = float(loss_raw.get("beta_start", 0.0))
    beta_end = float(loss_raw.get("beta_end", 1e-6))
    warmup = int(loss_raw.get("kl_warmup_epochs", max(10, epochs // 2)))
    dist_alpha = float(loss_raw.get("dist_alpha", 1.0))

    train_ld, val_ld, _ = make_loaders(cfg, verbose=True)
    assert train_ld and val_ld, "Need both train and val data"

    model = SRVAE(z_ch=z_ch, scale_factor=scale, base_ch=base_ch).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] z_ch={z_ch} scale={scale} base_ch={base_ch} params={params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
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
    dw_cache = {}

    for ep in range(start_ep, epochs + 1):
        beta = kl_beta(ep, epochs, warmup, beta_start, beta_end)
        lcfg = dict(rec_w=rec_w, ssim_w=ssim_w, grad_w=grad_w,
                     beta=beta, dist_alpha=dist_alpha)

        tl = train_one_epoch(model, opt, train_ld, device, lcfg, dw_cache)
        print(f"[train] ep{ep:03d}  beta={beta:.2e}  loss={tl:.4f}")

        vl = validate(model, val_ld, device, lcfg, dw_cache)
        print(f"[val]   ep{ep:03d}  loss={vl:.4f}")

        ckpt = dict(model=model.state_dict(), opt=opt.state_dict(),
                     epoch=ep, best_val=min(best_val, vl),
                     z_ch=z_ch, scale=scale, base_ch=base_ch, cfg=cfg)
        torch.save(ckpt, last_path)

        if vl < best_val:
            best_val = vl
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)
            print(f"[save] new best → {best_path} ({best_val:.4f})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg, resume_path=args.resume or None)


if __name__ == "__main__":
    main()