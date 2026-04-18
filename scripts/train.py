"""Train SR-VAE or HiCPlus baseline.

LR (1, H, H) -> HR (1, sH, sH), where s = scale_factor (default 2).
Loss = w_rec * L1 + w_ssim * (1 - SSIM) + w_grad * Sobel + (VAE only) beta * KL
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import make_loaders
from model import HiCPlus, SRVAE, build_model
from repro import set_global_seed, write_run_artifacts
from utils import center_crop_to_match, sobel_edge_loss, ssim_loss


def kl_beta(epoch: int, total: int, warmup: int, beta_start: float, beta_end: float) -> float:
    if epoch <= warmup:
        return beta_start
    t = min(1.0, (epoch - warmup) / max(1, total - warmup))
    return beta_start + (beta_end - beta_start) * t


def compute_loss(model, lr_px, hr_px, loss_cfg, training: bool):
    if isinstance(model, SRVAE):
        pred, mu, logvar = model(lr_px, sample=training)
    else:
        pred = model(lr_px)
        mu = logvar = None

    pred, hr_t = center_crop_to_match(pred, hr_px)

    L_rec  = F.l1_loss(pred, hr_t)
    L_ssim = ssim_loss(pred, hr_t, data_range=1.0)
    L_grad = sobel_edge_loss(pred, hr_t)

    loss = loss_cfg["rec_w"] * L_rec + loss_cfg["ssim_w"] * L_ssim + loss_cfg["grad_w"] * L_grad
    kl_val = torch.tensor(0.0, device=pred.device)
    if mu is not None:
        kl_val = SRVAE.kl_divergence(mu, logvar, free_bits_per_dim=loss_cfg.get("free_bits", 0.0))
        loss = loss + loss_cfg["beta"] * kl_val
    return loss, kl_val


def train_one_epoch(model, optimizer, loader, device, loss_cfg):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for lr_px, hr_px in pbar:
        lr_px, hr_px = lr_px.to(device), hr_px.to(device)
        loss, kl_val = compute_loss(model, lr_px, hr_px, loss_cfg, training=True)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = lr_px.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl_val.item():.2e}")
    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, device, loss_cfg):
    model.eval()
    total, n = 0.0, 0
    for lr_px, hr_px in tqdm(loader, desc="val", leave=False):
        lr_px, hr_px = lr_px.to(device), hr_px.to(device)
        loss, _ = compute_loss(model, lr_px, hr_px, loss_cfg, training=False)
        total += loss.item() * lr_px.size(0)
        n += lr_px.size(0)
    return total / max(1, n)


def train(cfg, model_name: str, resume_path: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  model={model_name}")

    vae_cfg = cfg.get("vae", {})
    loss_raw = cfg.get("loss", {})
    srvae_cfg = cfg.get("srvae", {})

    z_ch    = int(vae_cfg.get("z_ch", 32))
    base_ch = int(vae_cfg.get("base_ch", 32))
    scale   = int(srvae_cfg.get("scale", 2))
    epochs  = int(vae_cfg.get("epochs", 50))
    lr      = float(vae_cfg.get("lr", 2e-4))
    save_dir = vae_cfg.get("save_dir", "./runs/sr_vae")
    if model_name == "hicplus":
        save_dir = save_dir.rstrip("/\\") + "_hicplus"
    seed    = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))

    rec_w   = float(loss_raw.get("rec_w", 0.5))
    ssim_w  = float(loss_raw.get("ssim_w", 0.25))
    grad_w  = float(loss_raw.get("grad_w", 0.25))
    beta_start = float(loss_raw.get("beta_start", 0.0))
    beta_end   = float(loss_raw.get("beta_end", 1e-3))
    warmup     = int(loss_raw.get("kl_warmup_epochs", max(5, epochs // 5)))
    free_bits  = float(loss_raw.get("free_bits", 0.05))

    set_global_seed(seed, deterministic=deterministic)
    train_ld, val_ld, _, _ = make_loaders(cfg, verbose=True)
    assert train_ld and val_ld, "need both train and val data"

    model = build_model(model_name, z_ch=z_ch, base_ch=base_ch, scale_factor=scale).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {model_name} z_ch={z_ch} base_ch={base_ch} scale={scale} params={n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    os.makedirs(save_dir, exist_ok=True)
    write_run_artifacts(
        save_dir,
        script_name="scripts/train.py",
        args_dict={"model": model_name, "resume": resume_path or ""},
        cfg=cfg,
        extra={"seed": seed, "deterministic": deterministic, "model_params": n_params},
    )

    best_val, start_ep = float("inf"), 1
    if resume_path and os.path.isfile(resume_path):
        print(f"[resume] {resume_path}")
        ck = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ck["model"])  # strict=True
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        best_val = ck.get("best_val", best_val)
        start_ep = ck.get("epoch", 0) + 1

    best_path = os.path.join(save_dir, f"{model_name}_best.pt")
    last_path = os.path.join(save_dir, f"{model_name}_last.pt")

    for ep in range(start_ep, epochs + 1):
        beta = kl_beta(ep, epochs, warmup, beta_start, beta_end) if model_name == "srvae" else 0.0
        lcfg = dict(rec_w=rec_w, ssim_w=ssim_w, grad_w=grad_w, beta=beta, free_bits=free_bits)

        tl = train_one_epoch(model, opt, train_ld, device, lcfg)
        vl = validate(model, val_ld, device, lcfg)
        print(f"[ep{ep:03d}]  lr={opt.param_groups[0]['lr']:.2e}  beta={beta:.2e}  train={tl:.4f}  val={vl:.4f}")
        sched.step()

        ckpt = dict(
            model=model.state_dict(),
            opt=opt.state_dict(),
            epoch=ep,
            best_val=min(best_val, vl),
            z_ch=z_ch,
            base_ch=base_ch,
            scale=scale,
            model_name=model_name,
            cfg=cfg,
        )
        torch.save(ckpt, last_path)
        if vl < best_val:
            best_val = vl
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)
            print(f"  [save] new best -> {best_path} ({best_val:.4f})")


def _coerce(val: str):
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def _apply_override(cfg: dict, key: str, val):
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=["srvae", "hicplus"], default="srvae")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=None, help="Override cfg.seed")
    ap.add_argument("--save-dir", default="", help="Override cfg.vae.save_dir")
    ap.add_argument("--set", action="append", default=[], metavar="key=value",
                    help="Override any dotted config key, e.g. loss.beta_end=0 loss.ssim_w=0")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.save_dir:
        cfg.setdefault("vae", {})["save_dir"] = args.save_dir
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        _apply_override(cfg, k.strip(), _coerce(v.strip()))

    train(cfg, model_name=args.model, resume_path=args.resume or None)


if __name__ == "__main__":
    main()
