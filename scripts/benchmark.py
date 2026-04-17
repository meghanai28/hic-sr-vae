"""Benchmark inference latency, throughput, and peak memory.

Includes warmup batches (excluded from timing) so the first cuDNN dispatch
doesn't bias the mean / p95.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import make_loaders
from model import SRVAE, build_model
from repro import set_global_seed, write_run_artifacts


@torch.no_grad()
def benchmark(model, loader, device: str, max_batches: int, warmup: int) -> dict:
    model.eval()
    is_vae = isinstance(model, SRVAE)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup (not counted)
    seen = 0
    for lr_px, _ in loader:
        if seen >= warmup:
            break
        lr_px = lr_px.to(device)
        if is_vae:
            model(lr_px, sample=False)
        else:
            model(lr_px)
        seen += 1
    if device == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    total_samples = 0
    for batch_idx, (lr_px, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        lr_px = lr_px.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if is_vae:
            model(lr_px, sample=False)
        else:
            model(lr_px)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        total_samples += int(lr_px.size(0))

    arr = np.asarray(latencies_ms, dtype=np.float64)
    peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else None
    return {
        "warmup_batches": warmup,
        "batches_measured": int(arr.size),
        "samples_measured": total_samples,
        "latency_mean_ms": float(arr.mean()) if arr.size else None,
        "latency_std_ms": float(arr.std()) if arr.size else None,
        "latency_median_ms": float(np.median(arr)) if arr.size else None,
        "latency_p95_ms": float(np.percentile(arr, 95)) if arr.size else None,
        "throughput_samples_per_sec": float(1000.0 * total_samples / arr.sum()) if arr.size and arr.sum() > 0 else None,
        "peak_memory_mb": peak_mem_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-batches", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed, deterministic=bool(cfg.get("deterministic", True)))
    write_run_artifacts(args.outdir, script_name="scripts/benchmark.py",
                        args_dict=vars(args), cfg=cfg, extra={"seed": seed})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_ld, test_ld, _ = make_loaders(cfg, verbose=True)
    loader = test_ld or val_ld
    if loader is None:
        raise SystemExit("need val or test data")

    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck.get("model_name", "srvae"),
                        z_ch=ck.get("z_ch", 32), base_ch=ck.get("base_ch", 32),
                        scale_factor=ck.get("scale", 2)).to(device).eval()
    model.load_state_dict(ck["model"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results = benchmark(model, loader, device=device, max_batches=args.max_batches, warmup=args.warmup)
    results.update(
        {
            "device": device,
            "model": ck.get("model_name", "srvae"),
            "checkpoint_epoch": ck.get("epoch"),
            "model_params": int(n_params),
        }
    )

    out_path = os.path.join(args.outdir, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"[saved] {out_path}")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
