# Hi-C Super-Resolution VAE

Real 2x super-resolution for Hi-C contact maps using a residual VAE, with an
apples-to-apples HiCPlus learned baseline and a tile-mosaic reconstruction
pipeline for held-out chromosomes.

## Pipeline

- LR tile = `binomial_thin(HR, frac=1/16) -> 2x avg-pool` (so LR is half the spatial size of HR)
- Model: residual VAE that maps LR (HxH) -> HR (2H x 2H) via `bicubic(LR) + res_gain * decoder(z)`
- Normalization: `log1p(raw) / log1p(max_raw_chrom)`, with a single per-chromosome divisor used for *both* LR and HR (no LR/HR scale mismatch)
- Loss: L1 + (1 - SSIM) + Sobel + `beta * KL` (KL is summed over latent dims with free-bits to prevent posterior collapse)
- Baselines: `LR` (bicubic-up), `Bicubic`, `Gaussian`, `HiCPlus` (learned), `SR-VAE` (ours)
- Metrics: MSE, SSIM, GenomeDISCO, HiC-Spector

## Splits

- `train`: chr 1-16
- `val`:   chr 17-18
- `test`:  chr 19-22

## Setup

```bash
py -m pip install torch numpy matplotlib pyyaml tqdm cooler scipy
```

## End-to-end commands

> **Re-run from scratch.** The tile format changed (no more `_zoom` field, LR
> tiles are now half-resolution). Delete `tiles/`, `runs/`, `tiles_extended/`,
> `tiles_full_dense/` before starting.

```bash
rm -rf tiles tiles_extended tiles_full_dense runs
```

### 1. Extract HR tiles + per-chromosome normalization stats

```bash
py scripts/make_tiles.py --mcool data/GM12878.mcool --res 10000 --out tiles/hr --patch 256 --stride 64 --offset-max 256
```

Writes:
- `tiles/hr/{train,val,test}/{chrom}_{i}_{j}.npy` (raw counts, 256x256)
- `tiles/hr/stats.json` (`{chrom: log1p(max_raw)}`)

### 2. Generate LR tiles (half-resolution)

```bash
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/train/*.npy" --out tiles/lr/train --frac 0.0625 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/val/*.npy"   --out tiles/lr/val   --frac 0.0625 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy"  --out tiles/lr/test  --frac 0.0625 --scale 2 --seed 42
```

LR tiles are 128x128, named identically to their HR counterparts.

### 3. Train SR-VAE

Fast dev:
```bash
py scripts/train.py --config configs/paper_fast.yaml --model srvae
```

Paper run:
```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae
```

Resume:
```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae --resume runs/paper_full/srvae_last.pt
```

### 4. Train HiCPlus baseline (same data + same losses)

```bash
py scripts/train.py --config configs/paper_full.yaml --model hicplus
```

Saved to `runs/paper_full_hicplus/hicplus_best.pt`.

### 5. Evaluate (held-out tiles)

```bash
py scripts/evaluate.py ^
  --config configs/paper_full.yaml ^
  --ckpt runs/paper_full/srvae_best.pt ^
  --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt ^
  --outdir runs/paper_full/eval
```

Add `--no-disco` to skip GenomeDISCO/HiC-Spector for a fast sanity-check.
Add `--max-samples 64` for a quick limited eval.

### 6. Reconstruct held-out chromosome (tile mosaic)

```bash
py scripts/reconstruct_chromosome.py ^
  --config configs/paper_full.yaml ^
  --ckpt runs/paper_full/srvae_best.pt ^
  --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt ^
  --split test --chrom 19 ^
  --outdir runs/paper_full/reconstruction ^
  --save-npy
```

### 7. Benchmark inference

```bash
py scripts/benchmark.py ^
  --config configs/paper_full.yaml ^
  --ckpt runs/paper_full/srvae_best.pt ^
  --outdir runs/paper_full/benchmark ^
  --warmup 8 --max-batches 64
```

## Outputs

Every script writes `run_manifest.json` (CLI args, env, git state) and
`resolved_config.yaml` to its output dir.

- Training: `srvae_best.pt`, `srvae_last.pt`
- Evaluate: `metrics.csv`, `panel_*.png`
- Reconstruct: `*_mosaic.png`, `*_support.png`, `*_metrics.csv`, optional `*.npy`
- Benchmark: `benchmark.json`

## Notes for the paper writeup

- This is real 2x SR (LR 128 -> HR 256), not denoising at the same resolution.
- KL is properly weighted (sum over latent dims, free-bits per dim) so the
  model behaves as an actual VAE; ablate by setting `loss.beta_end: 0` and
  `free_bits: 0` to recover a deterministic AE for comparison.
- All four learned/non-learned baselines are evaluated on the same held-out
  tiles with the same per-chromosome normalization, so the comparison is fair.
- For paper variance bars, run training with `--config configs/paper_full.yaml`
  three times under `seed: 42`, `seed: 43`, `seed: 44` and average eval CSVs.
