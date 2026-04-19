---
name: hic-sr-vae-reproduce
description: Reproduce the residual-VAE Hi-C 2x super-resolution paper end-to-end — training, held-out evaluation, chromosome reconstruction, depth-robustness sweep, cross-cell-line (K562) transfer, and biological validation (insulation / TAD / chromatin loops).
allowed-tools: Bash(py *), Bash(python *), Bash(git *)
---

# Reproducing the SR-VAE Hi-C paper

This skill reproduces all numbers and figures in *"A Residual Variational
Autoencoder for 2x Super-Resolution of Hi-C Contact Maps"*
(Indukuri et al., clawRxiv).

## Prerequisites

- Python 3.12 with a working CUDA install (tested on Windows 11 + RTX 4060
  Laptop, CUDA 12.1, PyTorch 2.5.1). CPU works but is ~50x slower.
- ~30 GB of free disk for tiles + checkpoints + mosaics.
- 4DN `.mcool` files downloaded to `data/`:
  - GM12878 — accession `4DNFIZL8OZE1`
    (<https://data.4dnucleome.org/files-processed/4DNFIZL8OZE1/>), saved as
    `data/GM12878.mcool`.
  - K562 — accession `4DNFIOHY9ZX7`
    (<https://data.4dnucleome.org/files-processed/4DNFIOHY9ZX7/>), saved as
    `data/4DNFIOHY9ZX7.mcool`.

## Setup

```bash
git clone https://github.com/meghanai28/hic-sr-vae
cd hic-sr-vae
py -m pip install torch numpy matplotlib pyyaml tqdm cooler scipy
mkdir data && mv /path/to/GM12878.mcool /path/to/4DNFIOHY9ZX7.mcool data/
```

## Steps to reproduce

### 1. HR tile extraction

```bash
py scripts/make_tiles.py --mcool data/GM12878.mcool --res 10000 \
    --out tiles/hr --patch 256 --stride 64 --offset-max 256
```

Writes `tiles/hr/{train,val,test}/{chrom}_{i}_{j}.npy` (raw counts, 256×256)
and `tiles/hr/stats.json` (`{chrom: log1p(max_raw_count)}`, the shared
per-chrom divisor used at normalization).

### 2. LR tile simulation (1/16 depth)

```bash
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/train/*.npy" --out tiles/lr/train --frac 0.0625 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/val/*.npy"   --out tiles/lr/val   --frac 0.0625 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy"  --out tiles/lr/test  --frac 0.0625 --scale 2 --seed 42
```

### 3. Train SR-VAE and HiCPlus (seed 42, the paper's headline config)

```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae
py scripts/train.py --config configs/paper_full.yaml --model hicplus
```

SR-VAE writes to `runs/paper_full/`; HiCPlus is auto-suffixed to
`runs/paper_full_hicplus/`. Each run is deterministic under seed 42.
Expected runtime: ~2 h per model on RTX 4060 Laptop.

### 4. Tile-level held-out evaluation (Table 1 in the paper)

```bash
py scripts/evaluate.py --config configs/paper_full.yaml \
    --ckpt runs/paper_full/srvae_best.pt \
    --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt \
    --outdir runs/paper_full/eval
```

Produces `runs/paper_full/eval/metrics.csv` (per-sample MSE, SSIM, DISCO,
HiC-Spector for every method) and panel figures for the first 32 tiles.

### 5. Chromosome-scale reconstruction (Table 4)

```bash
for ch in 19 20 21 22; do
  py scripts/reconstruct_chromosome.py --config configs/paper_full.yaml \
      --ckpt runs/paper_full/srvae_best.pt \
      --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt \
      --split test --chrom $ch \
      --outdir runs/paper_full/reconstruction_chr$ch --save-npy
done
```

`--save-npy` is required for steps 7 and 8 below.

### 6. Depth-robustness sweep (Table 5, no retraining)

```bash
# Generate LR tiles at 1/8 and 1/32 depth
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy" --out tiles/lr_frac08/test --frac 0.125   --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy" --out tiles/lr_frac32/test --frac 0.03125 --scale 2 --seed 42
# Evaluate with dedicated configs that point at the new LR tiles
py scripts/evaluate.py --config configs/paper_full_frac08.yaml --ckpt runs/paper_full/srvae_best.pt --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt --outdir runs/paper_full/eval_frac08
py scripts/evaluate.py --config configs/paper_full_frac32.yaml --ckpt runs/paper_full/srvae_best.pt --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt --outdir runs/paper_full/eval_frac32
```

### 7. Biological validation I — insulation / TAD boundaries (Table 7)

```bash
for ch in 19 20 21 22; do
  py scripts/insulation_validation.py \
      --mosaic-dir runs/paper_full/reconstruction_chr$ch \
      --split test --chrom $ch \
      --outdir runs/paper_full/insulation_chr$ch --sweep-strength
done
```

Produces per-chromosome `*_insulation.csv` (Pearson IS correlation,
boundary P/R/F1) and `*_insulation_sweep.csv` (threshold-swept AUPRC).

### 8. Biological validation II — chromatin loops (Table 8)

```bash
py scripts/loop_validation.py \
    --mosaic-dir runs/paper_full/reconstruction_chr19 \
    --split test --chrom 19 \
    --outdir runs/paper_full/loops_chr19 --sweep
```

Self-contained HiCCUPS-style donut-enrichment peak caller; same code path
for every method.

### 9. Cross-cell-line zero-shot transfer (Tables 6 + 8 rows for K562)

```bash
# Tile extraction on K562 (test split only — no retraining)
py scripts/make_tiles.py --mcool data/4DNFIOHY9ZX7.mcool --res 10000 \
    --out tiles_k562/hr --patch 256 --stride 64 --offset-max 256 --splits test
py scripts/make_lr_tiles.py --hr-glob "tiles_k562/hr/test/*.npy" \
    --out tiles_k562/lr/test --frac 0.0625 --scale 2 --seed 42

# Evaluate GM12878-trained checkpoints on K562 tiles
py scripts/evaluate.py --config configs/paper_full_k562.yaml \
    --ckpt runs/paper_full/srvae_best.pt \
    --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt \
    --outdir runs/paper_full/eval_k562

# Reconstruct + validate biology on K562 chr19
py scripts/reconstruct_chromosome.py --config configs/paper_full_k562.yaml \
    --ckpt runs/paper_full/srvae_best.pt \
    --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt \
    --split test --chrom chr19 \
    --outdir runs/paper_full/reconstruction_k562_chr19 --save-npy
py scripts/insulation_validation.py \
    --mosaic-dir runs/paper_full/reconstruction_k562_chr19 \
    --split test --chrom 19 \
    --outdir runs/paper_full/insulation_k562_chr19 --sweep-strength
py scripts/loop_validation.py \
    --mosaic-dir runs/paper_full/reconstruction_k562_chr19 \
    --split test --chrom 19 \
    --outdir runs/paper_full/loops_k562_chr19 --sweep
```

### 10. Seed variance (Table 2) and loss ablations (Table 3)

```bash
# Extra seeds
py scripts/train.py --config configs/paper_full.yaml --model srvae   --seed 43 --save-dir runs/paper_full_seed43
py scripts/train.py --config configs/paper_full.yaml --model srvae   --seed 44 --save-dir runs/paper_full_seed44
py scripts/train.py --config configs/paper_full.yaml --model hicplus --seed 43 --save-dir runs/paper_full_seed43
py scripts/train.py --config configs/paper_full.yaml --model hicplus --seed 44 --save-dir runs/paper_full_seed44
# Evaluate and aggregate
for d in paper_full paper_full_seed43 paper_full_seed44; do
  py scripts/evaluate.py --config configs/paper_full.yaml \
      --ckpt runs/$d/srvae_best.pt \
      --hicplus-ckpt runs/${d}_hicplus/hicplus_best.pt \
      --outdir runs/$d/eval
done
py scripts/aggregate_seeds.py --csvs \
    runs/paper_full/eval/metrics.csv \
    runs/paper_full_seed43/eval/metrics.csv \
    runs/paper_full_seed44/eval/metrics.csv \
    --paired-csv runs/paper_full/eval/metrics.csv \
    --out runs/paper_full/eval/seed_summary.csv

# Loss ablations (single seed each)
py scripts/train.py --config configs/paper_full.yaml --model srvae --save-dir runs/paper_full_no_ssim  --set loss.ssim_w=0.0
py scripts/train.py --config configs/paper_full.yaml --model srvae --save-dir runs/paper_full_no_sobel --set loss.grad_w=0.0
py scripts/train.py --config configs/paper_full.yaml --model srvae --save-dir runs/paper_full_no_kl    --set loss.beta_end=0.0
for d in paper_full_no_ssim paper_full_no_sobel paper_full_no_kl; do
  py scripts/evaluate.py --config configs/paper_full.yaml \
      --ckpt runs/$d/srvae_best.pt \
      --outdir runs/$d/eval
done
```

## Expected outputs

After all 10 steps, `runs/paper_full/**/metrics.csv` contains every number in
every table of the paper. The headline check — SR-VAE beats HiCPlus by ~19%
MSE and ~13% SSIM on held-out chromosomes — is in
`runs/paper_full/eval/metrics.csv`.

Total end-to-end runtime on the target hardware (RTX 4060 Laptop): ~8 h,
dominated by the three retraining runs in step 10.

## Notes for agentic reproduction

- All scripts accept `--set key=value` to override any YAML field at the CLI;
  no config edits are required.
- Training is deterministic under a fixed seed
  (`torch.backends.cudnn.deterministic = True`,
  `use_deterministic_algorithms(True)`).
- K562 tile filenames carry a `chr` prefix (e.g. `chr19_0_0.npy`);
  GM12878 does not (`19_0_0.npy`). `scripts/reconstruct_chromosome.py`
  handles both.
- The reconstructed `.npy` mosaics are ~133 MB each and are
  intentionally excluded from the repo via `.gitignore`; regenerate them
  with step 5.
