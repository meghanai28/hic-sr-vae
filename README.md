# Hi-C Super-Resolution VAE

Real 2x super-resolution for Hi-C contact maps using a residual VAE, with an
apples-to-apples HiCPlus learned baseline, a tile-mosaic reconstruction pipeline
for held-out chromosomes, and biological validation via insulation-score /
TAD-boundary recall.

## Pipeline

- LR tile = `binomial_thin(HR, frac=1/16) -> 2x avg-pool` (LR is half the spatial size of HR)
- Model: residual VAE that maps LR (HxH) -> HR (2H x 2H) via `bicubic(LR) + res_gain * decoder(z)`
- Normalization: `log1p(raw) / log1p(max_raw_chrom)`, with a single per-chromosome divisor used for *both* LR and HR (no LR/HR scale mismatch)
- Loss: L1 + (1 - SSIM) + Sobel + `beta * KL` (KL summed over latent dims with free-bits)
- Baselines: `LR` (bicubic-up), `Bicubic`, `Gaussian`, `HiCPlus` (learned, reimplemented), `SR-VAE` (ours)
- Tile-level metrics: MSE, SSIM, GenomeDISCO, HiC-Spector
- Biological validation: insulation-score Pearson + TAD-boundary F1 vs HR

## Critical information for the paper

- **Data:** GM12878 Hi-C from a cooler file (`data/GM12878.mcool`) at **10 kb** resolution. Single cell line; cross-cell-line generalization is an explicit limitation.
- **Chromosome splits:** train chr1-16, val 17-18, test 19-22.
- **Tile geometry:** HR patch 256x256 (2.56 Mb), stride 64 (640 kb), `offset-max = 256` HR bins => tiles cover a **2.56 Mb band** around the main diagonal. The "full-chromosome" reconstructions in this repo are banded, not complete N x N matrices. This follows prior work (HiCPlus, HiCNN, DeepHiC, HiCSR, HiCARN).
- **LR realism:** `binomial_thin(frac=1/16)` simulates ~6% of original read depth, then 2x avg-pool gives spatially-downsampled LR.
- **Both LR and HR** are divided by the *same* per-chromosome `log1p(max)` constant (see `tiles/hr/stats.json`), so the SR task is a pure resolution/depth problem rather than a scale-matching problem.
- **Random seeds:** seed 42 is the default; seed-variance runs (42/43/44) are described below.
- **HiCPlus baseline:** reimplemented from Zhang et al. 2018 (`src/model.py:142`); trained on the same tiles with the same loss as SR-VAE for a fair comparison.

## Setup

```bash
py -m pip install torch numpy matplotlib pyyaml tqdm cooler scipy
```

## End-to-end commands

> **Re-run from scratch.** Delete `tiles/` and `runs/` before starting if the
> tile format changes.
>
> Commands below use the cross-shell form (no `^` / `` ` `` continuations); each
> command is one line and works in PowerShell, cmd, and bash.

### 1. Extract HR tiles + per-chromosome normalization stats

```bash
py scripts/make_tiles.py --mcool data/GM12878.mcool --res 10000 --out tiles/hr --patch 256 --stride 64 --offset-max 256
```

Writes `tiles/hr/{train,val,test}/{chrom}_{i}_{j}.npy` (raw counts, 256x256) and
`tiles/hr/stats.json` (`{chrom: log1p(max_raw)}`).

### 2. Generate LR tiles (half-resolution, 1/16 depth)

```bash
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/train/*.npy" --out tiles/lr/train --frac 0.0625 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/val/*.npy"   --out tiles/lr/val   --frac 0.0625 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy"  --out tiles/lr/test  --frac 0.0625 --scale 2 --seed 42
```

LR tiles are 128x128, named identically to their HR counterparts.

### 3. Train SR-VAE + HiCPlus (paper config, seed 42)

```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae
py scripts/train.py --config configs/paper_full.yaml --model hicplus
```

SR-VAE goes to `runs/paper_full/`; HiCPlus is auto-suffixed to `runs/paper_full_hicplus/`.

### 4. Tile-level evaluation

```bash
py scripts/evaluate.py --config configs/paper_full.yaml --ckpt runs/paper_full/srvae_best.pt --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt --outdir runs/paper_full/eval
```

Add `--no-disco` for a fast sanity-check, `--max-samples 64` for a quick eval.

### 5. Reconstruct held-out chromosome (mosaic + .npy dumps)

```bash
py scripts/reconstruct_chromosome.py --config configs/paper_full.yaml --ckpt runs/paper_full/srvae_best.pt --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt --split test --chrom 19 --outdir runs/paper_full/reconstruction --save-npy
```

`--save-npy` is required for biological validation in step 7.

### 6. Inference benchmark

```bash
py scripts/benchmark.py --config configs/paper_full.yaml --ckpt runs/paper_full/srvae_best.pt --outdir runs/paper_full/benchmark --warmup 8 --max-batches 64
```

### 7. Biological validation (insulation score + TAD boundaries)

```bash
py scripts/insulation_validation.py --mosaic-dir runs/paper_full/reconstruction --split test --chrom 19 --outdir runs/paper_full/insulation --window 20 --delta-window 10 --min-strength 0.1 --tol 5
```

Outputs `{split}_chr{chrom}_insulation.csv` (Pearson of IS profile + boundary
precision/recall/F1 vs HR) and `{split}_chr{chrom}_insulation.png`.

## Experiments for the paper

### A. Seed variance (report mean +/- std, paired Wilcoxon)

Run 3 seeds for each model. `--set` and `--save-dir` are overrides added to
`train.py`; the yaml stays untouched.

```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae   --seed 43 --save-dir runs/paper_full_seed43
py scripts/train.py --config configs/paper_full.yaml --model srvae   --seed 44 --save-dir runs/paper_full_seed44
py scripts/train.py --config configs/paper_full.yaml --model hicplus --seed 43 --save-dir runs/paper_full_seed43
py scripts/train.py --config configs/paper_full.yaml --model hicplus --seed 44 --save-dir runs/paper_full_seed44
```

HiCPlus with `--save-dir X` writes to `X_hicplus/`.

Evaluate each seed:

```bash
py scripts/evaluate.py --config configs/paper_full.yaml --ckpt runs/paper_full/srvae_best.pt        --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt        --outdir runs/paper_full/eval
py scripts/evaluate.py --config configs/paper_full.yaml --ckpt runs/paper_full_seed43/srvae_best.pt --hicplus-ckpt runs/paper_full_seed43_hicplus/hicplus_best.pt --outdir runs/paper_full_seed43/eval
py scripts/evaluate.py --config configs/paper_full.yaml --ckpt runs/paper_full_seed44/srvae_best.pt --hicplus-ckpt runs/paper_full_seed44_hicplus/hicplus_best.pt --outdir runs/paper_full_seed44/eval
```

Aggregate:

```bash
py scripts/aggregate_seeds.py --csvs runs/paper_full/eval/metrics.csv runs/paper_full_seed43/eval/metrics.csv runs/paper_full_seed44/eval/metrics.csv --ref-method SR-VAE --out runs/paper_full/eval/seed_summary.csv
```

### B. Deterministic AE ablation (VAE -> AE)

Zero KL + no free bits + no warmup => deterministic autoencoder with the same
architecture. Demonstrates what the stochastic latent contributes.

```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae --save-dir runs/paper_full_ae --set loss.beta_end=0.0 --set loss.beta_start=0.0 --set loss.kl_warmup_epochs=0 --set loss.free_bits=0.0
py scripts/evaluate.py --config configs/paper_full.yaml --ckpt runs/paper_full_ae/srvae_best.pt --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt --outdir runs/paper_full_ae/eval
```

### C. Loss-component ablations

Each variant zeroes one loss term.

```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae --save-dir runs/paper_full_no_ssim  --set loss.ssim_w=0.0
py scripts/train.py --config configs/paper_full.yaml --model srvae --save-dir runs/paper_full_no_sobel --set loss.grad_w=0.0
py scripts/train.py --config configs/paper_full.yaml --model srvae --save-dir runs/paper_full_no_kl    --set loss.beta_end=0.0 --set loss.beta_start=0.0
```

Then evaluate each like in step 4.

### D. Downsampling-rate robustness (different band of read depths)

Generate LR tiles at additional sparsity levels and evaluate the same trained
SR-VAE model (no retraining) for a robustness curve.

```bash
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy" --out tiles/lr_frac08/test --frac 0.125 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy" --out tiles/lr_frac32/test --frac 0.03125 --scale 2 --seed 42
```

Evaluate with a config-override for the test LR glob:

```bash
py scripts/evaluate.py --config configs/paper_full.yaml --ckpt runs/paper_full/srvae_best.pt --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt --outdir runs/paper_full/eval_frac08 --no-disco
py scripts/evaluate.py --config configs/paper_full.yaml --ckpt runs/paper_full/srvae_best.pt --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt --outdir runs/paper_full/eval_frac32 --no-disco
```

(If `evaluate.py` hardcodes the test path, temporarily point the `data.test_lr`
key in a copy of `configs/paper_full.yaml` at `tiles/lr_frac08/test/*.npy` etc.
The three tile dirs let you run the full SR-VAE vs HiCPlus comparison at
fractions 1/8, 1/16, 1/32.)

### E. Biological validation across chromosomes

Repeat step 5 + step 7 for each test chromosome to get a per-chromosome
insulation-score F1 table.

```bash
py scripts/reconstruct_chromosome.py --config configs/paper_full.yaml --ckpt runs/paper_full/srvae_best.pt --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt --split test --chrom 20 --outdir runs/paper_full/reconstruction_chr20 --save-npy
py scripts/insulation_validation.py --mosaic-dir runs/paper_full/reconstruction_chr20 --split test --chrom 20 --outdir runs/paper_full/insulation_chr20
```

Repeat for chr21, chr22 to get the 4-test-chromosome table.

## Outputs

Every script writes `run_manifest.json` (CLI args, env, git state) and
`resolved_config.yaml` to its output dir.

- Training: `{model}_best.pt`, `{model}_last.pt`
- Evaluate: `metrics.csv`, `panel_*.png`
- Reconstruct: `*_mosaic.png`, `*_support.png`, `*_metrics.csv`, `--save-npy` dumps per method
- Benchmark: `benchmark.json`
- Insulation: `*_insulation.csv`, `*_insulation.png`
- Seed aggregation: `seed_summary.csv` + printed Wilcoxon table

## Notes for the paper writeup

- **Real 2x SR**, not same-resolution denoising (LR 128 -> HR 256 in pixel space,
  1.28 Mb -> 2.56 Mb in genomic coordinates at 10 kb).
- **KL is sum-reduced over latent dims** (~32k nats) so a small `beta_end=1e-4`
  still provides meaningful regularization; free-bits per dim prevents posterior
  collapse. Ablate to AE by setting `loss.beta_end=0` and `free_bits=0` (see B).
- **Baselines trained with the same loss / data / optimizer** so the comparison
  is fair. HiCPlus is reimplemented with a built-in bicubic upsample so it also
  performs real 2x SR (the original operated at same resolution).
- **Band-only reconstruction** (~2.56 Mb around diagonal) is standard for Hi-C
  SR and consistent with HiCPlus/HiCNN/DeepHiC/HiCSR.
- **Known limitations to state explicitly:** single cell line (GM12878), single
  resolution (10 kb), band-only reconstruction, simulated LR (binomial
  thinning) rather than an independently-sequenced shallow library.

## Citations to include

- Rao et al., *A 3D map of the human genome at kilobase resolution...*, Cell 2014 (GM12878 Hi-C).
- Zhang et al., *HiCPlus: enhancing Hi-C resolution using a deep CNN*, Nat. Commun. 2018 (baseline).
- Liu & Wang, *HiCNN*, Bioinformatics 2019.
- Hong et al., *DeepHiC*, PLoS Comput. Biol. 2020.
- Dimmick et al., *HiCSR*, Bioinformatics 2020.
- Hicks & Oluwadare, *HiCARN*, Bioinformatics 2022.
- Ursu et al., *GenomeDISCO*, Bioinformatics 2018.
- Yang et al., *HiC-Spector*, Bioinformatics 2017.
- Crane et al., *Condensin-driven remodelling of X-chromosome topology during dosage compensation*, Nature 2015 (insulation score).
- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014.
