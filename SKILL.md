---
name: hic-sr-vae-reproduce
description: Reproduce the residual-VAE Hi-C 2x super-resolution paper. Offers a fast validation path (≈20 min, uses committed checkpoints to confirm the headline SR-VAE > HiCPlus claim on GM12878 held-out chromosomes) and a full end-to-end path (≈8 h, retrains everything and runs K562 transfer + biological validation).
allowed-tools: Bash(py *), Bash(python *), Bash(git *), Bash(curl *), Bash(mkdir *), Bash(mv *)
---

# Reproducing the SR-VAE Hi-C paper

This skill reproduces the numbers and figures in *"A Residual Variational
Autoencoder for 2x Super-Resolution of Hi-C Contact Maps"* (Indukuri et al.,
clawRxiv).

**Two paths are provided:**
- **Quick validation (≈20 min).** Steps A–D. Uses the committed checkpoints
  in `runs/paper_full/` and only the GM12878 `.mcool` (≈2 GB download).
  Validates the headline claim: SR-VAE beats HiCPlus by ≈19% MSE and ≈13%
  SSIM on held-out chromosomes 19–22.
- **Full reproduction (≈8 h on an RTX 4060 Laptop GPU).** Steps 1–10.
  Retrains both models, runs depth-robustness, cross-cell-line K562
  transfer, and both biological-validation tracks.

Claw should run the quick path by default. The full path is for reviewers
who want to re-derive every table.

## Prerequisites

- Python 3.12 with a working CUDA install for the full path (tested on
  Windows 11 + RTX 4060 Laptop, CUDA 12.1, PyTorch 2.5.1). The quick path
  runs on CPU in ≈20 min; full retraining on CPU is ~50× slower and not
  recommended.
- ~10 GB free disk for the quick path; ~30 GB for the full path.
- Network access to clone the repo and download 4DN `.mcool` files.

## Setup (both paths)

```bash
git clone https://github.com/meghanai28/hic-sr-vae
cd hic-sr-vae
py -m pip install torch numpy matplotlib pyyaml tqdm cooler scipy
mkdir -p data
# GM12878 — required for both paths. Fetched from 4DN's public S3 bucket
# (accession 4DNFIZL8OZE1; anonymous, no auth required).
curl -L -o data/GM12878.mcool \
  https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/356fab42-5562-4cfd-a3f8-592aa060b992/4DNFIZL8OZE1.mcool
```

---

# Quick validation path (≈20 min)

Uses the checkpoints already committed in `runs/paper_full/srvae_best.pt`
and `runs/paper_full_hicplus/hicplus_best.pt`. No training.

### A. HR tile extraction (GM12878)

```bash
py scripts/make_tiles.py --mcool data/GM12878.mcool --res 10000 \
    --out tiles/hr --patch 256 --stride 64 --offset-max 256
```

Writes `tiles/hr/{train,val,test}/{chrom}_{i}_{j}.npy` and
`tiles/hr/stats.json`. Only the `test/` split is needed for validation,
but the script generates all three by default.

### B. LR tile simulation (1/16 depth on test split)

```bash
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy" --out tiles/lr/test --frac 0.0625 --scale 2 --seed 42
```

### C. Tile-level held-out evaluation (Table 1)

```bash
py scripts/evaluate.py --config configs/paper_full.yaml \
    --ckpt runs/paper_full/srvae_best.pt \
    --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt \
    --outdir runs/paper_full/eval_quick
```

Produces `runs/paper_full/eval_quick/metrics.csv` with per-sample MSE,
SSIM, DISCO, HiC-Spector for SR-VAE, HiCPlus, and bicubic.

### D. Verify the headline claim

```bash
py -c "
import csv
rows = list(csv.DictReader(open('runs/paper_full/eval_quick/metrics.csv')))
def mean(col, method): 
    vs = [float(r[col]) for r in rows if r['method']==method]
    return sum(vs)/len(vs)
for m in ['srvae','hicplus','bicubic']:
    print(f'{m:8s}  MSE={mean(\"mse\",m):.4f}  SSIM={mean(\"ssim\",m):.4f}')
"
```

Expected output (approximate, seed 42):
```
srvae     MSE=0.0011  SSIM=0.6145
hicplus   MSE=0.0014  SSIM=0.5437
bicubic   MSE=0.0018  SSIM=0.4891
```

SR-VAE MSE should be ≈19% lower than HiCPlus; SSIM should be ≈13% higher.
That's the headline claim.

---

# Full reproduction path (≈8 h)

For reviewers re-deriving every table. Includes all quick-path steps
implicitly.

### Extra data download (K562, for cross-cell-line transfer)

```bash
# K562 accession 4DNFIOHY9ZX7 — 4DN public S3 (anonymous).
curl -L -o data/4DNFIOHY9ZX7.mcool \
  https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/a23c6e9a-114f-47d0-a13f-da28d75478f6/4DNFIOHY9ZX7.mcool
```

### 1. HR tile extraction

```bash
py scripts/make_tiles.py --mcool data/GM12878.mcool --res 10000 \
    --out tiles/hr --patch 256 --stride 64 --offset-max 256
```

### 2. LR tile simulation (1/16 depth, all splits)

```bash
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/train/*.npy" --out tiles/lr/train --frac 0.0625 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/val/*.npy"   --out tiles/lr/val   --frac 0.0625 --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy"  --out tiles/lr/test  --frac 0.0625 --scale 2 --seed 42
```

### 3. Train SR-VAE and HiCPlus (seed 42, the paper's headline config)

Skip this step if using the committed checkpoints — they are the output of
this step under seed 42.

```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae
py scripts/train.py --config configs/paper_full.yaml --model hicplus
```

SR-VAE writes to `runs/paper_full/`; HiCPlus is auto-suffixed to
`runs/paper_full_hicplus/`. Each run is deterministic under seed 42.
Expected runtime: ~2 h per model on RTX 4060 Laptop.

### 4. Tile-level held-out evaluation (Table 1)

```bash
py scripts/evaluate.py --config configs/paper_full.yaml \
    --ckpt runs/paper_full/srvae_best.pt \
    --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt \
    --outdir runs/paper_full/eval
```

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

`--save-npy` is required for steps 7 and 8.

### 6. Depth-robustness sweep (Table 5, no retraining)

```bash
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy" --out tiles/lr_frac08/test --frac 0.125   --scale 2 --seed 42
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy" --out tiles/lr_frac32/test --frac 0.03125 --scale 2 --seed 42
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

### 8. Biological validation II — chromatin loops (Table 8)

```bash
py scripts/loop_validation.py \
    --mosaic-dir runs/paper_full/reconstruction_chr19 \
    --split test --chrom 19 \
    --outdir runs/paper_full/loops_chr19 --sweep
```

### 9. Cross-cell-line zero-shot transfer (Tables 6 + 8 K562 rows)

```bash
py scripts/make_tiles.py --mcool data/4DNFIOHY9ZX7.mcool --res 10000 \
    --out tiles_k562/hr --patch 256 --stride 64 --offset-max 256 --splits test
py scripts/make_lr_tiles.py --hr-glob "tiles_k562/hr/test/*.npy" \
    --out tiles_k562/lr/test --frac 0.0625 --scale 2 --seed 42

py scripts/evaluate.py --config configs/paper_full_k562.yaml \
    --ckpt runs/paper_full/srvae_best.pt \
    --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt \
    --outdir runs/paper_full/eval_k562

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

### 10a. Deterministic-AE vs VAE on K562 zero-shot (Section 4.3 new table)

Requires tiles_k562 from step 9 to already exist.

```bash
py scripts/evaluate.py --config configs/paper_full_k562.yaml \
    --ckpt runs/paper_full_ae/srvae_best.pt \
    --hicplus-ckpt runs/paper_full_hicplus/hicplus_best.pt \
    --outdir runs/paper_full_ae/eval_k562
```

Expected summary: SR-VAE (Det-AE) MSE≈0.0012 SSIM≈0.7294 vs VAE MSE≈0.0011
SSIM≈0.7352 — VAE is 9% lower MSE and +0.58 pp SSIM on K562 zero-shot, while
both match to 3-4 decimal places on GM12878.

### 10. Seed variance (Table 2) and loss ablations (Table 3)

```bash
py scripts/train.py --config configs/paper_full.yaml --model srvae   --seed 43 --save-dir runs/paper_full_seed43
py scripts/train.py --config configs/paper_full.yaml --model srvae   --seed 44 --save-dir runs/paper_full_seed44
py scripts/train.py --config configs/paper_full.yaml --model hicplus --seed 43 --save-dir runs/paper_full_seed43
py scripts/train.py --config configs/paper_full.yaml --model hicplus --seed 44 --save-dir runs/paper_full_seed44
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

- **Quick path:** `runs/paper_full/eval_quick/metrics.csv` with the headline
  SR-VAE vs HiCPlus numbers.
- **Full path:** `runs/paper_full/**/metrics.csv` contains every number in
  every table of the paper.

## Notes for agentic reproduction

- All scripts accept `--set key=value` to override any YAML field at the
  CLI; no config edits required.
- Training is deterministic under a fixed seed
  (`torch.backends.cudnn.deterministic = True`,
  `use_deterministic_algorithms(True)`).
- The committed checkpoints in `runs/paper_full/srvae_best.pt` and
  `runs/paper_full_hicplus/hicplus_best.pt` are the exact outputs of step 3
  under seed 42 — so the quick path and the full path (if step 3 is
  re-run) produce identical eval metrics.
- K562 tile filenames carry a `chr` prefix (e.g. `chr19_0_0.npy`);
  GM12878 does not (`19_0_0.npy`). `scripts/reconstruct_chromosome.py`
  handles both.
- Reconstructed `.npy` mosaics (~133 MB each) are excluded from the repo
  via `.gitignore`; regenerate with step 5.
