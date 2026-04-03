# Hi-C Super-Resolution VAE

Denoising and super-resolution of Hi-C contact matrices using a Variational Autoencoder with U-Net skip connections, SE channel attention, and multi-scale diagonal tiling.

## Project Structure

```
hic-sr-vae/
├── configs/
│   └── default.yaml            # training config
├── src/
│   ├── model.py                # SR-VAE architecture
│   ├── datasets.py             # paired tile dataset + zoom parsing
│   └── utils.py                # OE norm, losses, spatial helpers
├── scripts/
│   ├── make_tiles.py           # step 1: extract multi-scale HR tiles from .mcool
│   ├── make_lr_tiles.py        # step 2: binomial thinning to generate LR tiles
│   ├── train.py                # step 3: train
│   ├── evaluate.py             # step 4: eval with triptychs + metrics
│   └── reconstruct_chromosome.py  # full chromosome reconstruction
├── data/                       # (gitignored) .mcool files
├── tiles/                      # (gitignored) extracted .npy tiles
│   ├── hr/{train,val,test}/
│   └── lr/{train,val,test}/
└── runs/sr_vae/                # (gitignored) checkpoints + eval output
    ├── sr_vae_best.pt
    ├── sr_vae_last.pt
    └── eval/
```

## Approach

Tiles are extracted **on the diagonal only** at three zoom levels (1×/2×/4×), each downsampled to 256×256 pixels. This covers 2.56 Mb, 5.12 Mb, and 10.24 Mb windows respectively. A zoom embedding conditions the model on scale, so one model handles all three. LR tiles are simulated via binomial thinning at 1/16 read fraction. The reconstruction pass runs all three zoom levels and accumulates into a full-chromosome matrix using cosine windowed overlap-add.

**Splits:** train=chr1-16, val=chr17-18, test=chr19-22

## Setup

```bash
pip install torch numpy matplotlib pyyaml tqdm cooler
```

## Pipeline

### 1. Extract HR tiles

```bash
python scripts/make_tiles.py \
    --mcool data/GM12878.mcool \
    --res 10000 \
    --out tiles/hr \
    --patch 256 \
    --zooms 1,2,4
```

### 2. Generate LR tiles

```bash
python scripts/make_lr_tiles.py --hr-glob "tiles/hr/train/*.npy" --out tiles/lr/train --frac 0.0625
python scripts/make_lr_tiles.py --hr-glob "tiles/hr/val/*.npy"   --out tiles/lr/val   --frac 0.0625
python scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy"  --out tiles/lr/test  --frac 0.0625
```

### 3. Train

```bash
python scripts/train.py --config configs/default.yaml

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume runs/sr_vae/sr_vae_last.pt
```

### 4. Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml --ckpt runs/sr_vae/sr_vae_best.pt
```

Outputs triptych PNGs (LR / SR-VAE / HR) to `runs/sr_vae/eval/` with zoom level and window size labeled on each figure.

### 5. Reconstruct chromosome

```bash
python scripts/reconstruct_chromosome.py \
    --mcool data/GM12878.mcool \
    --lr-res 10000 \
    --chrom chr17 \
    --ckpt runs/sr_vae/sr_vae_best.pt \
    --config configs/default.yaml \
    --zooms 1,2,4 \
    --stride 64 \
    --save-npy
```

## Config

Key fields in `configs/default.yaml`:

| Field | Default | Description |
|-------|---------|-------------|
| `vae.z_ch` | 64 | Latent channel dim |
| `vae.base_ch` | 64 | Base feature channels |
| `vae.epochs` | 100 | Training epochs |
| `vae.lr` | 2e-4 | Learning rate |
| `vae.batch_size` | 2 | Batch size |
| `num_zooms` | 3 | Number of zoom levels (1×/2×/4×) |
| `loss.rec_w` | 0.50 | L1 loss weight |
| `loss.ssim_w` | 0.25 | SSIM loss weight |
| `loss.grad_w` | 0.25 | Sobel gradient loss weight |
| `loss.dist_alpha` | 1.0 | Distance weight map strength |
| `loss.kl_warmup_epochs` | 60 | Epochs before KL term ramps in |
