# Hi-C Super-Resolution VAE

Super-resolution of Hi-C contact matrices using a Variational Autoencoder with U-Net skip connections, SE channel attention, and adaptive multi-band tiling.

## Project Structure

```
hic-sr-vae/
├── configs/
│   └── default.yaml
├── src/
│   ├── model.py                # SR-VAE architecture (zoom-conditioned)
│   ├── datasets.py             # paired tile dataset + zoom parsing
│   └── utils.py                # OE norm, losses, spatial helpers
├── scripts/
│   ├── make_tiles.py           # extract multi-band HR tiles from .mcool
│   ├── make_lr_tiles.py        # binomial thinning → LR tiles
│   ├── train.py                # train
│   ├── evaluate.py             # eval with triptychs + metrics
│   └── reconstruct_chromosome.py  # full chromosome reconstruction + Hi-C metrics
├── data/                       # (gitignored) .mcool files
├── tiles/                      # (gitignored) extracted .npy tiles
│   ├── hr/{train,val,test}/
│   └── lr/{train,val,test}/
└── runs/sr_vae/                # (gitignored) checkpoints + outputs
    ├── sr_vae_best.pt
    ├── sr_vae_last.pt
    ├── eval/
    └── reconstruction/
```

## Approach

### Multi-band tiling

Tiles are extracted from the upper triangle at 10kb resolution using an adaptive scheme:

| Distance from diagonal | Method | Stride | Zoom | Purpose |
|---|---|---|---|---|
| 0–256 bins | lateral | 64 | 1× | Dense overlap, finest detail (TADs, loops) |
| 256–512 bins | lateral | 128 | 1× | Good detail |
| 512–1024 bins | zoom-out | 512 | 2× | Moderate compression |
| 1024–2048 bins | zoom-out | 1024 | 4× | Compartment-level |
| 2048–4096 bins | zoom-out | 2048 | 8× | Broad structure |
| 4096–8192 bins | zoom-out | 4096 | 16× | Far off-diagonal |

Near-diagonal gets dense native-resolution tiles for maximum quality where biological features live. Far off-diagonal gets progressively zoomed-out tiles (larger genomic windows downsampled to 256×256), providing efficient coverage of weaker compartment-level signal.

A zoom embedding in the latent space conditions the model on scale, so one model handles all bands.

### Training

LR tiles are simulated via binomial thinning at 1/16 read fraction. Loss: L1 + SSIM + Sobel gradient + KL divergence (warm-up schedule).

### Reconstruction

All bands feed into a single global accumulator with cosine window blending. Tiles from all zoom levels contribute simultaneously — no hard boundaries. Zoomed tiles are bilinearly upsampled back to native resolution before accumulation. The upper triangle is mirrored to produce a full symmetric matrix.

### Evaluation metrics

- **MSE** / **SSIM** — standard image metrics
- **HiCRep SCC** — stratum-adjusted correlation coefficient (per-diagonal-distance correlation, smoothed)
- **GenomeDISCO** — random walk concordance between contact matrices
- **Per-distance MSE** — MSE broken down by genomic distance

**Splits:** train=chr1-16, val=chr17-18, test=chr19-22

## Setup

```bash
pip install torch numpy matplotlib pyyaml tqdm cooler scipy hicrep
```

## Pipeline

### 1. Extract HR tiles

```bash
py scripts/make_tiles.py \
    --mcool data/GM12878.mcool \
    --res 10000 \
    --out tiles/hr
```

### 2. Generate LR tiles

```bash
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/train/*.npy" --out tiles/lr/train
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/val/*.npy"   --out tiles/lr/val
py scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy"  --out tiles/lr/test
```

### 3. Train

```bash
py scripts/train.py --config configs/default.yaml

# Resume from checkpoint
py scripts/train.py --config configs/default.yaml --resume runs/sr_vae/sr_vae_last.pt
```

### 4. Evaluate

```bash
py scripts/evaluate.py --config configs/default.yaml
```

Outputs triptych PNGs (LR / SR-VAE / HR) to `runs/sr_vae/eval/`.

### 5. Reconstruct chromosome

```bash
py scripts/reconstruct_chromosome.py \
    --mcool data/GM12878.mcool \
    --res 10000 \
    --chrom chr17 \
    --ckpt runs/sr_vae/sr_vae_best.pt \
    --config configs/default.yaml \
    --save-npy
```

Outputs:
- `chr17_oe.png` — LR vs SR-VAE vs HR (OE normalized)
- `chr17_raw.png` — LR vs HR (log1p raw counts, depth comparison)
- `chr17_distance_mse.png` — per-distance MSE curve
- Printed metrics: MSE, SSIM, HiCRep SCC, GenomeDISCO

## Config

Key fields in `configs/default.yaml`:

| Field | Default | Description |
|---|---|---|
| `vae.z_ch` | 64 | Latent channel dim |
| `vae.base_ch` | 64 | Base feature channels |
| `vae.epochs` | 100 | Training epochs |
| `vae.lr` | 2e-4 | Learning rate |
| `vae.batch_size` | 2 | Batch size |
| `num_zooms` | 5 | Zoom levels (1×/2×/4×/8×/16×) |
| `loss.rec_w` | 0.50 | L1 loss weight |
| `loss.ssim_w` | 0.25 | SSIM loss weight |
| `loss.grad_w` | 0.25 | Sobel gradient loss weight |
| `loss.kl_warmup_epochs` | 60 | Epochs before KL ramps in |
