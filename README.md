# Hi-C Super-Resolution VAE

Denoising and super-resolution of Hi-C contact matrices using a
Variational Autoencoder with U-Net skip connections and SE channel attention.

## Project Structure

```
hic-sr-vae/
│
├── configs/                    # YAML configuration files
│   └── default.yaml            #   default training config
│
├── src/                        # Core library (importable modules)
│   ├── __init__.py
│   ├── model.py                #   SR-VAE architecture (encoder, decoder, SE, VAE)
│   ├── datasets.py             #   PyTorch Dataset/DataLoader for paired Hi-C tiles
│   └── utils.py                #   OE normalization, losses, spatial helpers
│
├── scripts/                    # Runnable entry points (CLI tools)
│   ├── make_tiles.py           #   Step 1: extract HR tiles from .mcool
│   ├── make_lr_tiles.py        #   Step 2: generate LR tiles via binomial thinning
│   ├── train.py                #   Step 3: train the SR-VAE
│   ├── evaluate.py             #   Step 4: eval with triptychs + metrics
│   ├── visualize_data.py       #   Sanity check: inspect LR/HR pairs before training
│   └── run_pipeline.py         #   One-command: data prep → train → eval
│
├── docs/                       # Documentation and notes
│   └── architecture.md         #   Architecture design rationale
│
├── requirements.txt
└── README.md                   # This file
```

### Directories created at runtime (gitignored)

```
├── data/                       # Raw .mcool files 
│   └── experiment.mcool
│
├── tiles/                      # Extracted .npy tiles
│   ├── hr/
│   │   ├── train/              #   HR tiles for chr1-16
│   │   ├── val/                #   HR tiles for chr17-18
│   │   └── test/               #   HR tiles for chr19-22
│   └── lr/
│       ├── train/              #   LR tiles
│       ├── val/
│       └── test/
│
└── runs/                       # Training outputs
    └── sr_vae/
        ├── sr_vae_best.pt      #   Best checkpoint (by val loss)
        ├── sr_vae_last.pt      #   Latest checkpoint (for resume)
        └── eval/               #   Triptych PNGs + metrics
```

## Pipeline Information

### 1. Install dependencies

```bash
pip install torch numpy matplotlib pyyaml tqdm cooler
```

### 2. Prepare data

```bash
# Extract HR tiles from your .mcool
python scripts/make_tiles.py --mcool data/GM12878.mcool --res 10000 --out tiles/hr

# Generate LR tiles (16x coverage downsampling)
python scripts/make_lr_tiles.py --hr-glob "tiles/hr/train/*.npy" --out tiles/lr/train
python scripts/make_lr_tiles.py --hr-glob "tiles/hr/val/*.npy"   --out tiles/lr/val
python scripts/make_lr_tiles.py --hr-glob "tiles/hr/test/*.npy"  --out tiles/lr/test
```

### 3. Sanity check your data

```bash
python scripts/visualize_data.py --config configs/default.yaml --n 6
```

### 4. Train

```bash
python scripts/train.py --config configs/default.yaml
```

### 5. Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml --ckpt runs/sr_vae/sr_vae_best.pt
```
