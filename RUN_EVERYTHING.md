# Run Everything

This guide explains the recommended execution order for the modular wildfire ignition prediction repository.

The upgraded architecture uses a coarse-to-fine cascade:

```text
Alberta-wide scan
↓
Model B: 1 km, 64×64 patch gatekeeper
↓
Only high-risk coarse patches continue
↓
Generate 25 m, 64×64 fine patches inside/around flagged coarse regions
↓
Model A: 25 m spatial U-Net refiner
↓
Final ignition probability heatmap
```

This keeps the original two-stage detector/refiner concept while making province-wide inference practical. The repository is designed to keep full datasets and trained model files private. Commands below assume you are running from the repository root.

## 1. Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

If the repository contains `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

At minimum, the training and inference scripts require TensorFlow, NumPy, and PyTorch's dataset base class dependency used by `src/data/dataset.py`.

## 3. Confirm the module imports compile

Run these checks before training:

```bash
python -m py_compile src/data/dataset.py
python -m py_compile src/data/keras_generator.py
python -m py_compile src/data/npz_loader.py
python -m py_compile src/data/compute_channel_stats.py
python -m py_compile src/models/unet.py
python -m py_compile src/models/loaders.py
python -m py_compile src/training/losses.py
python -m py_compile src/training/metrics.py
python -m py_compile src/training/train_model_a.py
python -m py_compile src/training/train_model_b.py
python -m py_compile pipeline/two_stage_inference.py
```

## 4. Prepare patch folders

The modular training code expects `.npz` patch files named like:

```text
patch_*.npz
```

Each patch should contain:

- multiple feature arrays shaped `(64, 64)`
- a label/mask array under the key `class`

Use separate patch folders for the two model resolutions:

```text
patches_1km_balanced/
├── train/
├── val/
└── test/

patches_25m_balanced/
├── train/
├── val/
└── test/
```

Model B should use the 1 km folders. Model A should use the 25 m folders.

For full private training data, use your local/private patch folders instead of committing them to GitHub.

## 5. Compute channel normalization statistics

Use only the training split to compute normalization statistics.

### Model B: 1 km gatekeeper stats

```bash
python -m src.data.compute_channel_stats \
  --data-dir patches_1km_balanced/train \
  --output patches_1km_balanced/channel_stats.json \
  --max-files 300
```

To scan all 1 km training patches:

```bash
python -m src.data.compute_channel_stats \
  --data-dir patches_1km_balanced/train \
  --output patches_1km_balanced/channel_stats.json \
  --max-files -1
```

### Model A: 25 m refiner stats

```bash
python -m src.data.compute_channel_stats \
  --data-dir patches_25m_balanced/train \
  --output patches_25m_balanced/channel_stats.json \
  --max-files 300
```

To scan all 25 m training patches:

```bash
python -m src.data.compute_channel_stats \
  --data-dir patches_25m_balanced/train \
  --output patches_25m_balanced/channel_stats.json \
  --max-files -1
```

The output JSON contains:

```json
{
  "mean": [],
  "std": [],
  "feature_keys": [],
  "label_key": "class",
  "num_files_used": 0,
  "num_pixels_used": 0,
  "std_floor": 0.1,
  "skipped_files": []
}
```

## 6. Train Model B: 1 km patch-level gatekeeper

Model B is the first stage. It scans coarse 1 km patches across Alberta and decides whether a coarse patch is likely enough to justify 25 m refinement.

```bash
python -m src.training.train_model_b \
  --train-dir patches_1km_balanced/train \
  --val-dir patches_1km_balanced/val \
  --channel-stats patches_1km_balanced/channel_stats.json \
  --output-model models/model_B_1km_gatekeeper.keras \
  --resolution-m 1000 \
  --patch-size 64 \
  --epochs 20 \
  --batch-size 16
```

Notes:

- `--resolution-m 1000` records the intended pixel resolution in logs.
- The model input channel count is inferred from the `.npz` feature keys.
- The U-Net still receives a `(64, 64, C)` tensor.

## 7. Train Model A: 25 m spatial refiner

Model A is the second stage. It produces the pixel-wise ignition probability map on fine 25 m patches.

```bash
python -m src.training.train_model_a \
  --train-dir patches_25m_balanced/train \
  --val-dir patches_25m_balanced/val \
  --channel-stats patches_25m_balanced/channel_stats.json \
  --output-model models/model_A_25m_spatial_unet.keras \
  --resolution-m 25 \
  --patch-size 64 \
  --epochs 20 \
  --batch-size 16
```

Notes:

- `--resolution-m 25` records the intended pixel resolution in logs.
- The model input channel count is inferred from the `.npz` feature keys.
- Use the 25 m channel stats with Model A, not the 1 km stats.

## 8. Sample-data smoke workflow

If you only have `data/sample_patches`, use it as a syntax/smoke test, not as proof that the 1 km/25 m cascade is trained correctly.

```bash
python -m src.data.compute_channel_stats \
  --data-dir data/sample_patches/train \
  --output data/sample_patches/channel_stats.json

python -m src.training.train_model_b \
  --train-dir data/sample_patches/train \
  --val-dir data/sample_patches/test \
  --channel-stats data/sample_patches/channel_stats.json \
  --output-model models/sample_model_B_gatekeeper.keras \
  --resolution-m 1000 \
  --epochs 1

python -m src.training.train_model_a \
  --train-dir data/sample_patches/train \
  --val-dir data/sample_patches/test \
  --channel-stats data/sample_patches/channel_stats.json \
  --output-model models/sample_model_A_spatial_unet.keras \
  --resolution-m 25 \
  --epochs 1
```

## 9. Run two-stage inference on one patch

The current single-patch inference script still assumes both stages run on one already-created `.npz` patch. It is useful for testing model loading and thresholding, but it is not yet the final Alberta-wide 1 km → 25 m production inference pipeline.

```bash
python pipeline/two_stage_inference.py \
  --model-a models/model_A_25m_spatial_unet.keras \
  --model-b models/model_B_1km_gatekeeper.keras \
  --patch data/sample_patches/test/patch_0001.npz \
  --stats data/sample_patches/channel_stats.json \
  --threshold-a 0.30 \
  --threshold-b 0.25
```

The production inference pipeline still needs a later module that:

1. Tiles Alberta into 1 km `64×64` patches.
2. Runs Model B over those coarse patches.
3. Converts flagged 1 km regions into corresponding 25 m windows.
4. Generates 25 m `64×64` patches for those windows.
5. Runs Model A only on the selected fine patches.
6. Mosaics Model A outputs into a final heatmap.

## 10. Important consistency rules

Use separate normalization files:

```text
patches_1km_balanced/channel_stats.json  -> Model B only
patches_25m_balanced/channel_stats.json  -> Model A only
```

The code sorts feature keys alphabetically before stacking channels. This keeps channel order consistent across:

- `src/data/compute_channel_stats.py`
- `src/data/dataset.py`
- `src/data/npz_loader.py`

Do not compute channel statistics from validation or test data. Use only the training split.

## 11. What should not be committed

Do not commit:

```text
*.npz
*.npy
*.tif
*.tiff
*.keras
*.h5
*.pt
*.pth
*.pkl
*.joblib
models/
large private datasets
full generated outputs
.env files
```

It is safe to commit:

```text
source code
README files
research report files
small sample patches, if intentionally public
plots
small sample outputs
```

## 12. Recommended full training workflow

```bash
# 1. Check syntax
python -m py_compile src/data/compute_channel_stats.py
python -m py_compile src/models/unet.py
python -m py_compile src/models/loaders.py
python -m py_compile src/training/train_model_a.py
python -m py_compile src/training/train_model_b.py
python -m py_compile pipeline/two_stage_inference.py

# 2. Compute 1 km stats for Model B
python -m src.data.compute_channel_stats \
  --data-dir patches_1km_balanced/train \
  --output patches_1km_balanced/channel_stats.json

# 3. Train 1 km gatekeeper
python -m src.training.train_model_b \
  --train-dir patches_1km_balanced/train \
  --val-dir patches_1km_balanced/val \
  --channel-stats patches_1km_balanced/channel_stats.json \
  --output-model models/model_B_1km_gatekeeper.keras \
  --resolution-m 1000

# 4. Compute 25 m stats for Model A
python -m src.data.compute_channel_stats \
  --data-dir patches_25m_balanced/train \
  --output patches_25m_balanced/channel_stats.json

# 5. Train 25 m spatial refiner
python -m src.training.train_model_a \
  --train-dir patches_25m_balanced/train \
  --val-dir patches_25m_balanced/val \
  --channel-stats patches_25m_balanced/channel_stats.json \
  --output-model models/model_A_25m_spatial_unet.keras \
  --resolution-m 25
```

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the repository root. If needed, use:

```bash
export PYTHONPATH=.
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "."
```

### `No patch files found`

Confirm the folder contains files matching:

```text
patch_*.npz
```

### `Feature key mismatch`

At least one patch has different feature keys from the first valid patch. Regenerate or inspect that patch before training.

### Shape error between model and patch tensors

The training scripts now infer the channel count from the dataset. If a shape error happens, check:

1. whether all patches in the folder have identical feature keys,
2. whether the model was trained with the same feature set used during inference,
3. whether Model B is using 1 km stats and Model A is using 25 m stats.
