# Run Everything

This guide explains the recommended execution order for the modular wildfire ignition prediction repository.

The repository is designed to keep full datasets and trained model files private. Commands below assume you are running from the repository root.

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

Expected training/validation layout example:

```text
data/sample_patches/
├── train/
│   ├── patch_0001.npz
│   └── ...
└── test/
    ├── patch_0001.npz
    └── ...
```

For full private training data, use your local/private patch folders instead of committing them to GitHub.

## 5. Compute channel normalization statistics

Use only the training split to compute normalization statistics.

For sample patches:

```bash
python -m src.data.compute_channel_stats \
  --data-dir data/sample_patches/train \
  --output data/sample_patches/channel_stats.json
```

For full private training patches:

```bash
python -m src.data.compute_channel_stats \
  --data-dir patches_25m_balanced/train \
  --output channel_stats.json \
  --max-files 300
```

To scan all available training patches:

```bash
python -m src.data.compute_channel_stats \
  --data-dir patches_25m_balanced/train \
  --output channel_stats.json \
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

## 6. Train Model B: patch-level gatekeeper

Model B is the first stage. It decides whether a patch is likely to contain ignition.

Sample-data command:

```bash
python -m src.training.train_model_b \
  --train-dir data/sample_patches/train \
  --val-dir data/sample_patches/test \
  --channel-stats data/sample_patches/channel_stats.json \
  --output-model models/model_B_gatekeeper.keras \
  --epochs 20 \
  --batch-size 16
```

Private full-data command example:

```bash
python -m src.training.train_model_b \
  --train-dir patches_25m_balanced/train \
  --val-dir patches_25m_balanced/val \
  --channel-stats channel_stats.json \
  --output-model models/model_B_gatekeeper.keras \
  --epochs 20 \
  --batch-size 16
```

## 7. Train Model A: spatial refiner

Model A produces the pixel-wise ignition probability map.

Sample-data command:

```bash
python -m src.training.train_model_a \
  --train-dir data/sample_patches/train \
  --val-dir data/sample_patches/test \
  --channel-stats data/sample_patches/channel_stats.json \
  --output-model models/model_A_spatial_unet.keras \
  --epochs 20 \
  --batch-size 16
```

Private full-data command example:

```bash
python -m src.training.train_model_a \
  --train-dir patches_25m_balanced/train \
  --val-dir patches_25m_balanced/val \
  --channel-stats channel_stats.json \
  --output-model models/model_A_spatial_unet.keras \
  --epochs 20 \
  --batch-size 16
```

## 8. Run two-stage inference on one patch

After both models are trained, run:

```bash
python pipeline/two_stage_inference.py \
  --model-a models/model_A_spatial_unet.keras \
  --model-b models/model_B_gatekeeper.keras \
  --patch data/sample_patches/test/patch_0001.npz \
  --stats data/sample_patches/channel_stats.json \
  --threshold-a 0.30 \
  --threshold-b 0.25
```

The script prints:

- number of feature channels
- whether the patch passed the Model B gate
- Stage B confidence
- Stage A confidence
- predicted ignition pixels
- true ignition pixels, if the patch contains the `class` label

## 9. Important consistency rules

Use the same `channel_stats.json` for:

- Model B training
- Model A training
- two-stage inference

The code sorts feature keys alphabetically before stacking channels. This keeps channel order consistent across:

- `src/data/compute_channel_stats.py`
- `src/data/dataset.py`
- `src/data/npz_loader.py`

Do not compute channel statistics from validation or test data. Use only the training split.

## 10. What should not be committed

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

## 11. Recommended full workflow

```bash
# 1. Check syntax
python -m py_compile src/data/compute_channel_stats.py
python -m py_compile src/models/unet.py
python -m py_compile src/models/loaders.py
python -m py_compile src/training/train_model_a.py
python -m py_compile src/training/train_model_b.py
python -m py_compile pipeline/two_stage_inference.py

# 2. Compute normalization stats
python -m src.data.compute_channel_stats \
  --data-dir data/sample_patches/train \
  --output data/sample_patches/channel_stats.json

# 3. Train gatekeeper
python -m src.training.train_model_b \
  --train-dir data/sample_patches/train \
  --val-dir data/sample_patches/test \
  --channel-stats data/sample_patches/channel_stats.json \
  --output-model models/model_B_gatekeeper.keras

# 4. Train spatial refiner
python -m src.training.train_model_a \
  --train-dir data/sample_patches/train \
  --val-dir data/sample_patches/test \
  --channel-stats data/sample_patches/channel_stats.json \
  --output-model models/model_A_spatial_unet.keras

# 5. Run two-stage inference
python pipeline/two_stage_inference.py \
  --model-a models/model_A_spatial_unet.keras \
  --model-b models/model_B_gatekeeper.keras \
  --patch data/sample_patches/test/patch_0001.npz \
  --stats data/sample_patches/channel_stats.json
```

## 12. Troubleshooting

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

### Shape error: expected `(64, 64, 17)`

The model architecture currently expects 17 feature channels. If your patch generation changes the number of features, update the model input shape consistently in the training scripts and inference pipeline.
