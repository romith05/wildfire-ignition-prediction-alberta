# Training Log

This file tracks Model B gatekeeper and Model A spatial-refiner experiments for the wildfire ignition project.

## Goal

Model B is the 1 km patch gatekeeper. It should keep ignition recall high while reducing the number of coarse patches sent to 25 m Model A refinement.

Model A is the 25 m spatial refiner. It should produce accurate pixel-level ignition probability masks on fine-resolution patches, including patches that are true positives from Model B and false-positive patches that still pass the gate.

Current cascade:

```text
1 km scan -> Model B gatekeeper -> selected coarse patches -> 25 m Model A refiner
```

## Run 001 — Initial Phased 1 km Model B Baseline

Date: 2026-05-11

### Purpose

Start the training log from the first successful phased 1 km Model B run after switching to this flow:

```text
1. train 1 km ignition-only base U-Net
2. initialize Model B from the base model
3. train Model B on balanced 1 km data
4. save each phase model separately
```

### Data and Model Inputs

Training data:

```text
/mnt/work/wildfire/1km/patches_1km_balanced/train
```

Validation data:

```text
/mnt/work/wildfire/1km/patches_1km_balanced/val
```

Channel stats:

```text
/mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json
```

Base model:

```text
models/model_A_1km_base_unet.keras
```

Output model prefix:

```text
models/model_B_1km_gatekeeper.keras
```

### Phase Parameters

| Phase | Goal | Loss | Epochs | LR | Patch suppression |
|---|---|---|---:|---:|---:|
| 1 | Preserve ignition sensitivity from base model | focal | 3 | 1e-4 | none |
| 2 | Begin reducing no-ignition patch activations | combined | 5 | 7.5e-5 | 0.10 |
| 3 | Strongly suppress patch false positives | combined | 5 | 5e-5 | 0.50 |
| 4 | Recover recall and calibrate | combined | 3 | 3e-5 | 0.30 |

### Why These Parameters Were Used

- Phase 1 avoids patch suppression so the base ignition signal is not destroyed immediately.
- Phase 2 introduces weak patch suppression to reduce coarse false alarms.
- Phase 3 tests stronger patch suppression.
- Phase 4 relaxes suppression to recover recall.
- Learning rate decreases phase by phase to make later updates less disruptive.

### Results Summary

| Phase | val_patch_fp_rate_05 | val_precision | val_recall | val_loss | Interpretation |
|---|---:|---:|---:|---:|---|
| 1 | 0.1798 | 0.7381 | 0.9958 | 1.1371e-05 | Excellent recall, patch false positives still high. |
| 2 | 0.0499 | 0.7086 | 0.8966 | 0.0231 | Big patch-FP reduction, but recall dropped. |
| 3 | 0.0639 at best-val-loss epoch | 0.4008 | 0.9483 | 0.1121 | Suppression likely too strong; precision degraded. |
| 4 | 0.0562 final, 0.0485 best checkpoint | 0.3996 final, 0.3989 best checkpoint | 0.9344 final, 0.9092 best checkpoint | 0.0820 final, 0.0784 best checkpoint | Recall recovered, but precision stayed weak. |

### Phase 2 Threshold Sweep Decision

Decision date: 2026-05-14

Selected operating point for the current Phase 2 model:

```text
model: models/model_B_1km_gatekeeper_phase2.keras
threshold: 0.40
```

Reason:

- Threshold `0.40` is the safer default gatekeeper threshold because it keeps recall higher than `0.45` while still reducing the 25 m refinement workload.
- Threshold `0.45` remains a stricter/high-confidence option, but it misses more positive patches.

Phase 2 sweep comparison:

| Threshold | Patch Recall | Patch Precision | Patch FP Rate | Pass Rate | Passed Patches | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.9441 | 0.8848 | 0.1250 | 0.5380 | 764 | 676 | 88 | 616 | 40 |
| 0.45 | 0.9260 | 0.8972 | 0.1080 | 0.5204 | 739 | 663 | 76 | 628 | 53 |

Interpretation:

```text
Moving from 0.40 to 0.45 saves only 25 passed patches, but misses 13 additional positive patches. For a wildfire gatekeeper, the recall loss is not worth the small workload reduction at this stage.
```

### Phase 4 Sweep Comparison

Phase 4 was also evaluated after the Phase 2 decision.

Important Phase 4 points:

| Threshold | Patch Recall | Patch Precision | Patch FP Rate | Pass Rate | Passed Patches | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.9455 | 0.8724 | 0.1406 | 0.5465 | 776 | 677 | 99 | 605 | 39 |
| 0.45 | 0.9399 | 0.8890 | 0.1193 | 0.5331 | 757 | 673 | 84 | 620 | 43 |
| 0.50 | 0.9358 | 0.8957 | 0.1108 | 0.5268 | 748 | 670 | 78 | 626 | 46 |

Comparison against Phase 2:

```text
Phase 2 @ 0.40: recall 0.9441, patch FP rate 0.1250, pass rate 0.5380.
Phase 4 @ 0.45: recall 0.9399, patch FP rate 0.1193, pass rate 0.5331.
```

Interpretation:

```text
Phase 4 @ 0.45 is a strong backup operating point. It has slightly lower recall than Phase 2 @ 0.40, but slightly lower patch false positive rate and pass rate. The tradeoff is close.
```

Current decision after reviewing Phase 4:

```text
Default operating point: models/model_B_1km_gatekeeper_phase2.keras @ threshold 0.40
Backup stricter point: models/model_B_1km_gatekeeper_phase4.keras @ threshold 0.45
```

### Takeaways

1. Phase 1 worked as a sensitivity warm start.
2. Phase 2 gave the cleanest patch false-positive reduction.
3. Phase 3 appears too aggressive with `lambda_patch=0.50`.
4. Phase 4 recovered recall but did not clearly beat Phase 2 at the selected operating point.
5. Do not automatically choose the final model.
6. Phase 2 threshold `0.40` is the current default operating point.
7. Phase 4 threshold `0.45` is the backup stricter operating point.

### Candidate Models To Evaluate

```text
models/model_B_1km_gatekeeper_phase2.keras
models/model_B_1km_gatekeeper_phase4.keras
```

### Recommended Next Evaluation

Use the selected Phase 2 model in the first coarse-to-fine pipeline test.

Track:

```text
number of 1 km patches scanned
number of 1 km patches passed by Model B
percentage of coarse grid passed to 25 m refinement
number of known ignition regions retained
number of false coarse regions passed
runtime reduction compared with all-25 m scanning
```

## Run 002 — Initial 25 m Model A Spatial Refiner Smoke Test

Date: 2026-05-15

### Purpose

Start Model A 25 m spatial-refiner training after Model B reached a usable gatekeeper operating point.

This was a short 5-epoch smoke test to verify:

```text
1. 25 m patch loading works.
2. 25 m channel normalization works.
3. Model A trains without NaN loss.
4. The updated shape-safe training script works.
```

### Training Choice

Model A should be trained on balanced 25 m patches, not ignition-only patches.

Reason:

```text
Model B false positives will not be zero. Some no-ignition coarse regions will still reach Model A, so Model A must learn to output near-empty masks on no-ignition fine patches instead of hallucinating ignition.
```

### Initial Smoke-Test Result

| Epoch | train_loss | train_precision | train_recall | train_pixel_fp_rate | val_loss | val_precision | val_recall | val_pixel_fp_rate | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 6.9683e-05 | 0.4731 | 0.6956 | 5.7471e-04 | 3.0070e-06 | 0.5056 | 0.4965 | 3.3775e-05 | Model started learning. |
| 2 | 3.8690e-06 | 0.7234 | 0.9874 | 4.8429e-05 | 2.6111e-06 | 0.5056 | 0.5035 | 3.7718e-05 | Best validation loss. |
| 3 | 3.1632e-06 | 0.7377 | 0.9890 | 4.3717e-05 | 3.1566e-06 | 0.5056 | 0.5021 | 3.6175e-05 | Validation did not improve. |
| 4 | 2.9941e-06 | 0.7376 | 0.9901 | 4.3918e-05 | 2.6640e-06 | 0.5056 | 0.5035 | 3.8576e-05 | Validation nearly flat. |
| 5 | 2.8868e-06 | 0.7439 | 0.9903 | 4.2299e-05 | 2.6566e-06 | 0.5056 | 0.5035 | 3.6004e-05 | LR reduced after no improvement. |

### Interpretation

- Training metrics improved rapidly.
- Validation precision and recall stayed around `0.50`, which suggests threshold `0.5` may not be the right evaluation threshold for sparse 25 m masks.
- Best validation loss occurred at epoch 2.
- The run confirms the code path works, but it is not enough to judge final Model A quality.

### Decision

Continue with balanced 25 m training, but lower the learning rate for the next run.

Recommended next command:

```bash
python -m src.training.train_model_a \
  --train-dir /mnt/work/wildfire/25m/patches_25m_balanced/train \
  --val-dir /mnt/work/wildfire/25m/patches_25m_balanced/val \
  --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
  --output-model models/model_A_25m_spatial_unet.keras \
  --resolution-m 25 \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 5e-5
```

### Next Evaluation Need

After a longer balanced run, Model A should be evaluated with threshold sweeps and spatial metrics:

```text
Dice
IoU
precision
recall
false-positive pixels
empty-patch behavior
visual sample outputs
```

## Future Run Template

### Run XXX — Name

Date:

Code branch:

Commit:

Purpose:

### Data

Training data:

Validation data:

Channel stats:

Base model:

Output model:

### Command

```bash
paste command here
```

### Parameter Changes

| Parameter | Previous | New | Reason |
|---|---:|---:|---|
| phase2_lambda_patch |  |  |  |
| phase3_lambda_patch |  |  |  |
| phase4_lambda_patch |  |  |  |
| phase1_lr |  |  |  |
| phase2_lr |  |  |  |
| phase3_lr |  |  |  |
| phase4_lr |  |  |  |
| model_a_learning_rate |  |  |  |
| model_a_focal_alpha |  |  |  |
| model_a_focal_gamma |  |  |  |

### Results

| Phase / Model | val_patch_fp_rate_05 | val_precision | val_recall | val_loss | Notes |
|---|---:|---:|---:|---:|---|
| Model B Phase 1 |  |  |  |  |  |
| Model B Phase 2 |  |  |  |  |  |
| Model B Phase 3 |  |  |  |  |  |
| Model B Phase 4 |  |  |  |  |  |
| Model A |  |  |  |  |  |

### Interpretation

What improved?

What got worse?

Did the tuning do what it was supposed to do?

### Decision

Choose one:

```text
Keep this run
Reject this run
Threshold-sweep this run
Retrain with adjusted parameters
Collect hard negatives next
Proceed to pipeline test
```

### Next Planned Change

Write the next tuning adjustment and why.
