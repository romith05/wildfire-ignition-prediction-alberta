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
- Validation precision and recall appeared stuck around `0.50` before the evaluation label-shape bug was fixed.
- The run confirmed the code path worked, but the initial validation interpretation was unreliable.

## Run 003 — Model A Lower-LR Training and Corrected Validation Diagnostic

Date: 2026-05-15

### Purpose

Rerun 25 m Model A training with a lower learning rate and diagnose why validation metrics seemed flat.

### Parameter Change

| Parameter | Previous | New | Reason |
|---|---:|---:|---|
| model_a_learning_rate | 1e-4 | 5e-5 | Reduce update size after early validation plateau in smoke test. |

### Training Outcome

Training improved normally, and early stopping restored the best checkpoint from epoch 8.

Best checkpoint:

```text
epoch: 8
val_loss: 2.4324e-06
```

The raw Keras validation precision/recall still appeared near `0.50`, but this was later traced to an evaluation shape issue in the diagnostic path rather than a failed model.

### Evaluation Bug Found and Fixed

Bug:

```text
src/data/npz_loader.py returned labels as (H, W), while predictions were (H, W, 1). This caused NumPy broadcasting during diagnostics and inflated positive-pixel counts.
```

Fix:

```text
src/data/npz_loader.py now returns labels as (H, W, 1), matching NPZPatchDataset and training-time label shape.
```

### Corrected 1000-Patch Validation Diagnostic

Validation sample:

```text
files_seen: 1000
usable_patches: 1000
positive_patches: 716
negative_patches: 284
total_pixels: 4096000
total_positive_pixels: 716
positive_pixel_fraction: 0.00017480
invalid_feature_values_before_cleaning: 12288
invalid_label_values_before_cleaning: 0
```

Prediction distribution:

```text
pred_max_min: 0.00269359
pred_max_median: 0.78716451
pred_max_max: 0.99515307
pred_mean_mean: 0.00020524
```

Corrected threshold sweep:

| Threshold | Precision | Recall | FP Rate | Dice | IoU | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.5046 | 1.0000 | 0.00017166 | 0.6707 | 0.5046 | 716 | 703 | 0 |
| 0.10 | 0.7982 | 1.0000 | 0.00004420 | 0.8878 | 0.7982 | 716 | 181 | 0 |
| 0.15 | 0.8404 | 1.0000 | 0.00003321 | 0.9133 | 0.8404 | 716 | 136 | 0 |
| 0.20 | 0.8524 | 1.0000 | 0.00003028 | 0.9203 | 0.8524 | 716 | 124 | 0 |
| 0.25 | 0.8637 | 1.0000 | 0.00002759 | 0.9269 | 0.8637 | 716 | 113 | 0 |
| 0.30 | 0.8742 | 1.0000 | 0.00002515 | 0.9329 | 0.8742 | 716 | 103 | 0 |
| 0.35 | 0.8795 | 0.9986 | 0.00002393 | 0.9353 | 0.8784 | 715 | 98 | 1 |
| 0.40 | 0.8904 | 0.9986 | 0.00002149 | 0.9414 | 0.8893 | 715 | 88 | 1 |
| 0.45 | 0.9074 | 0.9986 | 0.00001783 | 0.9508 | 0.9062 | 715 | 73 | 1 |
| 0.50 | 0.9261 | 0.9972 | 0.00001392 | 0.9603 | 0.9237 | 714 | 57 | 2 |

### Model A Threshold Decision

Selected operating point:

```text
model: models/model_A_25m_spatial_unet.keras
threshold: 0.50
```

Reason:

```text
Threshold 0.50 gives the best Dice/IoU tradeoff in the corrected diagnostic. Moving from 0.45 to 0.50 removes 16 false-positive pixels while adding only 1 false-negative pixel.
```

### Visual Sample Inspection

Visual samples were exported and manually inspected for:

```text
true positives
false negatives
false positives
clean negatives
```

Observed counts from the sample exporter:

```text
true_positive: 12
false_negative: 2
false_positive: 12
clean_negative: 12
```

Visual findings:

- True positives looked spatially aligned; predicted hotspots landed on the one-pixel ground-truth ignition labels.
- False negatives were borderline one-pixel cases, not catastrophic misses.
- One false negative had `pred_max` near the selected threshold, so a lower threshold could recover it.
- False positives were tiny isolated activations rather than large noisy blobs.
- Clean negatives were mostly empty at the binary threshold.

### Current Model A Decision

```text
Default operating point: models/model_A_25m_spatial_unet.keras @ threshold 0.50
Backup recall point: models/model_A_25m_spatial_unet.keras @ threshold 0.45
```

Reason:

```text
Threshold 0.50 is the best default because it has strong precision, strong Dice/IoU, and visually acceptable errors. Threshold 0.45 is kept as a recall-favoring backup because one false negative was close to 0.50.
```

### Next Evaluation Need

Proceed to the first coarse-to-fine pipeline test:

```text
Model B Phase 2 @ threshold 0.40
↓
Model A 25 m spatial refiner @ threshold 0.50
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
