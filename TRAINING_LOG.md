# Training Log

This file tracks Model B gatekeeper and Model A spatial-refiner experiments for the wildfire ignition project.

## Goal

Model B is the 1 km patch gatekeeper. It should keep ignition recall high while reducing the number of coarse patches sent to 25 m Model A refinement.

Model A is the 25 m spatial refiner. It should produce accurate pixel-level ignition probability masks on fine-resolution patches, including patches that are true positives from Model B and false-positive patches that still pass the gate.

Current cascade:

```text
1 km scan -> Model B gatekeeper -> selected coarse patches -> 25 m Model A refiner
```

## Current Operating Decisions

```text
Model B default: models/model_B_1km_gatekeeper_phase2.keras @ threshold 0.40
Model B backup:  models/model_B_1km_gatekeeper_phase4.keras @ threshold 0.45
Model A default: models/model_A_25m_spatial_unet.keras @ threshold 0.50
Model A backup:  models/model_A_25m_spatial_unet.keras @ threshold 0.45
Model A operational min positive pixels: 1
```

Reason for Model A min positive pixels:

```text
The current 25 m labels are one-pixel ignition targets. Requiring more than one predicted positive pixel incorrectly suppresses true ignition patches, so min_pixels > 1 is rejected as an operational rule for now.
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

### Results Summary

| Phase | val_patch_fp_rate_05 | val_precision | val_recall | val_loss | Interpretation |
|---|---:|---:|---:|---:|---|
| 1 | 0.1798 | 0.7381 | 0.9958 | 1.1371e-05 | Excellent recall, patch false positives still high. |
| 2 | 0.0499 | 0.7086 | 0.8966 | 0.0231 | Big patch-FP reduction, but recall dropped. |
| 3 | 0.0639 at best-val-loss epoch | 0.4008 | 0.9483 | 0.1121 | Suppression likely too strong; precision degraded. |
| 4 | 0.0562 final, 0.0485 best checkpoint | 0.3996 final, 0.3989 best checkpoint | 0.9344 final, 0.9092 best checkpoint | 0.0820 final, 0.0784 best checkpoint | Recall recovered, but precision stayed weak. |

### Threshold Decisions

Phase 2 selected operating point:

```text
model: models/model_B_1km_gatekeeper_phase2.keras
threshold: 0.40
```

Phase 2 comparison:

| Threshold | Patch Recall | Patch Precision | Patch FP Rate | Pass Rate | Passed Patches | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.9441 | 0.8848 | 0.1250 | 0.5380 | 764 | 676 | 88 | 616 | 40 |
| 0.45 | 0.9260 | 0.8972 | 0.1080 | 0.5204 | 739 | 663 | 76 | 628 | 53 |

Phase 4 backup operating point:

```text
model: models/model_B_1km_gatekeeper_phase4.keras
threshold: 0.45
```

Phase 4 @ 0.45 is a strong backup but does not clearly beat Phase 2 @ 0.40 for the default setting.

### Takeaways

1. Phase 2 threshold `0.40` is the current default Model B operating point.
2. Phase 4 threshold `0.45` is the backup stricter operating point.
3. Phase 3 appears too aggressive with `lambda_patch=0.50`.
4. Do not automatically choose the final model; evaluate saved phase checkpoints.

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

The run confirmed the training path worked, but the initial validation interpretation was unreliable because a later diagnostic found an evaluation label-shape bug.

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

```text
epoch: 8
val_loss: 2.4324e-06
```

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

```text
model: models/model_A_25m_spatial_unet.keras
threshold: 0.50
```

Reason:

```text
Threshold 0.50 gives the best Dice/IoU tradeoff in the corrected diagnostic. Moving from 0.45 to 0.50 removes 16 false-positive pixels while adding only 1 false-negative pixel.
```

### Visual Sample Inspection

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
- False positives were tiny isolated activations rather than large noisy blobs.
- Clean negatives were mostly empty at the binary threshold.

### Current Model A Decision

```text
Default operating point: models/model_A_25m_spatial_unet.keras @ threshold 0.50
Backup recall point: models/model_A_25m_spatial_unet.keras @ threshold 0.45
```

## Run 004 — Test-Only Paired Patch Pipeline Baseline

Date: 2026-05-18

### Purpose

Run the first end-to-end coarse-to-fine smoke test using filename-paired 1 km and 25 m patches.

This is a controlled test only. It uses matching filenames between the two patch folders instead of real geospatial scaling from 1 km footprints to generated 25 m patches.

### Model Settings

```text
Model B: models/model_B_1km_gatekeeper_phase2.keras @ threshold 0.40
Model A: models/model_A_25m_spatial_unet.keras @ threshold 0.50
Model A min positive pixels: 1
```

### Pipeline Summary

```text
rows: 1000
completed: 701
blocked_by_model_b: 299
Model B passed patches: 701 / 1000 = 0.7010
Model A ran patches: 701 / 1000 = 0.7010
Final positive patches: 699 / 1000 = 0.6990
Rows with 25 m labels: 701 / 1000
Missing paired 25 m patches: 0
```

### Model B Patch-Level Gatekeeper Result

| Metric | Value |
|---|---:|
| TP | 676 |
| FP | 25 |
| TN | 259 |
| FN | 40 |
| patch precision | 0.9643 |
| patch recall | 0.9441 |
| patch FP rate | 0.0880 |
| patch accuracy | 0.9350 |

### Final Cascade Patch-Level Result on Available 25 m Labels

| Metric | Value |
|---|---:|
| TP | 676 |
| FP | 23 |
| TN | 2 |
| FN | 0 |
| patch precision | 0.9671 |
| patch recall | 1.0000 |
| patch FP rate | 0.9200 |
| patch accuracy | 0.9672 |

### Model A Effect on Model-B-Passed Patches

```text
Model B false-positive patch candidates: 25
Removed by Model A: 2
Kept as final positives: 23
Patch removal rate: 0.0800
```

### Decision

```text
Keep this as the first paired-pipeline baseline.
Proceed to a negative-heavy test set before judging Alberta-wide operational false alarms.
```

## Run 005 — Negative-Heavy Paired Pipeline and Min-Pixel Rule Test

Date: 2026-05-18

### Purpose

Stress-test the paired pipeline on a more realistic negative-heavy sample and evaluate whether Model A can reduce patch-level false positives after Model B.

### Test Set

```text
rows: 804
positive patches: 100
negative patches: 704
```

### Model Settings

Default negative-heavy run:

```text
Model B: models/model_B_1km_gatekeeper_phase2.keras @ threshold 0.40
Model A: models/model_A_25m_spatial_unet.keras @ threshold 0.50
Model A min positive pixels: 1
```

### Negative-Heavy Result With Min Pixels = 1

```text
rows: 804
blocked_by_model_b: 620
completed: 184
Model B passed patches: 184 / 804 = 0.2289
Model A ran patches: 184 / 804 = 0.2289
Final positive patches: 181 / 804 = 0.2251
Rows with 25 m labels: 184 / 804
Missing paired 25 m patches: 0
```

Model B patch-level result:

| Metric | Value |
|---|---:|
| TP | 96 |
| FP | 88 |
| TN | 616 |
| FN | 4 |
| patch precision | 0.5217 |
| patch recall | 0.9600 |
| patch FP rate | 0.1250 |
| patch accuracy | 0.8856 |

Model A effect on Model-B-passed false positives:

```text
Model B false-positive patch candidates: 88
Removed by Model A: 3
Kept as final positives: 85
Patch removal rate: 0.0341
```

### Min-Positive-Pixel Experiment

The pipeline was extended with:

```text
--model-a-min-positive-pixels
```

This tested whether requiring more than one predicted positive 25 m pixel would suppress patch-level false positives.

Finding:

```text
min_pixels > 1 is not appropriate as an operational rule with the current labels.
```

Reason:

```text
Many true 25 m ignition patches have exactly one labeled ignition pixel, and Model A often predicts exactly one high-confidence positive pixel for those true positives. Requiring 2 or 3 pixels suppresses real ignition detections.
```

Example pattern observed in the CSV:

```text
label_25m_positive = True
model_a_max_prob is high
model_a_positive_pixels = 1
model_a_min_positive_pixels = 3
final_positive = 0
```

Interpretation:

```text
The min-pixel rule can be useful for analysis, but it should not be used as the production final decision rule unless the label definition changes from one-pixel ignition points to area masks.
```

### Decision

```text
Keep Model A operational min positive pixels = 1.
Reject min_pixels > 1 as the default final-decision rule for current labels.
```

### Main Takeaway

The negative-heavy test shows that Model B is recall-safe and reduces workload substantially, but its false-positive burden is still the main operational issue.

```text
Model B should remain the focus for patch-level false-positive reduction.
Model A should remain a pixel-level spatial refiner, not a second patch gatekeeper.
```

### Next Planned Change

Improve Model B false-positive behavior by collecting and using hard negatives from Model-B-passed no-ignition patches.

Possible next steps:

```text
1. Use the negative-heavy paired pipeline CSV to identify Model B false positives.
2. Add those no-ignition patches to a hard-negative set.
3. Fine-tune Model B with hard negatives or run another phased training pass with gentler suppression.
4. Re-run the negative-heavy pipeline test.
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
