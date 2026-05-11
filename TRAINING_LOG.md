# Training Log

This file tracks Model B gatekeeper experiments for the wildfire ignition project.

## Goal

Model B is the 1 km patch gatekeeper. It should keep ignition recall high while reducing the number of coarse patches sent to 25 m Model A refinement.

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

### Takeaways

1. Phase 1 worked as a sensitivity warm start.
2. Phase 2 gave the cleanest patch false-positive reduction.
3. Phase 3 appears too aggressive with `lambda_patch=0.50`.
4. Phase 4 recovered recall but did not recover precision.
5. Do not automatically choose the final model.
6. Shortlist Phase 2 and Phase 4 for threshold sweep.

### Candidate Models To Evaluate

```text
models/model_B_1km_gatekeeper_phase2.keras
models/model_B_1km_gatekeeper_phase4.keras
```

### Recommended Next Evaluation

Run a threshold sweep before retraining.

Thresholds:

```text
0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
```

Track:

```text
patch recall
patch false positive rate
patch precision
number of patches passed
percentage of coarse grid passed to 25 m refinement
```

### Recommended Next Tuning If Retraining

Try gentler suppression:

```text
phase2_lambda_patch = 0.10
phase3_lambda_patch = 0.25
phase4_lambda_patch = 0.15
phase1_lr = 1e-4
phase2_lr = 5e-5
phase3_lr = 3e-5
phase4_lr = 2e-5
```

Expected effect:

```text
Phase 3 should become less noisy. Precision should degrade less severely. Recall should remain more stable.
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

### Results

| Phase | val_patch_fp_rate_05 | val_precision | val_recall | val_loss | Notes |
|---|---:|---:|---:|---:|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |

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
```

### Next Planned Change

Write the next tuning adjustment and why.
