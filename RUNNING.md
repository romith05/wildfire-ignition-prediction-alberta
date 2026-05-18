# Running the Current Wildfire Ignition Pipeline

This guide records the current frozen paired-patch baseline for the wildfire ignition risk project.

## Current Selected Pipeline

```text
1 km patch
↓
Model B gatekeeper
↓
if passed, find matching 25 m patch by filename
↓
Model A spatial refiner
↓
final 25 m prediction summary
```

## Frozen Model Settings

```text
Model B default:
models/model_B_1km_gatekeeper_hardneg_phase2.keras @ threshold 0.30

Model B recall backup:
models/model_B_1km_gatekeeper_phase2.keras @ threshold 0.40

Model A default:
models/model_A_25m_spatial_unet.keras @ threshold 0.50

Model A recall backup:
models/model_A_25m_spatial_unet.keras @ threshold 0.45

Model A operational minimum positive pixels:
1
```

## Why These Settings Are Frozen

The selected Model B checkpoint was retrained with explicit phase-wise hard-negative mixing.

Current Model B default:

```text
models/model_B_1km_gatekeeper_hardneg_phase2.keras @ threshold 0.30
```

This was selected because it reduced false positives and 25 m workload substantially on the negative-heavy paired test while keeping recall near the previous operating range.

The Model A minimum positive pixel rule remains `1` because the current 25 m labels are one-pixel ignition targets. Requiring more than one predicted positive pixel incorrectly suppresses real one-pixel ignition detections.

## Run the Frozen Held-Out Paired Test

```bash
python -m src.inference.run_paired_patch_pipeline \
  --model-b models/model_B_1km_gatekeeper_hardneg_phase2.keras \
  --model-a models/model_A_25m_spatial_unet.keras \
  --patches-1km-dir /mnt/work/wildfire/1km/patches_1km_balanced/test \
  --patches-25m-dir /mnt/work/wildfire/25m/patches_25m_balanced/test \
  --channel-stats-1km /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
  --channel-stats-25m /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
  --model-b-threshold 0.30 \
  --model-a-threshold 0.50 \
  --model-a-min-positive-pixels 1 \
  --output-csv results/paired_patch_pipeline_test_frozen_hardneg_phase2_t030.csv
```

Summarize:

```bash
python -m src.evaluation.summarize_paired_pipeline_results \
  --csv results/paired_patch_pipeline_test_frozen_hardneg_phase2_t030.csv
```

## Frozen Held-Out Paired Test Result

CSV:

```text
results/paired_patch_pipeline_test_frozen_hardneg_phase2_t030.csv
```

Summary:

```text
rows: 7205
completed: 4254
blocked_by_model_b: 2951
Model B passed patches: 4254 / 7205 = 0.5904
Model A ran patches:    4254 / 7205 = 0.5904
Final positive patches: 4220 / 7205 = 0.5857
Rows with 25 m labels:  4254 / 7205
Missing paired 25 m patches: 0
```

Model B gatekeeper result:

| Metric | Value |
|---|---:|
| TP | 3418 |
| FP | 836 |
| TN | 2741 |
| FN | 210 |
| patch precision | 0.8035 |
| patch recall | 0.9421 |
| patch FP rate | 0.2337 |
| patch accuracy | 0.8548 |

Model A patch outcome on Model-B-passed patches:

| Metric | Value |
|---|---:|
| TP | 3409 |
| FP | 811 |
| TN | 25 |
| FN | 9 |
| patch precision | 0.8078 |
| patch recall | 0.9974 |
| patch FP rate | 0.9701 |
| patch accuracy | 0.8072 |

Model A removal of Model B false-positive patches:

```text
Model B false-positive patch candidates: 836
Removed by Model A: 25
Kept as final positives: 811
Patch removal rate: 0.0299
```

## Interpretation

The frozen held-out paired test confirms that the hard-negative Model B generalizes reasonably well in recall:

```text
negative-heavy recall: 0.9400
held-out test recall:  0.9421
```

However, broad false-positive behavior remains a known limitation:

```text
negative-heavy FP rate: 0.0426
held-out test FP rate:  0.2337
```

Model A should still be treated as a pixel-level spatial refiner, not a patch-level false-positive filter. In the held-out test, Model A removed only `25 / 836 = 0.0299` of Model B false-positive patches.

## Important Evaluation Rule

Do not tune thresholds or retrain models based on the held-out test result.

This test set was used as a frozen sanity check of the already-selected pipeline. Using it to tune thresholds would turn it into another validation set and weaken the credibility of the final reported result.

## Known Limitation

The current paired-patch pipeline uses filename matching:

```text
1 km patch filename == 25 m patch filename
```

This is valid for controlled testing only. It is not the final geospatial prototype.

The final prototype must map a flagged 1 km geospatial footprint to the corresponding generated or retrieved 25 m patches, run Model A on those patches, and stitch predictions back into a map.

## Next Development Step

Start the real coarse-to-fine geospatial prototype:

```text
1. Run Model B over 1 km geospatial patches.
2. Recover each passed patch footprint.
3. Generate or retrieve matching 25 m patches inside that footprint.
4. Run Model A on those 25 m patches.
5. Stitch 25 m predictions back into a geospatial output.
```
