# Model B Weather Sensitivity Diagnostic

Date: 2026-06-16
Branch: `feature/1km-model-b-gatekeeper`
Model: `models/model_B_1km_gatekeeper_hardneg_phase2.keras`
Runner: `src/inference/run_live_weather_model_b_scan.py`

## Purpose

This diagnostic was run after full-province live inference produced a high number of Model B candidate cells. The goal was to determine whether Model B was ignoring live weather, responding to weather in an unexpected direction, or behaving as a weather-conditioned ignition susceptibility gatekeeper.

## Model B feature context

Model B is the 1 km coarse gatekeeper model. It uses 32 x 32 patches with 17 channels:

```text
DEM_1km
cos_month
distance_to_road_1km
landcover_1km
municipalities_multiband_band1
municipalities_multiband_band2
municipalities_multiband_band3
municipalities_multiband_band4
municipalities_multiband_band5
municipalities_multiband_band6
municipalities_multiband_band7
municipalities_multiband_band8
relative_humidity
sin_month
temperature
water_1km
wind_speed
```

The weather channels are injected into generated NPZ patches before Model B scoring.

## Training weather channel statistics

From the 1 km balanced patch channel statistics:

| Channel | Mean | Standard deviation |
|---|---:|---:|
| `relative_humidity` | 44.66 | 20.10 |
| `temperature` | 17.03 | 8.05 |
| `wind_speed` | 8.68 | 7.17 |

These values are in physical units. The diagnostic did not suggest a simple weather unit mismatch.

## Live June 16 run summary

Run ID:

```text
live_weather_province_current_20260616_224141
```

Model B score distribution:

```text
count    692.000000
mean       0.456319
std        0.412920
min        0.000000
25%        0.000331
50%        0.414307
75%        0.914556
max        0.999807
```

Threshold sensitivity:

| Threshold | Coarse patches >= threshold |
|---:|---:|
| 0.3 | 378 |
| 0.4 | 347 |
| 0.5 | 325 |
| 0.6 | 307 |
| 0.7 | 273 |
| 0.8 | 230 |
| 0.9 | 182 |

Full pipeline output for this run:

```text
Model B score rows: 692
Model B failed rows: 0
Model B candidate cells: 495
Model A prediction rows: 495
Model A failed rows: 0
Model A final-positive cells: 475
Model A positive rate: 95.96%
```

## Live weather summary and correlation

Weather was extracted from the `metadata_json` column of `model_b_manifest.csv`.

```text
       temperature  relative_humidity  wind_speed
count   692.000000         692.000000  692.000000
mean     16.707081          58.260116   18.000145
std       3.655239          20.346107    9.547734
min       0.600000          21.000000    1.400000
25%      14.775000          41.000000   11.200000
50%      16.800000          57.000000   15.900000
75%      19.300000          75.000000   23.300000
max      26.700000          99.000000   59.400000
```

Correlation with Model B maximum patch probability:

```text
                   model_b_max_prob  temperature  relative_humidity  wind_speed
model_b_max_prob           1.000000    -0.052459           0.276284   -0.430381
temperature               -0.052459     1.000000          -0.546711    0.242922
relative_humidity          0.276284    -0.546711           1.000000   -0.216020
wind_speed                -0.430381     0.242922          -0.216020    1.000000
```

At first glance, this looked counterintuitive. Later constant-weather ablation showed the correlation was likely confounded by geography and static susceptibility.

## Top-risk vs low-risk patch comparison

Top 30 Model B patches:

```text
DEM_1km                  595.414011
distance_to_road_1km    9582.248254
landcover_1km            168.684408
water_1km                  1.952344
temperature               15.643333
relative_humidity         85.666667
wind_speed                 8.226667
prob                       0.998601
```

Low 30 Model B patches:

```text
DEM_1km                 883.6298
distance_to_road_1km    537.6612
landcover_1km             0.6986
water_1km                 1.9675
temperature              20.6333
relative_humidity        51.1667
wind_speed               34.0933
prob                      0.0000000863
```

Difference, top minus low:

```text
DEM_1km                 -288.215744
distance_to_road_1km    9044.587051
landcover_1km            167.985840
water_1km                 -0.015169
temperature               -4.990000
relative_humidity         34.500000
wind_speed               -25.866667
prob                       0.998601
```

This showed that top-risk patches were much farther from roads and had very different landcover values. Spatial susceptibility was therefore a major confounder in the simple weather correlation table.

## Constant-weather mode added

The Model B runner was updated to support:

```text
--weather-mode api|constant
--constant-temperature
--constant-relative-humidity
--constant-wind-speed
```

Implementation commit:

```text
01d4cf09d80b072de729849cd7a919d73425214b
```

## Constant-weather ablation results

| Scenario | Passed coarse patches | Coarse pass rate | Candidate 1 km cells |
|---|---:|---:|---:|
| Constant mean training weather | 176 | 0.2543 | 269 |
| Live June 16 weather | 325 | not recorded here | 495 |
| Constant severe fire-weather setting | 520 | 0.7514 | 571 |

Mean training weather run:

```text
constant_temperature = 17.03
constant_relative_humidity = 44.66
constant_wind_speed = 8.68
```

Output:

```text
Coarse patch rows: 692
Completed: 692
Failed: 0
Passed coarse patches: 176
Coarse pass rate: 0.2543
Candidate 1 km cells: 269
```

Severe fire-weather constant run:

```text
constant_temperature = 28
constant_relative_humidity = 20
constant_wind_speed = 35
```

Output:

```text
Coarse patch rows: 692
Completed: 692
Failed: 0
Passed coarse patches: 520
Coarse pass rate: 0.7514
Candidate 1 km cells: 571
```

## Conclusion

Model B is not ignoring weather. The ablation showed a strong response to weather severity:

```text
mean training weather -> 269 candidate cells
live June 16 weather  -> 495 candidate cells
severe fire weather   -> 571 candidate cells
```

The current interpretation is:

```text
Model B = static susceptibility backbone + weather modulation
```

Static layers such as landcover, road distance, and geographic context create a baseline susceptibility surface. Weather variables then expand or contract the active candidate set under current or hypothetical conditions.

The model should be described as a weather-conditioned ignition susceptibility gatekeeper or dynamic ignition risk screening model, not as a pure short-term ignition probability model.

## Next validation question

The remaining quality question is:

```text
Are the weather-conditioned high-risk cells predictive of future active fires?
```

This is the role of the prospective validation loop using newly detected active fires after a prediction snapshot is created.
