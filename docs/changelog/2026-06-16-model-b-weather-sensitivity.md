# 2026-06-16 - Model B weather sensitivity diagnostic

## Change

Added constant-weather support to `src/inference/run_live_weather_model_b_scan.py`.

New CLI arguments:

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

## Diagnostic results

Three Model B full-Alberta scenarios were compared:

| Scenario | Passed coarse patches | Candidate 1 km cells |
|---|---:|---:|
| Constant mean training weather | 176 | 269 |
| Live June 16 weather | 325 | 495 |
| Constant severe fire-weather setting | 520 | 571 |

## Interpretation

Model B is not ignoring weather. It has a static susceptibility backbone from landscape/geographic features, but weather strongly modulates how many candidate cells are activated.

Current interpretation:

```text
Model B = static susceptibility backbone + weather modulation
```

Use the wording `weather-conditioned ignition susceptibility gatekeeper` or `dynamic ignition risk screening model` rather than describing Model B as a pure short-term ignition probability model.

## Related documentation

See:

```text
docs/model_b_weather_sensitivity_diagnostic.md
```
