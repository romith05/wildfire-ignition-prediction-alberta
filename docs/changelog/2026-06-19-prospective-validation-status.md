# 2026-06-19 Prospective Validation Status

## Summary

The automated three-hour prospective-validation loop is running and producing validation evidence against newly observed Alberta active fires.

The system is now operating as a live, weather-conditioned regional ignition-risk screening workflow:

1. A prediction snapshot is generated before future fire observations.
2. The active-fire feed is polled on later cycles.
3. Newly observed fires are matched only against prediction snapshots that existed before the reported fire start time.
4. Validation distances and hit indicators are appended to `results/validation/prospective_validation_log.csv`.

## Current summary output

As of the latest checked summary on 2026-06-19:

```text
Prospective validation summary
========================================
Latest prediction run: live_weather_province_current_20260619_211701
Validation log: results/validation/prospective_validation_log.csv
Validated fires: 5
Lead time median: 14.24 hours
Lead time range: 0.77 to 139.80 hours
Model B candidate nearest distance median: 16.07 km
Model A candidate nearest distance median: 16.07 km
Model A positive nearest distance median: 16.07 km

Hit rates
----------------------------------------
Model B candidate hit <=  1 km: 0 / 5 = 0.0%
Model B candidate hit <=  5 km: 0 / 5 = 0.0%
Model B candidate hit <= 10 km: 0 / 5 = 0.0%
Model B candidate hit <= 25 km: 4 / 5 = 80.0%
Model A candidate hit <=  1 km: 0 / 5 = 0.0%
Model A candidate hit <=  5 km: 0 / 5 = 0.0%
Model A candidate hit <= 10 km: 0 / 5 = 0.0%
Model A candidate hit <= 25 km: 4 / 5 = 80.0%
Model A positive hit <=  1 km: 0 / 5 = 0.0%
Model A positive hit <=  5 km: 0 / 5 = 0.0%
Model A positive hit <= 10 km: 0 / 5 = 0.0%
Model A positive hit <= 25 km: 4 / 5 = 80.0%
```

## Interpretation

This is an early prospective-validation result, not a final performance estimate.

The result supports the system as a regional early-warning or ignition-risk screening tool. It does not yet support claims of exact ignition-point localization.

Current evidence:

- The model has identified broad ignition-prone regions before several new fires were observed.
- Four of five post-baseline fires were within 25 km of a prior predicted candidate area.
- No validated fires were within 1 km, 5 km, or 10 km of the prior candidate areas yet.
- Model B candidate, Model A candidate, and Model A positive hit rates are identical at this stage, which means Model A is not yet materially reducing false positives at the prospective-validation level.

## Project objective alignment

Initial objective:

> Build a system that can predict or identify areas in Alberta where wildfire ignition is likely, using geospatial, environmental, and weather data.

Current status:

- Achieved: automated province-wide prediction snapshots.
- Achieved: live-weather-conditioned inference.
- Achieved: two-stage Model B plus Model A operational workflow.
- Achieved: prospective validation against fires observed after prediction snapshots.
- Early evidence: regional predictive skill at the 25 km screening scale.
- Not yet achieved: reliable exact ignition-cell prediction at 1 km, 5 km, or 10 km.
- Not yet achieved: large-sample statistical validation.
- Not yet achieved: calibrated operational probability interpretation.

Recommended wording:

> The project has progressed from model development to an automated prospective-validation system. Early results suggest the model is beginning to meet the original objective as a regional wildfire ignition-risk screening tool, but it should not yet be presented as an exact ignition-point locator.

## Monitoring command

Run:

```bash
python -m src.validation.summarize_prospective_validation
```

This command summarizes the evolving validation log and should be used after cron cycles to monitor the result as the sample size increases.
