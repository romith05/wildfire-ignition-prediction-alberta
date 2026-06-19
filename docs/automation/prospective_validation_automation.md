# Prospective Validation Automation

This document describes how to automate the wildfire prospective-validation cycle.

## What gets automated

The automation runs:

```text
scripts/run_three_hour_validation_cycle.sh
```

Each cycle does the following:

1. Downloads the latest active-fire feed.
2. Validates newly observed Alberta fires against prediction snapshots that existed before each fire start time.
3. Creates a fresh full-province prediction snapshot for the next cycle.
4. Runs Model B with live weather.
5. Runs Model A on Model B candidate cells.
6. Exports Model A candidate cells to GeoJSON.
7. Validates output counts, failure rows, and CRS.
8. Updates `results/validation/latest_prediction_run.txt`.

The cycle script uses a lock file to prevent overlapping runs:

```text
/tmp/wildfire_three_hour_validation_cycle.lock
```

## Prerequisites

Before enabling cron, confirm the state file has already been initialized:

```text
data/validation/state/seen_active_fires.json
```

The first initialization should be done manually, not by cron:

```bash
python -m src.validation.fetch_alberta_active_fires \
  --output-geojson data/validation/alberta_activefires_current.geojson \
  --output-csv data/validation/alberta_activefires_current.csv

python -m src.validation.poll_active_fires_and_validate \
  --active-fires-csv data/validation/alberta_activefires_current.csv \
  --runs-root results/runs \
  --agency AB \
  --fire-id-column Fire_Name \
  --start-date-column Start_Date \
  --start-date-unit ms \
  --initialize-state
```

Then run a poll-only smoke test:

```bash
POLL_ONLY=1 bash scripts/run_three_hour_validation_cycle.sh
```

Then run one full manual cycle:

```bash
bash scripts/run_three_hour_validation_cycle.sh
```

Enable cron only after the poll-only and manual full-cycle checks work.

## Install the cron job

From the repository root, run:

```bash
git pull origin feature/1km-model-b-gatekeeper
chmod +x scripts/install_three_hour_validation_cron.sh
PYTHON_BIN="$(which python)" bash scripts/install_three_hour_validation_cron.sh
```

If the project uses a specific virtual environment, pass that Python path explicitly:

```bash
PYTHON_BIN=/home/bondada.romith/wildfire/jlab/bin/python \
  bash scripts/install_three_hour_validation_cron.sh
```

Default schedule:

```text
17 */3 * * *
```

This runs every three hours at minute 17. The staggered minute avoids running exactly on the hour.

## Change the schedule

Example: run every six hours at minute 22:

```bash
SCHEDULE="22 */6 * * *" \
PYTHON_BIN=/home/bondada.romith/wildfire/jlab/bin/python \
  bash scripts/install_three_hour_validation_cron.sh
```

Example: run daily at 02:30:

```bash
SCHEDULE="30 2 * * *" \
PYTHON_BIN=/home/bondada.romith/wildfire/jlab/bin/python \
  bash scripts/install_three_hour_validation_cron.sh
```

## Logs

Scheduler-level log:

```text
results/validation/cron/three_hour_validation_cron.log
```

Per-cycle detailed logs:

```text
results/validation/cycles/
```

Latest completed prediction run pointer:

```text
results/validation/latest_prediction_run.txt
```

Prospective validation output log:

```text
results/validation/prospective_validation_log.csv
```

Prediction-run registry:

```text
data/validation/state/prediction_runs.csv
```

## Summary command

Use the summary command after any manual or scheduled cycle to inspect the current prospective-validation evidence without reading the CSV manually:

```bash
python -m src.validation.summarize_prospective_validation
```

The command reads:

```text
results/validation/prospective_validation_log.csv
results/validation/latest_prediction_run.txt
```

It prints:

- validated fire count
- lead-time median and range
- nearest-distance medians for Model B candidates, Model A candidates, and Model A positives
- hit rates at 1 km, 5 km, 10 km, and 25 km

The command is read-only and does not modify validation state, prediction outputs, raw data, rasters, NPZ files, or model artifacts.

## Check that cron is installed

```bash
crontab -l | sed -n '/wildfire_three_hour_validation_cycle:start/,/wildfire_three_hour_validation_cycle:end/p'
```

Expected block:

```text
# wildfire_three_hour_validation_cycle:start
17 */3 * * * cd ... && PYTHON_BIN=... bash scripts/run_three_hour_validation_cycle.sh >> .../results/validation/cron/three_hour_validation_cron.log 2>&1
# wildfire_three_hour_validation_cycle:end
```

## Disable the cron job

Remove the marked block manually:

```bash
crontab -e
```

Delete everything between:

```text
# wildfire_three_hour_validation_cycle:start
# wildfire_three_hour_validation_cycle:end
```

## Operational notes

- The cycle script requires an initialized active-fire state file.
- The cycle script should not be run in multiple overlapping schedulers.
- The script uses `flock` when available to prevent overlap.
- Full cycles make Open-Meteo requests for all selected coarse grid patches.
- `OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS` defaults to `1.0` for safer API pacing.
- Use `POLL_ONLY=1` for a cheap health check that does not start Model B or Model A prediction.

## Interpretation of future validation rows

This automation is intended to support true prospective validation. The key evidence is not how many candidate cells are produced in a run, but whether fires that appear after a prediction snapshot fall near previously predicted high-risk cells.

A new fire can only validate a prediction snapshot if the prediction snapshot was created before the fire start time.
