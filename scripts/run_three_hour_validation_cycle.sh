#!/usr/bin/env bash

# Run one prospective wildfire-validation cycle.
#
# Order matters:
#   1. Download the current active-fire feed.
#   2. Validate newly observed fires against prediction snapshots that existed
#      before each fire's reported start time.
#   3. Generate a fresh full-province prediction snapshot for the next cycle.
#   4. Optionally publish a public-safe dashboard bundle.
#
# Safe poll-only smoke test:
#   POLL_ONLY=1 bash scripts/run_three_hour_validation_cycle.sh
#
# Full manual cycle:
#   bash scripts/run_three_hour_validation_cycle.sh
#
# Full manual cycle with public dashboard publish:
#   PUBLISH_PUBLIC_DASHBOARD=1 bash scripts/run_three_hour_validation_cycle.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_ONLY="${POLL_ONLY:-0}"
PUBLISH_PUBLIC_DASHBOARD="${PUBLISH_PUBLIC_DASHBOARD:-0}"
PUBLISH_PUBLIC_DASHBOARD_FAILS_CYCLE="${PUBLISH_PUBLIC_DASHBOARD_FAILS_CYCLE:-0}"
OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS="${OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS:-1.0}"
export OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS

ACTIVE_GEOJSON="${ACTIVE_GEOJSON:-data/validation/alberta_activefires_current.geojson}"
ACTIVE_CSV="${ACTIVE_CSV:-data/validation/alberta_activefires_current.csv}"
STATE_JSON="${STATE_JSON:-data/validation/state/seen_active_fires.json}"
PREDICTION_REGISTRY="${PREDICTION_REGISTRY:-data/validation/state/prediction_runs.csv}"
VALIDATION_LOG="${VALIDATION_LOG:-results/validation/prospective_validation_log.csv}"
RUNS_ROOT="${RUNS_ROOT:-results/runs}"

GRID="${GRID:-data/grids/alberta_coarse_grid_epsg3979.geojson}"
MODEL_B_FEATURE_CONFIG="${MODEL_B_FEATURE_CONFIG:-configs/model_b_1km_features.json}"
MODEL_A_FEATURE_CONFIG="${MODEL_A_FEATURE_CONFIG:-configs/model_a_25m_features.json}"
MODEL_B="${MODEL_B:-models/model_B_1km_gatekeeper_hardneg_phase2.keras}"
MODEL_A="${MODEL_A:-models/model_A_25m_spatial_unet.keras}"
MODEL_B_STATS="${MODEL_B_STATS:-/mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json}"
MODEL_A_STATS="${MODEL_A_STATS:-/mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json}"

MODEL_B_THRESHOLD="${MODEL_B_THRESHOLD:-0.30}"
MODEL_B_CANDIDATE_THRESHOLD="${MODEL_B_CANDIDATE_THRESHOLD:-0.30}"
MODEL_A_THRESHOLD="${MODEL_A_THRESHOLD:-0.50}"
MODEL_A_MIN_POSITIVE_PIXELS="${MODEL_A_MIN_POSITIVE_PIXELS:-1}"
SEED="${SEED:-42}"

CYCLE_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-live_weather_province_current_${CYCLE_TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-results/validation/cycles}"
LOG_FILE="${LOG_DIR}/${CYCLE_TIMESTAMP}.log"
LOCK_FILE="${LOCK_FILE:-/tmp/wildfire_three_hour_validation_cycle.lock}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-results/validation/latest_prediction_run.txt}"

mkdir -p "${LOG_DIR}" "$(dirname "${STATE_JSON}")" "$(dirname "${VALIDATION_LOG}")"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "Another validation cycle is already running; exiting."
    exit 0
  fi
else
  echo "Warning: flock is unavailable; overlapping cycles are not automatically prevented."
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

on_error() {
  local exit_code=$?
  local line_number="${1:-unknown}"
  echo "Cycle failed at line ${line_number} with exit code ${exit_code}."
  echo "Log: ${LOG_FILE}"
  exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}"
    exit 1
  fi
}

publish_public_dashboard() {
  if [[ "${PUBLISH_PUBLIC_DASHBOARD}" != "1" ]]; then
    echo "Public dashboard publish skipped. Set PUBLISH_PUBLIC_DASHBOARD=1 to enable."
    return 0
  fi

  local publish_script="scripts/publish_public_dashboard_bundle.sh"
  if [[ ! -f "${publish_script}" ]]; then
    echo "Warning: public dashboard publish script not found: ${publish_script}"
    if [[ "${PUBLISH_PUBLIC_DASHBOARD_FAILS_CYCLE}" == "1" ]]; then
      exit 1
    fi
    return 0
  fi

  echo "Publishing public dashboard bundle..."
  if bash "${publish_script}"; then
    echo "Public dashboard publish completed."
  else
    local publish_exit_code=$?
    echo "Warning: public dashboard publish failed with exit code ${publish_exit_code}."
    if [[ "${PUBLISH_PUBLIC_DASHBOARD_FAILS_CYCLE}" == "1" ]]; then
      exit "${publish_exit_code}"
    fi
  fi
}

echo "============================================================"
echo "Wildfire prospective validation cycle"
echo "Cycle UTC: ${CYCLE_TIMESTAMP}"
echo "Repository: ${REPO_ROOT}"
echo "Poll only: ${POLL_ONLY}"
echo "Publish public dashboard: ${PUBLISH_PUBLIC_DASHBOARD}"
echo "Open-Meteo request interval: ${OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS} seconds"
echo "Log: ${LOG_FILE}"
echo "============================================================"

require_file "${STATE_JSON}"
require_file "${GRID}"
require_file "${MODEL_B_FEATURE_CONFIG}"
require_file "${MODEL_A_FEATURE_CONFIG}"
require_file "${MODEL_B}"
require_file "${MODEL_A}"
require_file "${MODEL_B_STATS}"
require_file "${MODEL_A_STATS}"

# Step 1: refresh the current active-fire snapshot.
"${PYTHON_BIN}" -m src.validation.fetch_alberta_active_fires \
  --output-geojson "${ACTIVE_GEOJSON}" \
  --output-csv "${ACTIVE_CSV}"

# Step 2: validate only newly observed Alberta incidents. The state file must
# already have been initialized manually; this script never initializes it.
"${PYTHON_BIN}" -m src.validation.poll_active_fires_and_validate \
  --active-fires-csv "${ACTIVE_CSV}" \
  --runs-root "${RUNS_ROOT}" \
  --state-json "${STATE_JSON}" \
  --prediction-registry-csv "${PREDICTION_REGISTRY}" \
  --validation-log-csv "${VALIDATION_LOG}" \
  --agency AB \
  --fire-id-column Fire_Name \
  --start-date-column Start_Date \
  --start-date-unit ms \
  --distance-thresholds-m 1000,5000,10000,25000

if [[ "${POLL_ONLY}" == "1" ]]; then
  echo "Poll-only cycle completed. No prediction run was started."
  publish_public_dashboard
  exit 0
fi

# Step 3: create the next full-province prediction snapshot.
if [[ -e "${RUNS_ROOT}/${RUN_ID}" || -e "data/runs/${RUN_ID}" ]]; then
  echo "Run ID already exists: ${RUN_ID}"
  exit 1
fi

echo "Starting full-province prediction run: ${RUN_ID}"

"${PYTHON_BIN}" -m src.inference.run_live_weather_model_b_scan \
  --grid "${GRID}" \
  --feature-config "${MODEL_B_FEATURE_CONFIG}" \
  --model "${MODEL_B}" \
  --channel-stats "${MODEL_B_STATS}" \
  --all \
  --run-id "${RUN_ID}" \
  --threshold "${MODEL_B_THRESHOLD}" \
  --candidate-threshold "${MODEL_B_CANDIDATE_THRESHOLD}" \
  --batch-size 128 \
  --seed "${SEED}" \
  --overwrite

MODEL_B_CANDIDATES="${RUNS_ROOT}/${RUN_ID}/model_b_candidates.csv"
MODEL_B_MANIFEST="${RUNS_ROOT}/${RUN_ID}/model_b_manifest.csv"
require_file "${MODEL_B_CANDIDATES}"
require_file "${MODEL_B_MANIFEST}"

CANDIDATE_COUNT="$("${PYTHON_BIN}" - "${MODEL_B_CANDIDATES}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
print(len(df))
PY
)"

echo "Model B candidate cells: ${CANDIDATE_COUNT}"
if [[ "${CANDIDATE_COUNT}" -le 0 ]]; then
  echo "Model B produced no candidates. The run cannot currently be registered as a complete two-stage snapshot."
  exit 1
fi

"${PYTHON_BIN}" -m src.inference.run_live_weather_model_a_candidates \
  --candidates "${MODEL_B_CANDIDATES}" \
  --model-b-manifest "${MODEL_B_MANIFEST}" \
  --weather-mode model-b-manifest \
  --feature-config "${MODEL_A_FEATURE_CONFIG}" \
  --model "${MODEL_A}" \
  --channel-stats "${MODEL_A_STATS}" \
  --run-id "${RUN_ID}" \
  --threshold "${MODEL_A_THRESHOLD}" \
  --min-positive-pixels "${MODEL_A_MIN_POSITIVE_PIXELS}" \
  --batch-size 16 \
  --seed "${SEED}" \
  --overwrite

MODEL_A_PREDICTIONS="${RUNS_ROOT}/${RUN_ID}/model_a_predictions.csv"
MODEL_A_GEOJSON="${RUNS_ROOT}/${RUN_ID}/model_a_candidate_cells.geojson"
MODEL_A_SUMMARY="${RUNS_ROOT}/${RUN_ID}/model_a_candidate_cells_summary.json"
require_file "${MODEL_A_PREDICTIONS}"

"${PYTHON_BIN}" -m src.geospatial.export_model_a_cells_geojson \
  --predictions-csv "${MODEL_A_PREDICTIONS}" \
  --output-geojson "${MODEL_A_GEOJSON}" \
  --summary-json "${MODEL_A_SUMMARY}"

require_file "${MODEL_A_GEOJSON}"
require_file "${MODEL_A_SUMMARY}"

# Fail the cycle if either model contains failed records or the final GeoJSON
# does not contain one feature per Model A prediction row.
"${PYTHON_BIN}" - \
  "${RUNS_ROOT}/${RUN_ID}/model_b_scores.csv" \
  "${MODEL_B_CANDIDATES}" \
  "${MODEL_A_PREDICTIONS}" \
  "${MODEL_A_GEOJSON}" <<'PY'
import sys
import geopandas as gpd
import pandas as pd

scores_path, candidates_path, predictions_path, geojson_path = sys.argv[1:]
scores = pd.read_csv(scores_path)
candidates = pd.read_csv(candidates_path)
predictions = pd.read_csv(predictions_path)
geojson = gpd.read_file(geojson_path)

model_b_failed = int((scores["status"] != "completed").sum())
model_a_failed = int((predictions["status"] != "completed").sum())
final_positive = int(
    pd.to_numeric(predictions["final_positive"], errors="coerce").fillna(0).astype(int).sum()
)

print("Final cycle validation")
print(f"Model B score rows: {len(scores)}")
print(f"Model B failed rows: {model_b_failed}")
print(f"Model B candidate cells: {len(candidates)}")
print(f"Model A prediction rows: {len(predictions)}")
print(f"Model A failed rows: {model_a_failed}")
print(f"Model A final-positive cells: {final_positive}")
print(f"GeoJSON features: {len(geojson)}")
print(f"GeoJSON CRS: {geojson.crs}")

if model_b_failed:
    raise SystemExit("Model B contains failed rows.")
if model_a_failed:
    raise SystemExit("Model A contains failed rows.")
if len(predictions) != len(candidates):
    raise SystemExit("Model A prediction count does not match Model B candidate count.")
if len(geojson) != len(predictions):
    raise SystemExit("GeoJSON feature count does not match Model A prediction count.")
if str(geojson.crs).upper() != "EPSG:3979":
    raise SystemExit(f"Unexpected GeoJSON CRS: {geojson.crs}")
PY

mkdir -p "$(dirname "${LATEST_RUN_FILE}")"
printf '%s\n' "${RUN_ID}" > "${LATEST_RUN_FILE}"

echo "Prediction snapshot completed: ${RUN_ID}"
echo "Latest-run pointer: ${LATEST_RUN_FILE}"
publish_public_dashboard
echo "Cycle log: ${LOG_FILE}"
