#!/usr/bin/env bash

# Install or replace a user crontab entry for the three-hour wildfire
# prospective-validation cycle.
#
# Usage from the repository root:
#   PYTHON_BIN=/absolute/path/to/python bash scripts/install_three_hour_validation_cron.sh
#
# Optional environment variables:
#   SCHEDULE="17 */3 * * *"     # default: run every 3 hours at minute 17
#   PYTHON_BIN="python"         # default: current `python` found at install time
#   OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS="1.0"
#   MODEL_B_THRESHOLD="0.30"
#   MODEL_B_CANDIDATE_THRESHOLD="0.30"
#   MODEL_A_THRESHOLD="0.50"
#   MODEL_A_MIN_POSITIVE_PIXELS="1"
#
# The installed cron command appends scheduler-level output to:
#   results/validation/cron/three_hour_validation_cron.log
#
# Per-cycle detailed logs are still written by scripts/run_three_hour_validation_cycle.sh
# under:
#   results/validation/cycles/

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SCHEDULE="${SCHEDULE:-17 */3 * * *}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS="${OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS:-1.0}"
MODEL_B_THRESHOLD="${MODEL_B_THRESHOLD:-0.30}"
MODEL_B_CANDIDATE_THRESHOLD="${MODEL_B_CANDIDATE_THRESHOLD:-0.30}"
MODEL_A_THRESHOLD="${MODEL_A_THRESHOLD:-0.50}"
MODEL_A_MIN_POSITIVE_PIXELS="${MODEL_A_MIN_POSITIVE_PIXELS:-1}"
CRON_LOG="${CRON_LOG:-${REPO_ROOT}/results/validation/cron/three_hour_validation_cron.log}"

mkdir -p "$(dirname "${CRON_LOG}")"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the Python executable from the project environment." >&2
  exit 1
fi

if ! command -v crontab >/dev/null 2>&1; then
  echo "crontab command not found. Install cron or use the documented manual command instead." >&2
  exit 1
fi

BEGIN_MARKER="# wildfire_three_hour_validation_cycle:start"
END_MARKER="# wildfire_three_hour_validation_cycle:end"

CRON_COMMAND="cd ${REPO_ROOT} && PYTHON_BIN=${PYTHON_BIN} OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS=${OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS} MODEL_B_THRESHOLD=${MODEL_B_THRESHOLD} MODEL_B_CANDIDATE_THRESHOLD=${MODEL_B_CANDIDATE_THRESHOLD} MODEL_A_THRESHOLD=${MODEL_A_THRESHOLD} MODEL_A_MIN_POSITIVE_PIXELS=${MODEL_A_MIN_POSITIVE_PIXELS} bash scripts/run_three_hour_validation_cycle.sh >> ${CRON_LOG} 2>&1"

TMP_CRON="$(mktemp)"
trap 'rm -f "${TMP_CRON}"' EXIT

# Preserve existing crontab, removing any previously installed block from this script.
crontab -l 2>/dev/null \
  | awk -v begin="${BEGIN_MARKER}" -v end="${END_MARKER}" '
      $0 == begin {skip=1; next}
      $0 == end {skip=0; next}
      skip != 1 {print}
    ' > "${TMP_CRON}" || true

{
  cat "${TMP_CRON}"
  echo "${BEGIN_MARKER}"
  echo "${SCHEDULE} ${CRON_COMMAND}"
  echo "${END_MARKER}"
} | crontab -

echo "Installed wildfire prospective-validation cron entry."
echo "Schedule: ${SCHEDULE}"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Cron log: ${CRON_LOG}"
echo "Detailed cycle logs: ${REPO_ROOT}/results/validation/cycles"
