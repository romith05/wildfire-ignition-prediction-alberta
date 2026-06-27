#!/usr/bin/env bash
set -euo pipefail

# Export and publish the public dashboard bundle into a sibling public repo.
#
# Expected directory layout:
#   /mnt/work/wildfire/25m/
#     wildfire-ignition-prediction-alberta/   # private/main repo
#     wildfire-dashboard-public/              # public dashboard repo
#
# Usage from the private/main repo:
#   bash scripts/publish_public_dashboard_bundle.sh
#
# Optional environment variables:
#   PUBLIC_DASHBOARD_REPO_DIR=/path/to/wildfire-dashboard-public
#   PUBLIC_DASHBOARD_BRANCH=main
#   PYTHON_BIN=/path/to/python
#   SKIP_PUBLIC_DASHBOARD_PUSH=1

MAIN_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$MAIN_REPO_DIR")"
PUBLIC_DASHBOARD_REPO_DIR="${PUBLIC_DASHBOARD_REPO_DIR:-$PARENT_DIR/wildfire-dashboard-public}"
PUBLIC_DASHBOARD_BRANCH="${PUBLIC_DASHBOARD_BRANCH:-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="$PUBLIC_DASHBOARD_REPO_DIR/public_dashboard_bundle"

log() {
  printf '[public-dashboard] %s\n' "$*"
}

fail() {
  printf '[public-dashboard] ERROR: %s\n' "$*" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || fail "git is not available"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"

[[ -d "$MAIN_REPO_DIR/.git" ]] || fail "Main repo .git directory not found: $MAIN_REPO_DIR"
[[ -d "$PUBLIC_DASHBOARD_REPO_DIR/.git" ]] || fail "Public dashboard repo is not a git repo: $PUBLIC_DASHBOARD_REPO_DIR"
[[ -f "$PUBLIC_DASHBOARD_REPO_DIR/app.py" ]] || fail "Public dashboard app.py not found: $PUBLIC_DASHBOARD_REPO_DIR/app.py"

log "Main repo: $MAIN_REPO_DIR"
log "Public repo: $PUBLIC_DASHBOARD_REPO_DIR"
log "Export output: $OUTPUT_DIR"

cd "$MAIN_REPO_DIR"

log "Exporting public-safe dashboard bundle"
"$PYTHON_BIN" scripts/export_public_dashboard_bundle.py --output-dir "$OUTPUT_DIR"

log "Checking exported bundle for internal paths"
if grep -R -n -E '/mnt/|/home/|npz_path' "$OUTPUT_DIR" >/tmp/public_dashboard_leak_check.txt 2>/dev/null; then
  cat /tmp/public_dashboard_leak_check.txt >&2
  fail "Exported bundle contains internal paths or npz_path. Refusing to publish."
fi
rm -f /tmp/public_dashboard_leak_check.txt

cd "$PUBLIC_DASHBOARD_REPO_DIR"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "$PUBLIC_DASHBOARD_BRANCH" ]]; then
  log "Switching public repo branch: $PUBLIC_DASHBOARD_BRANCH"
  git checkout "$PUBLIC_DASHBOARD_BRANCH"
fi

log "Staging public dashboard changes"
git add app.py requirements.txt public_dashboard_bundle

if git diff --cached --quiet; then
  log "No public dashboard changes to commit"
  exit 0
fi

run_id="unknown"
if [[ -f public_dashboard_bundle/latest_prediction_run.txt ]]; then
  run_id="$(tr -d '\n\r' < public_dashboard_bundle/latest_prediction_run.txt)"
fi

generated_at="$(date -u +%Y%m%dT%H%M%SZ)"
commit_message="Update public dashboard bundle ${run_id} ${generated_at}"

log "Committing: $commit_message"
git commit -m "$commit_message"

if [[ "${SKIP_PUBLIC_DASHBOARD_PUSH:-0}" == "1" ]]; then
  log "SKIP_PUBLIC_DASHBOARD_PUSH=1 set; not pushing"
else
  log "Pushing public dashboard repo"
  git push origin "$PUBLIC_DASHBOARD_BRANCH"
fi

log "Done"
