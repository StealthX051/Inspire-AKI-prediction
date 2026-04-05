#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_reviewer_missingness_sensitivity.sh [CONFIG_PATH] [SENSITIVITY_ARTIFACTS_DIR] [OUT_DIR]

Runs a fresh reviewer baseline rerun for the combined xgb model and then launches
the reviewer-specific missingness sensitivity analysis.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

CONFIG_PATH="${1:-$REPO_ROOT/configs/aki/reviewer_combined_xgb_baseline.yaml}"
SENSITIVITY_ARTIFACTS_DIR="${2:-/media/volume/ncs_inspire_data/ncs_aki/artifacts/reviewer_combined_xgb_baseline_median_plus_indicator_gt10}"
OUT_DIR="${3:-$REPO_ROOT/reports}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Reviewer baseline config not found: $CONFIG_PATH" >&2
  exit 1
fi

resolve_python_bin() {
  if [[ -n "${INSPIRE_AKI_PYTHON_BIN:-}" ]]; then
    echo "$INSPIRE_AKI_PYTHON_BIN"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    echo "$REPO_ROOT/.venv/bin/python"
    return 0
  fi
  return 1
}

resolve_cli_bin() {
  if [[ -n "${INSPIRE_AKI_CLI_BIN:-}" ]]; then
    echo "$INSPIRE_AKI_CLI_BIN"
    return 0
  fi
  if command -v inspire-aki >/dev/null 2>&1; then
    command -v inspire-aki
    return 0
  fi
  if [[ -x "$REPO_ROOT/.venv/bin/inspire-aki" ]]; then
    echo "$REPO_ROOT/.venv/bin/inspire-aki"
    return 0
  fi
  return 1
}

PYTHON_BIN="$(resolve_python_bin || true)"
CLI_BIN="$(resolve_cli_bin || true)"

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find a usable python interpreter. Activate your project environment or set INSPIRE_AKI_PYTHON_BIN." >&2
  exit 1
fi

run_cli() {
  if [[ -n "$CLI_BIN" ]]; then
    "$CLI_BIN" "$@"
  else
    PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -c 'from inspire_aki.cli import app; app()' "$@"
  fi
}

echo "Running reviewer baseline combined/xgb rerun with config: $CONFIG_PATH"
run_cli preprocess preop --config "$CONFIG_PATH"
run_cli preprocess intraop --config "$CONFIG_PATH"
run_cli preprocess tabular --config "$CONFIG_PATH"
run_cli preprocess labels --config "$CONFIG_PATH"
INSPIRE_AKI_DATASET_REGIMES=combined run_cli evaluate generate --config "$CONFIG_PATH"
INSPIRE_AKI_DATASET_REGIMES=combined INSPIRE_AKI_MODEL_KEYS=xgb run_cli train tabular --config "$CONFIG_PATH"
run_cli evaluate calibrate --config "$CONFIG_PATH"
run_cli evaluate metrics --config "$CONFIG_PATH"
run_cli explain shap --config "$CONFIG_PATH"

echo "Running reviewer sensitivity analysis into: $SENSITIVITY_ARTIFACTS_DIR"
PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" "$REPO_ROOT/scripts/combined_xgb_missingness_sensitivity.py" \
  --config "$CONFIG_PATH" \
  --sensitivity-artifacts-dir "$SENSITIVITY_ARTIFACTS_DIR" \
  --out-dir "$OUT_DIR"

echo
echo "Reviewer missingness sensitivity workflow finished."
echo "Baseline artifacts: $(PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -c 'from inspire_aki.config import load_config; import sys; print(load_config(sys.argv[1])["paths"]["artifacts_dir"])' "$CONFIG_PATH")"
echo "Sensitivity artifacts: $SENSITIVITY_ARTIFACTS_DIR"
echo "Reviewer summary: $OUT_DIR/missingness_sensitivity_summary.md"
