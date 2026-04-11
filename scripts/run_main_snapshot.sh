#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python3}"

TASK1_DATA_ROOT="${TASK1_DATA_ROOT:-}"
INPUT_CSV="${INPUT_CSV:-}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/mainline}"

if [[ -z "$INPUT_CSV" ]]; then
  if [[ -z "$TASK1_DATA_ROOT" ]]; then
    echo "Set INPUT_CSV or TASK1_DATA_ROOT." >&2
    exit 2
  fi
  INPUT_CSV="$TASK1_DATA_ROOT/test.csv"
fi

if [[ -z "$TASK1_DATA_ROOT" ]]; then
  TASK1_DATA_ROOT="$(cd "$(dirname "$INPUT_CSV")" && pwd)"
fi

WORK_DIR="$RUN_ROOT/work"
OUTPUT_DIR="$RUN_ROOT/output"
mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

ABS_CSV="$WORK_DIR/test_abs.csv"
"$PY" "$REPO_ROOT/tools/make_absolute_csv.py" \
  --input "$INPUT_CSV" \
  --task1-root "$TASK1_DATA_ROOT" \
  --output "$ABS_CSV"

"$PY" "$REPO_ROOT/src/postprocess/template_router.py" \
  --input-csv "$ABS_CSV" \
  --output-txt "$OUTPUT_DIR/prediction.txt" \
  --output-json "$OUTPUT_DIR/model.json" \
  --policy-json "$REPO_ROOT/policies/template_router_policy_v2.json" \
  --cache-a "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/cache/route_a.txt" \
  --cache-b "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/cache/route_b.txt" \
  --cache-c "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/cache/route_c.txt" \
  --cache-d "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/cache/route_d.txt" \
  --cache-l "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/cache/route_l.txt" \
  --cache-w "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/cache/route_w.txt" \
  --cache-pub "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/cache/route_pub.txt" \
  --cache-rr2 "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/cache/route_rr2.txt" \
  --cache-tree "$REPO_ROOT/assets/frozen_inputs/mainline/clean_routes/output/prediction_tree.txt" \
  --cache-v9 "$REPO_ROOT/assets/frozen_inputs/mainline/v9_full/output/prediction.txt"

"$PY" "$REPO_ROOT/tools/make_result_zip.py" \
  --prediction "$OUTPUT_DIR/prediction.txt" \
  --model "$OUTPUT_DIR/model.json" \
  --output "$OUTPUT_DIR/result.zip"

printf 'run_root=%s\n' "$RUN_ROOT"
printf 'prediction=%s\n' "$OUTPUT_DIR/prediction.txt"
printf 'model=%s\n' "$OUTPUT_DIR/model.json"
printf 'result_zip=%s\n' "$OUTPUT_DIR/result.zip"
