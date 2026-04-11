#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python3}"
KEY_FILE="$REPO_ROOT/.openai_api_key"

TASK1_DATA_ROOT="${TASK1_DATA_ROOT:-}"
INPUT_CSV="${INPUT_CSV:-}"
RUN_TAG="${RUN_TAG:-live_template_pipeline_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/$RUN_TAG}"

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

if [[ -z "${VQA_API_KEY:-}" && -f "$KEY_FILE" ]]; then
  export VQA_API_KEY
  VQA_API_KEY="$(cat "$KEY_FILE")"
fi

ABS_CSV="$RUN_ROOT/work/test_abs.csv"
CACHE_DIR="$RUN_ROOT/upstream_clean_routes/cache"
CLEAN_OUT_DIR="$RUN_ROOT/upstream_clean_routes/output"
CLEAN_LOG_DIR="$RUN_ROOT/upstream_clean_routes/logs"
V9_OUT_DIR="$RUN_ROOT/upstream_v9/output"
V9_META_DIR="$RUN_ROOT/upstream_v9/meta"
FINAL_OUT_DIR="$RUN_ROOT/final_router/output"
FINAL_LOG_DIR="$RUN_ROOT/final_router/logs"

mkdir -p "$RUN_ROOT/work" "$CACHE_DIR" "$CLEAN_OUT_DIR" "$CLEAN_LOG_DIR" "$V9_OUT_DIR" "$V9_META_DIR" "$FINAL_OUT_DIR" "$FINAL_LOG_DIR"

{
  printf 'run_tag=%s\n' "$RUN_TAG"
  printf 'input_csv=%s\n' "$INPUT_CSV"
  printf 'task1_data_root=%s\n' "$TASK1_DATA_ROOT"
  printf 'python_bin=%s\n' "$PY"
  printf 'vqa_base_url=%s\n' "${VQA_BASE_URL:-}"
  printf 'vqa_judge_model_a=%s\n' "${VQA_JUDGE_MODEL_A:-gpt-5.2}"
  printf 'vqa_judge_model_b=%s\n' "${VQA_JUDGE_MODEL_B:-gpt-5.1}"
  printf 'vqa_judge_model_c=%s\n' "${VQA_JUDGE_MODEL_C:-gpt-5.2}"
  printf 'vqa_judge_model_d=%s\n' "${VQA_JUDGE_MODEL_D:-gpt-5.1}"
  printf 'vqa_judge_model_l=%s\n' "${VQA_JUDGE_MODEL_L:-gpt-5.2}"
  printf 'vqa_judge_model_w=%s\n' "${VQA_JUDGE_MODEL_W:-gpt-5.2}"
  printf 'vqa_judge_model_v9=%s\n' "${VQA_JUDGE_MODEL:-gpt-5.2}"
} >"$RUN_ROOT/run_meta.txt"

"$PY" "$REPO_ROOT/tools/make_absolute_csv.py" \
  --input "$INPUT_CSV" \
  --task1-root "$TASK1_DATA_ROOT" \
  --output "$ABS_CSV"

run_profile() {
  local label="$1"
  local script="$2"
  local out_txt="$3"
  local out_json="$4"
  INPUT_CSV="$ABS_CSV" OUTPUT_TXT="$out_txt" OUTPUT_JSON="$out_json" PYTHON_BIN="$PY" bash "$script" >"$CLEAN_LOG_DIR/${label}.log" 2>&1
}

run_profile A "$REPO_ROOT/src/upstream/run_profile_A.sh" "$CACHE_DIR/route_a.txt" "$CACHE_DIR/route_a.json" &
pid_a=$!
run_profile B "$REPO_ROOT/src/upstream/run_profile_B.sh" "$CACHE_DIR/route_b.txt" "$CACHE_DIR/route_b.json" &
pid_b=$!
run_profile C "$REPO_ROOT/src/upstream/run_profile_C.sh" "$CACHE_DIR/route_c.txt" "$CACHE_DIR/route_c.json" &
pid_c=$!
run_profile D "$REPO_ROOT/src/upstream/run_profile_D.sh" "$CACHE_DIR/route_d.txt" "$CACHE_DIR/route_d.json" &
pid_d=$!
run_profile L "$REPO_ROOT/src/upstream/run_profile_L.sh" "$CACHE_DIR/route_l.txt" "$CACHE_DIR/route_l.json" &
pid_l=$!
run_profile W "$REPO_ROOT/src/upstream/run_profile_W.sh" "$CACHE_DIR/route_w.txt" "$CACHE_DIR/route_w.json" &
pid_w=$!

wait "$pid_a" "$pid_b" "$pid_c" "$pid_d" "$pid_l" "$pid_w"

INPUT_CSV="$ABS_CSV" \
CACHE_A="$CACHE_DIR/route_a.txt" \
CACHE_B="$CACHE_DIR/route_b.txt" \
CACHE_C="$CACHE_DIR/route_c.txt" \
CACHE_D="$CACHE_DIR/route_d.txt" \
OUTPUT_TXT="$CACHE_DIR/route_pub.txt" \
OUTPUT_JSON="$CACHE_DIR/route_pub.json" \
PYTHON_BIN="$PY" \
bash "$REPO_ROOT/src/upstream/run_hybrid_from_abc.sh" >"$CLEAN_LOG_DIR/PUB.log" 2>&1

INPUT_CSV="$ABS_CSV" \
CACHE_PUB="$CACHE_DIR/route_pub.txt" \
CACHE_A="$CACHE_DIR/route_a.txt" \
CACHE_B="$CACHE_DIR/route_b.txt" \
CACHE_C="$CACHE_DIR/route_c.txt" \
CACHE_D="$CACHE_DIR/route_d.txt" \
CACHE_L="$CACHE_DIR/route_l.txt" \
CACHE_W="$CACHE_DIR/route_w.txt" \
OUTPUT_TXT="$CACHE_DIR/route_rr2.txt" \
OUTPUT_JSON="$CACHE_DIR/route_rr2.json" \
PYTHON_BIN="$PY" \
bash "$REPO_ROOT/src/upstream/run_hybrid_randomroute_v2.sh" >"$CLEAN_LOG_DIR/RR2.log" 2>&1

"$PY" "$REPO_ROOT/src/postprocess/task1_template_tree_runner.py" \
  --input-csv "$ABS_CSV" \
  --output-txt "$CLEAN_OUT_DIR/prediction_tree.txt" \
  --output-json "$CLEAN_OUT_DIR/model_tree.json" \
  --cache-a "$CACHE_DIR/route_a.txt" \
  --cache-b "$CACHE_DIR/route_b.txt" \
  --cache-c "$CACHE_DIR/route_c.txt" \
  --cache-d "$CACHE_DIR/route_d.txt" \
  --cache-l "$CACHE_DIR/route_l.txt" \
  --cache-w "$CACHE_DIR/route_w.txt" \
  --cache-pub "$CACHE_DIR/route_pub.txt" \
  --cache-rr2 "$CACHE_DIR/route_rr2.txt" >"$CLEAN_LOG_DIR/TREE.log" 2>&1

sha256sum "$ABS_CSV" >"$V9_META_DIR/input_sha256.txt"
{
  printf 'input_csv=%s\n' "$INPUT_CSV"
  printf 'abs_csv=%s\n' "$ABS_CSV"
  printf 'python_bin=%s\n' "$PY"
  printf 'vqa_base_url=%s\n' "${VQA_BASE_URL:-}"
  printf 'vqa_judge_model=%s\n' "${VQA_JUDGE_MODEL:-gpt-5.2}"
} >"$V9_META_DIR/run_meta.txt"
"$PY" "$REPO_ROOT/src/upstream/challenge_core_v9_public.py" \
  --input-csv "$ABS_CSV" \
  --output-txt "$V9_OUT_DIR/prediction.txt" \
  --output-json "$V9_OUT_DIR/model.json" >"$RUN_ROOT/upstream_v9/driver.log" 2>&1

"$PY" "$REPO_ROOT/src/postprocess/template_router.py" \
  --input-csv "$ABS_CSV" \
  --output-txt "$FINAL_OUT_DIR/prediction.txt" \
  --output-json "$FINAL_OUT_DIR/model.json" \
  --policy-json "$REPO_ROOT/policies/template_router_policy_v2.json" \
  --cache-a "$CACHE_DIR/route_a.txt" \
  --cache-b "$CACHE_DIR/route_b.txt" \
  --cache-c "$CACHE_DIR/route_c.txt" \
  --cache-d "$CACHE_DIR/route_d.txt" \
  --cache-l "$CACHE_DIR/route_l.txt" \
  --cache-w "$CACHE_DIR/route_w.txt" \
  --cache-pub "$CACHE_DIR/route_pub.txt" \
  --cache-rr2 "$CACHE_DIR/route_rr2.txt" \
  --cache-tree "$CLEAN_OUT_DIR/prediction_tree.txt" \
  --cache-v9 "$V9_OUT_DIR/prediction.txt" >"$FINAL_LOG_DIR/router.log" 2>&1

"$PY" "$REPO_ROOT/tools/make_result_zip.py" \
  --prediction "$FINAL_OUT_DIR/prediction.txt" \
  --model "$FINAL_OUT_DIR/model.json" \
  --output "$FINAL_OUT_DIR/result.zip"

printf 'run_root=%s\n' "$RUN_ROOT"
printf 'prediction=%s\n' "$FINAL_OUT_DIR/prediction.txt"
printf 'model=%s\n' "$FINAL_OUT_DIR/model.json"
printf 'result_zip=%s\n' "$FINAL_OUT_DIR/result.zip"
