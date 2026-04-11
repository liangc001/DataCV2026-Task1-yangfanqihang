#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"
: "${INPUT_CSV:?INPUT_CSV is required}"
: "${OUTPUT_TXT:?OUTPUT_TXT is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"

export VQA_JUDGE_MODEL="${VQA_JUDGE_MODEL_B:-gpt-5.1}"
export VQA_FALLBACK_MODEL="${VQA_FALLBACK_MODEL_B:-gpt-5.1}"
export VQA_V8_CHALLENGE_ENABLE="${VQA_V8_CHALLENGE_ENABLE:-1}"

"$PY" "$DIR/challenge_runner_task1_v8_public_impl.py" \
  --input-csv "$INPUT_CSV" \
  --output-txt "$OUTPUT_TXT" \
  --output-json "$OUTPUT_JSON"
