#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"
: "${INPUT_CSV:?INPUT_CSV is required}"
: "${CACHE_A:?CACHE_A is required}"
: "${CACHE_B:?CACHE_B is required}"
: "${CACHE_C:?CACHE_C is required}"
: "${CACHE_D:?CACHE_D is required}"
: "${OUTPUT_TXT:?OUTPUT_TXT is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"

"$PY" "$DIR/hybrid_runner_task1_public_v8_compliant.py" \
  --input-csv "$INPUT_CSV" \
  --output-txt "$OUTPUT_TXT" \
  --output-json "$OUTPUT_JSON" \
  --cache-a "$CACHE_A" \
  --cache-b "$CACHE_B" \
  --cache-c "$CACHE_C" \
  --cache-d "$CACHE_D"
