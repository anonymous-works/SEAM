#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LED_ROOT="$ROOT/src/led"
cd "$ROOT"
export PYTHONPATH="$LED_ROOT:$ROOT/src/t5gemma2${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="${1:?Usage: bash src/led/scripts/run_led.sh DATASET --num-train-epochs N [--config PATH]}"
shift
case "$DATASET" in pubmed|arxiv|booksum|govreport) ;; *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;; esac

CONFIG="src/led/configs/led_benchmark.yaml"
EPOCHS=""
OVERWRITE=()
while (($#)); do
  case "$1" in
    --config)
      (($# >= 2)) || { echo "--config requires a path" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --num-train-epochs)
      (($# >= 2)) || { echo "--num-train-epochs requires a value" >&2; exit 2; }
      EPOCHS="$2"
      shift 2
      ;;
    --overwrite-output-dir)
      OVERWRITE+=(--overwrite-output-dir)
      shift
      ;;
    *) echo "Unknown LED runner option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$EPOCHS" ]] || { echo "Pass the run's verified epoch count with --num-train-epochs" >&2; exit 2; }

"$PYTHON_BIN" -m led.train --dataset "$DATASET" --config "$CONFIG" --num-train-epochs "$EPOCHS" "${OVERWRITE[@]}"
"$PYTHON_BIN" -m led.evaluate --dataset "$DATASET" --config "$CONFIG"
PREDICTIONS="$ROOT/runs/led/$DATASET/eval/predictions.jsonl"
bash src/scripts/evaluate_all.sh "$PREDICTIONS" "$DATASET"
