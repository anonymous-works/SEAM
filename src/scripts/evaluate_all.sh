set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PRED="${1:?Usage: bash src/scripts/evaluate_all.sh PREDICTIONS.jsonl DATASET}"
DATASET="${2:?Dataset must be pubmed, arxiv, booksum, or govreport}"
[[ "$DATASET" =~ ^(pubmed|arxiv|booksum|govreport)$ ]] || { echo "Unknown dataset: $DATASET" >&2; exit 2; }
: "${PYROUGE_HOME_DIR:?Set PYROUGE_HOME_DIR to ROUGE-1.5.5}"
: "${BERTSCORE_MODEL_PATH:?Set BERTSCORE_MODEL_PATH to a local model directory}"
SOURCE="$ROOT/src/seam/datasets/$DATASET/test.jsonl"
[[ -f "$PRED" && -f "$SOURCE" ]] || { echo "Missing predictions or test source: $PRED $SOURCE" >&2; exit 2; }
[[ -f "$PYROUGE_HOME_DIR/ROUGE-1.5.5.pl" && -d "$BERTSCORE_MODEL_PATH" ]] || { echo "ROUGE or BERTScore model path is invalid" >&2; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" src/evaluate/evaluate_rouge.py "$PRED"
"$PYTHON_BIN" src/evaluate/evaluate_bertscore.py "$PRED" --model-path "$BERTSCORE_MODEL_PATH"
