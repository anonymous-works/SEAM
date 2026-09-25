set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SYSTEM="${1:?Usage: bash src/scripts/run_rq1.sh seam|t5gemma2|decoder_only DATASET [MODEL_OR_CONFIG]}"
DATASET="${2:?Dataset must be pubmed, arxiv, booksum, or govreport}"
[[ "$DATASET" =~ ^(pubmed|arxiv|booksum|govreport)$ ]] || { echo "Unknown dataset: $DATASET" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
for variable in PYROUGE_HOME_DIR BERTSCORE_MODEL_PATH ALIGNSCORE_MODEL_PATH ALIGNSCORE_CHECKPOINT_PATH SCALE_MODEL_PATH; do
  [[ -n "${!variable:-}" ]] || { echo "Set $variable before starting training" >&2; exit 2; }
done
[[ -f "$PYROUGE_HOME_DIR/ROUGE-1.5.5.pl" && -d "$BERTSCORE_MODEL_PATH" && -d "$ALIGNSCORE_MODEL_PATH" && -f "$ALIGNSCORE_CHECKPOINT_PATH" && -d "$SCALE_MODEL_PATH" ]] || { echo "One or more local metric model paths are invalid" >&2; exit 2; }

case "$SYSTEM" in
  seam)
    export PYTHONPATH="$ROOT/src/seam${PYTHONPATH:+:$PYTHONPATH}"
    CONFIG="${3:-src/seam/configs/seam_${DATASET}.yaml}"
    RUN_DIR="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from seam.config import load_config, resolve_path
config = load_config(sys.argv[1])
print(resolve_path(config["experiment"]["output_dir"], config))
PY
)"
    if ((${#GPUS[@]} > 1)); then
      "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" -m seam.cli train "$CONFIG"
    else
      "$PYTHON_BIN" -m seam.cli train "$CONFIG"
    fi
    PRED="$RUN_DIR/test_predictions.jsonl"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -m seam.cli evaluate "$RUN_DIR/resolved_config.yaml" "$RUN_DIR/last.pt" "$PRED" --split test
    ;;
  t5gemma2)
    CONFIG="${3:-src/t5gemma2/configs/t5gemma2_${DATASET}.yaml}"
    RUN_DIR="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["project"]["output_dir"])
PY
)"
    CONFIG="$CONFIG" bash src/t5gemma2/scripts/train_t5gemma2.sh
    CONFIG="$CONFIG" bash src/t5gemma2/scripts/evaluate_t5gemma2.sh
    PRED="${EVAL_DIR:-$RUN_DIR/eval_outputs}/predictions.jsonl"
    ;;
  decoder_only)
    MODEL="${3:?Provide a decoder_only model key from the benchmark config}"
    CONFIG="${DECODER_CONFIG:-src/decoder_only/configs/decoder_only_benchmark.yaml}"
    export CONFIG
    RUN_DIR="$("$PYTHON_BIN" - "$CONFIG" "$MODEL" "$DATASET" <<'PY'
import sys
from pathlib import Path
import yaml
path = Path(sys.argv[1]).resolve()
config = yaml.safe_load(path.read_text(encoding="utf-8"))
root = Path(config["output_root"])
print((root if root.is_absolute() else path.parent / root).resolve() / f"{sys.argv[2]}__{sys.argv[3]}")
PY
)"
    bash src/decoder_only/scripts/run_decoder_only.sh --models "$MODEL" --datasets "$DATASET"
    PRED="$RUN_DIR/test_predictions.jsonl"
    ;;
  *) echo "Unknown system: $SYSTEM" >&2; exit 2 ;;
esac

bash src/scripts/evaluate_all.sh "$PRED" "$DATASET"
