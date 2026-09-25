set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
DATASET="${1:?Usage: bash src/scripts/run_rq3.sh DATASET pplx|qwen_embed|qwen_causal|nemotron_embed|llama3_2_1b}"
ENCODER="${2:?Choose pplx, qwen_embed, qwen_causal, nemotron_embed, or llama3_2_1b}"
[[ "$DATASET" =~ ^(pubmed|arxiv|booksum|govreport)$ ]] || { echo "Unknown dataset: $DATASET" >&2; exit 2; }
case "$ENCODER" in
  pplx) MODEL="perplexity-ai/pplx-embed-v1-0.6b" ;;
  qwen_embed) MODEL="Qwen/Qwen3-Embedding-0.6B" ;;
  qwen_causal) MODEL="Qwen/Qwen3-0.6B" ;;
  nemotron_embed) MODEL="${ENCODER_MODEL_PATH:-${NEMOTRON_ENCODER_PATH:?Set NEMOTRON_ENCODER_PATH to the local Nemotron-3-Embed-1B-BF16 checkpoint}}" ;;
  llama3_2_1b) MODEL="meta-llama/Llama-3.2-1B" ;;
  *) echo "Unknown encoder: $ENCODER" >&2; exit 2 ;;
esac
MODEL="${ENCODER_MODEL_PATH:-$MODEL}"
[[ "$ENCODER" != nemotron_embed || -d "$MODEL" ]] || { echo "Nemotron checkpoint directory not found: $MODEL" >&2; exit 2; }
export PYTHONPATH="$ROOT/src/seam${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_CONFIG="src/seam/configs/seam_${DATASET}.yaml"
RUN_DIR="$ROOT/runs/rq3/${DATASET}_${ENCODER}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
echo "RQ3 $DATASET $ENCODER -> $RUN_DIR"
"$PYTHON_BIN" - "$ROOT" "$BASE_CONFIG" "$DATASET" "$ENCODER" "$MODEL" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
import yaml
from seam.config import load_config, resolve_path, validate_config

root, base_config, dataset, encoder, model, run_dir = sys.argv[1:]
config = load_config(Path(root) / base_config)
if config["architecture"].get("bridge_mode", "seam") != "seam" or not config["decoder"]["source_copy"]["enabled"]:
    raise ValueError("RQ3 requires the full SEAM bridge and source-copy route")
for field in ("train_file", "validation_file", "test_file"):
    config["data"][field] = str(resolve_path(config["data"][field], config))
config["model"]["encoder_name"] = model
config["experiment"].update(name=f"rq3_{dataset}_{encoder}", output_dir=run_dir)
config.pop("_meta", None)
validate_config(config)
(Path(run_dir) / "input_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
bash src/scripts/run_rq1.sh seam "$DATASET" "$RUN_DIR/input_config.yaml"
