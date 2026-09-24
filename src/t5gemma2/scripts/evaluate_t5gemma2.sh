set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${BASELINE_ROOT}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${BASELINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-src/t5gemma2/configs/t5gemma2_pubmed.yaml}"
RUN_DIR="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["project"]["output_dir"])
PY
)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_DIR}/final_model}"
EVAL_DIR="${EVAL_DIR:-${RUN_DIR}/eval_outputs}"
LOG_DIR="${LOG_DIR:-runs/t5gemma2/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_eval.log"
"${PYTHON_BIN}" -m t5gemma2.evaluate --config "${CONFIG}" --checkpoint "${CHECKPOINT_PATH}" --output_dir "${EVAL_DIR}" "$@" 2>&1 | tee "${LOG_FILE}"
