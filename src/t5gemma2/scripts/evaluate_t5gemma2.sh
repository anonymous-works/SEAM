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
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${#GPU_IDS[@]}" -lt 1 || "${#GPU_IDS[@]}" -gt 2 ]]; then
  echo "Expected one or two visible GPUs" >&2
  exit 2
fi
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID: ${gpu}" >&2
    exit 2
  fi
done
if [[ "${#GPU_IDS[@]}" -eq 2 && "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
  echo "GPU IDs must be distinct" >&2
  exit 2
fi
if [[ "${#GPU_IDS[@]}" -eq 1 ]]; then
  "${PYTHON_BIN}" -m t5gemma2.evaluate --config "${CONFIG}" --checkpoint "${CHECKPOINT_PATH}" --output_dir "${EVAL_DIR}" "$@" 2>&1 | tee "${LOG_FILE}"
else
  mkdir -p "${EVAL_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${PYTHON_BIN}" -m t5gemma2.evaluate \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT_PATH}" \
    --output_dir "${EVAL_DIR}/shard0" --shard-rank 0 --num-shards 2 "$@" \
    > "${LOG_FILE%.log}_shard0.log" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" "${PYTHON_BIN}" -m t5gemma2.evaluate \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT_PATH}" \
    --output_dir "${EVAL_DIR}/shard1" --shard-rank 1 --num-shards 2 "$@" \
    > "${LOG_FILE%.log}_shard1.log" 2>&1 &
  pid1=$!
  failed=0
  wait "${pid0}" || failed=1
  wait "${pid1}" || failed=1
  if [[ "${failed}" -ne 0 ]]; then
    echo "T5Gemma evaluation failed; see ${LOG_FILE%.log}_shard{0,1}.log" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m t5gemma2.merge_eval_shards \
    --output "${EVAL_DIR}/predictions.jsonl" \
    "${EVAL_DIR}/shard0/predictions.jsonl" "${EVAL_DIR}/shard1/predictions.jsonl"
fi
