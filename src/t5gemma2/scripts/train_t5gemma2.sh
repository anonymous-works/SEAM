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
LOG_DIR="${LOG_DIR:-runs/t5gemma2/logs}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPU_IDS <<< "${VISIBLE_GPUS}"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ "${GPU_COUNT}" -lt 1 || "${GPU_COUNT}" -gt 2 ]]; then
  echo "Expected one or two visible GPUs, got ${VISIBLE_GPUS}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_train.log"
if [[ "${GPU_COUNT}" == "2" ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 -m t5gemma2.train --config "${CONFIG}" "$@" 2>&1 | tee "${LOG_FILE}"
else
  "${PYTHON_BIN}" -m t5gemma2.train --config "${CONFIG}" "$@" 2>&1 | tee "${LOG_FILE}"
fi
