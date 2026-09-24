set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${BASELINE_ROOT}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
elif [[ -x "/Users/kieugiangbien/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/Users/kieugiangbien/bienkieu_env/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_ID must be a comma-separated list of GPU indices, for example GPU_ID=0,1." >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${GPU_ID}"
if (( ${#GPU_IDS[@]} > 1 )); then
  for ((index = 0; index < ${#GPU_IDS[@]}; index++)); do
    for ((other = index + 1; other < ${#GPU_IDS[@]}; other++)); do
      if [[ "${GPU_IDS[index]}" == "${GPU_IDS[other]}" ]]; then
        echo "GPU_ID contains duplicate devices: ${GPU_ID}" >&2
        exit 2
      fi
    done
  done
fi
export GPU_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${BASELINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-${BASELINE_ROOT}/configs/decoder_only_benchmark.yaml}"
exec "${PYTHON_BIN}" -m decoder_only.benchmark --config "${CONFIG}" "$@"
