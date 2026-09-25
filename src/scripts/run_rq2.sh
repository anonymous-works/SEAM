set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
DATASET="${1:?Usage: bash src/scripts/run_rq2.sh DATASET full|no_bridge|no_copy|direct_ca}"
ARM="${2:?Choose full, no_bridge, no_copy, or direct_ca}"
if [[ "$ARM" == vanilla ]]; then ARM=direct_ca; fi
[[ "$DATASET" =~ ^(pubmed|arxiv|booksum|govreport)$ && "$ARM" =~ ^(full|no_bridge|no_copy|direct_ca)$ ]] || { echo "Invalid dataset or ablation" >&2; exit 2; }
export PYTHONPATH="$ROOT/src/seam${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="$ROOT/runs/rq2/${DATASET}_${ARM}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
echo "RQ2 $DATASET $ARM -> $RUN_DIR"
"$PYTHON_BIN" - "$ROOT" "$DATASET" "$ARM" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
import yaml
from seam.config import load_config, resolve_path, validate_config

root, dataset, arm, run_dir = sys.argv[1:]
config = load_config(Path(root) / "src/seam/configs" / f"seam_{dataset}.yaml")
for field in ("train_file", "validation_file", "test_file"):
    config["data"][field] = str(resolve_path(config["data"][field], config))
config["experiment"].update(name=f"rq2_{dataset}_{arm}", output_dir=run_dir)
if arm in {"no_bridge", "direct_ca"}:
    config["architecture"]["bridge_mode"] = "direct_projection"
if arm in {"no_copy", "direct_ca"}:
    config["decoder"]["source_copy"]["enabled"] = False
config.pop("_meta", None)
validate_config(config)
(Path(run_dir) / "input_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
bash src/scripts/run_rq1.sh seam "$DATASET" "$RUN_DIR/input_config.yaml"
