from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

LED_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LED_ROOT.parents[1]
DEFAULT_CONFIG = LED_ROOT / "configs" / "led_benchmark.yaml"
DATASETS = {"pubmed", "arxiv", "booksum", "govreport"}


def load_config(
    dataset: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    num_train_epochs: int | None = None,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(sorted(DATASETS))}")
    if num_train_epochs is not None and num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")

    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    config = copy.deepcopy(raw)
    data = config["datasets"][dataset]
    config["dataset"] = dataset
    config["data"] = data
    config["model"]["name_or_path"] = os.environ.get("LED_MODEL_PATH", config["model"]["model_id"])
    config["training"]["num_train_epochs"] = num_train_epochs
    config["generation"]["max_new_tokens"] = int(data["max_target_length"])
    config["run"] = {
        "name": f"led_{dataset}",
        "output_dir": str((PROJECT_ROOT / config["output_root"] / dataset).resolve()),
    }
    for split in ("train_file", "validation_file", "test_file"):
        value = Path(data[split]).expanduser()
        data[split] = str(value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve())
    config["_meta"] = {
        "config_path": str(path),
        "project_root": str(PROJECT_ROOT),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    validate_config(config, training=num_train_epochs is not None)
    return config


def validate_config(config: dict[str, Any], *, training: bool = False) -> None:
    model = config["model"]
    data = config["data"]
    train = config["training"]
    generation = config["generation"]
    if not model.get("name_or_path"):
        raise ValueError("Set LED_MODEL_PATH or model.model_id")
    if model.get("local_files_only") is not True:
        raise ValueError("LED runs must use a local checkpoint; model.local_files_only must be true")
    for field in ("train_file", "validation_file", "test_file"):
        if not data.get(field):
            raise ValueError(f"data.{field} is required")
    for field in ("max_source_length", "max_target_length"):
        if int(data.get(field, 0)) <= 0:
            raise ValueError(f"data.{field} must be positive")
    if int(data["max_source_length"]) > int(model["max_position_embeddings"]):
        raise ValueError("Dataset input limit exceeds LED's configured position limit")
    if int(train["per_device_train_batch_size"]) * int(train["gradient_accumulation_steps"]) != int(
        train["effective_batch_size"]
    ):
        raise ValueError("per-device batch × gradient accumulation must equal the configured effective batch")
    if int(config["_meta"].get("world_size", 1)) != 1:
        raise ValueError("The reported LED recipe uses one GPU; launch with exactly one process")
    if training and int(train.get("num_train_epochs") or 0) <= 0:
        raise ValueError("Supply the actual training horizon with --num-train-epochs")
    if int(generation["max_new_tokens"]) != int(data["max_target_length"]):
        raise ValueError("Generation cap must match the dataset output limit")
