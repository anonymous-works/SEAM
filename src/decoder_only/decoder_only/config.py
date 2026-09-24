from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_TOP_LEVEL = {"run", "model", "data", "training", "generation", "limits"}


def _check_keys(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown decoder-baseline {section} key(s): {sorted(unknown)}")


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def load_config(path: str | Path) -> dict[str, Any]:

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Decoder-baseline config must be a mapping: {config_path}")
    config = copy.deepcopy(config)
    config.setdefault("_meta", {})
    config["_meta"]["config_path"] = str(config_path)
    validate_config(config)
    data = config["data"]
    for key in ("train_file", "validation_file", "test_file"):
        data[key] = str(_resolve(data[key], config_path.parent))
    config["run"]["output_dir"] = str(_resolve(config["run"]["output_dir"], config_path.parent))
    return config


def validate_config(config: dict[str, Any]) -> None:
    _check_keys(config, _TOP_LEVEL | {"_meta"}, "top-level")
    for section in ("run", "model", "data", "training", "generation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing decoder-baseline section: {section}")

    run = config["run"]
    _check_keys(run, {"name", "output_dir"}, "run")
    if not str(run.get("name", "")).strip() or not str(run.get("output_dir", "")).strip():
        raise ValueError("run.name and run.output_dir are required")

    model = config["model"]
    _check_keys(
        model,
        {
            "family",
            "name_or_path",
            "model_id",
            "local_files_only",
            "trust_remote_code",
            "torch_dtype",
            "eval_torch_dtype",
            "attn_implementation",
            "gradient_checkpointing",
            "use_cache",
            "diffusion_paradigm",
        },
        "model",
    )
    model_ref = model.get("name_or_path", model.get("model_id", ""))
    if not str(model_ref).strip():
        raise ValueError("model.name_or_path is required")
    if model.get("local_files_only") is not True:
        raise ValueError("model.local_files_only must be true; model downloads are disabled")
    family = str(model.get("family", "causal_lm"))
    if family not in {"causal_lm", "nemotron_diffusion"}:
        raise ValueError("model.family must be causal_lm or nemotron_diffusion")
    if family == "nemotron_diffusion" and str(model.get("diffusion_paradigm", "autoregressive")) != "autoregressive":
        raise ValueError(
            "Nemotron diffusion baselines use the autoregressive objective for a matched decoder-only control"
        )
    for key in ("torch_dtype", "eval_torch_dtype"):
        if str(model.get(key, "bfloat16")).lower() not in {"float32", "float16", "bfloat16", "fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported model.{key}: {model[key]}")
    attention = str(model.get("attn_implementation", "sdpa"))
    if attention not in {"sdpa", "flash_attention_2", "eager"}:
        raise ValueError("model.attn_implementation must be sdpa, flash_attention_2, or eager")

    data = config["data"]
    _check_keys(
        data,
        {
            "train_file",
            "validation_file",
            "test_file",
            "source_field",
            "target_field",
            "id_field",
            "source_prefix",
            "prompt_suffix",
            "max_source_length",
            "max_target_length",
            "max_sequence_length",
            "clean_text",
            "dataset",
        },
        "data",
    )
    for key in ("train_file", "validation_file", "test_file", "source_field", "target_field"):
        if not str(data.get(key, "")).strip():
            raise ValueError(f"data.{key} is required")
    for key in ("max_source_length", "max_target_length", "max_sequence_length"):
        if int(data.get(key, 0)) <= 0:
            raise ValueError(f"data.{key} must be positive")
    if int(data["max_sequence_length"]) < int(data["max_target_length"]):
        raise ValueError("data.max_sequence_length must fit max_target_length")

    training = config["training"]
    _check_keys(
        training,
        {
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "bf16",
            "fp16",
            "tf32",
            "gradient_checkpointing",
            "optim",
            "dataloader_num_workers",
            "dataloader_persistent_workers",
            "dataloader_prefetch_factor",
            "length_bucketing",
            "logging_steps",
            "seed",
        },
        "training",
    )
    for key in ("num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if not 0 <= float(training.get("warmup_ratio", 0.0)) < 1:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    if float(training.get("max_grad_norm", 0.0)) <= 0:
        raise ValueError("training.max_grad_norm must be positive")
    if bool(training.get("bf16", False)) and bool(training.get("fp16", False)):
        raise ValueError("training.bf16 and training.fp16 cannot both be true")
    if int(training.get("dataloader_num_workers", 0)) < 0:
        raise ValueError("training.dataloader_num_workers must be non-negative")
    if int(training.get("dataloader_prefetch_factor", 2)) <= 0:
        raise ValueError("training.dataloader_prefetch_factor must be positive")

    generation = config["generation"]
    _check_keys(
        generation,
        {
            "batch_size",
            "max_new_tokens",
            "min_new_tokens",
            "num_beams",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "no_repeat_ngram_size",
        },
        "generation",
    )
    if int(generation.get("batch_size", 0)) <= 0:
        raise ValueError("generation.batch_size must be positive")
    if int(generation.get("num_beams", 1)) != 1:
        raise ValueError("Decoder-only controls use num_beams=1 for the greedy/sampling comparison")
    if int(generation.get("max_new_tokens", 0)) <= 0 or not 0 <= int(generation.get("min_new_tokens", 0)) < int(
        generation.get("max_new_tokens", 0)
    ):
        raise ValueError("Require 0 <= generation.min_new_tokens < generation.max_new_tokens")
    if bool(generation.get("do_sample", False)) and float(generation.get("temperature", 0.0)) <= 0:
        raise ValueError("generation.temperature must be positive when do_sample=true")
    if not bool(generation.get("do_sample", False)) and float(generation.get("temperature", 0.0)) < 0:
        raise ValueError("generation.temperature must be non-negative")
    if int(generation.get("top_k", 0)) < 0:
        raise ValueError("generation.top_k must be non-negative")
    if not 0 < float(generation.get("top_p", 1.0)) <= 1:
        raise ValueError("generation.top_p must be in (0, 1]")
    if float(generation.get("repetition_penalty", 1.0)) <= 0:
        raise ValueError("generation.repetition_penalty must be positive")
    if int(generation.get("no_repeat_ngram_size", 0)) < 0:
        raise ValueError("generation.no_repeat_ngram_size must be non-negative")
    limits = config.get("limits", {})
    if not isinstance(limits, dict):
        raise ValueError("limits must be a mapping")
    _check_keys(limits, {"max_train_examples", "max_validation_examples", "max_test_examples"}, "limits")
    for key, value in limits.items():
        if int(value) < 0:
            raise ValueError(f"limits.{key} must be non-negative")
