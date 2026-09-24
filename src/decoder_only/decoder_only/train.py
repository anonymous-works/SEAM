from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import load_config
from .data import CausalCollator, CausalSummarizationDataset

LOGGER = logging.getLogger("decoder_only.train")


@dataclass(frozen=True)
class _DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _read_distributed_context() -> _DistributedContext:

    def _integer(name: str, default: str) -> int:
        raw = os.environ.get(name, default)
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    world_size = _integer("WORLD_SIZE", "1")
    rank = _integer("RANK", "0")
    local_rank = _integer("LOCAL_RANK", "0" if world_size > 1 else "-1")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, {world_size}), got {rank}")
    if world_size > 1 and local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative for DDP, got {local_rank}")
    return _DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def _initialize_distributed(context: _DistributedContext) -> None:

    if not context.enabled:
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable; cannot run a multi-process baseline")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if context.local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={context.local_rank} but only {device_count} visible CUDA device(s) are available"
            )
        torch.cuda.set_device(context.local_rank)
        return


def _wait_for_path(path: Path, *, timeout_seconds: float = 300.0) -> None:

    deadline = time.monotonic() + float(timeout_seconds)
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for rank-zero setup file: {path}")
        time.sleep(0.1)


def _destroy_distributed(context: _DistributedContext) -> None:
    if context.enabled and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _validate_trainer_distribution(training_args: Any, context: _DistributedContext) -> None:

    if not context.enabled:
        return
    actual_world_size = int(getattr(training_args, "world_size", 1))
    parallel_mode = str(getattr(training_args, "parallel_mode", "")).lower()
    if (
        actual_world_size != context.world_size
        or "distributed" not in parallel_mode
        or "not_distributed" in parallel_mode
    ):
        raise RuntimeError(
            "DDP launch was not recognized by Transformers: "
            f"expected world_size={context.world_size}, got world_size={actual_world_size}, "
            f"parallel_mode={parallel_mode!r}"
        )


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float32", "fp32"}:
        return torch.float32
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _context_length(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "max_seq_len", "n_positions", "max_sequence_length"):
        value = getattr(config, name, None)
        if value is not None and int(value) > 0:
            return int(value)
    raise RuntimeError("Cannot verify decoder context length")


def _load_tokenizer_and_model(config: dict[str, Any], *, evaluation: bool = False) -> tuple[Any, torch.nn.Module]:

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    name = str(model_config.get("name_or_path", model_config.get("model_id", "")))
    local_path = Path(name).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(
            "Local model directory does not exist: "
            f"{local_path}. Set the model's *_PATH variable; Hugging Face IDs and downloads are disabled."
        )
    name = str(local_path)
    common = {
        "local_files_only": True,
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
    }
    tokenizer = AutoTokenizer.from_pretrained(name, **common)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if evaluation else "right"
    dtype_name = model_config.get("eval_torch_dtype" if evaluation else "torch_dtype", "bfloat16")
    model_kwargs = {
        **common,
        "dtype": _dtype(str(dtype_name)),
        "attn_implementation": str(model_config.get("attn_implementation", "sdpa")),
        "low_cpu_mem_usage": True,
    }
    if str(model_config.get("family", "causal_lm")) == "nemotron_diffusion":
        raw_config = AutoConfig.from_pretrained(name, **common)
        raw_config.dlm_paradigm = str(model_config.get("diffusion_paradigm", "autoregressive"))
        model = AutoModel.from_pretrained(name, config=raw_config, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = bool(model_config.get("use_cache", evaluation))
    return tokenizer, model


def _enable_training_features(model: torch.nn.Module, config: dict[str, Any]) -> None:
    model_config = config["model"]
    training = config["training"]
    if bool(model_config.get("gradient_checkpointing", training.get("gradient_checkpointing", True))):
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if enable is None:
            raise RuntimeError("Configured gradient checkpointing is unsupported by this model")
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable()
        input_grads = getattr(model, "enable_input_require_grads", None)
        if input_grads is not None:
            input_grads()
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def _training_arguments(config: dict[str, Any], output_dir: Path) -> Any:
    from transformers import TrainingArguments

    training = config["training"]
    values: dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_state"),
        "num_train_epochs": int(training["num_train_epochs"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "adam_beta1": float(training.get("adam_beta1", 0.9)),
        "adam_beta2": float(training.get("adam_beta2", 0.95)),
        "adam_epsilon": float(training.get("adam_epsilon", 1e-8)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": "cosine",
        "optim": str(training.get("optim", "adamw_torch_fused")),
        "bf16": bool(training.get("bf16", True)),
        "fp16": bool(training.get("fp16", False)),
        "tf32": bool(training.get("tf32", True)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "logging_steps": int(training.get("logging_steps", 10)),
        "logging_strategy": "steps",
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": int(training.get("dataloader_num_workers", 4)),
        "dataloader_pin_memory": True,
        "seed": int(training.get("seed", 42)),
        "ddp_find_unused_parameters": False,
    }
    num_workers = int(training.get("dataloader_num_workers", 4))
    if num_workers > 0:
        values.update(
            {
                "dataloader_persistent_workers": bool(training.get("dataloader_persistent_workers", True)),
                "dataloader_prefetch_factor": int(training.get("dataloader_prefetch_factor", 2)),
            }
        )
    parameters = __import__("inspect").signature(TrainingArguments.__init__).parameters
    values = {key: value for key, value in values.items() if key in parameters}
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        values["evaluation_strategy"] = "no"
    return TrainingArguments(**values)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(config_path: str | Path, *, overwrite_output_dir: bool = False) -> Path:
    config = load_config(config_path)
    output_dir = Path(config["run"]["output_dir"])
    distributed = _read_distributed_context()

    has_artifacts = distributed.is_main and output_dir.exists() and any(output_dir.iterdir())
    if distributed.is_main and has_artifacts and not overwrite_output_dir:
        raise FileExistsError(
            f"Refusing to mix a decoder-baseline run with existing artifacts: {output_dir}. "
            "Use --overwrite-output-dir for an intentional rerun."
        )

    _initialize_distributed(distributed)
    if distributed.is_main:
        if has_artifacts:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / "RUNNING"
        marker.write_text(f"pid={os.getpid()} rank={distributed.rank}\n", encoding="utf-8")
    else:
        marker = output_dir / "RUNNING"
        _wait_for_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        resolved_config = output_dir / "resolved_config.yaml"
        if distributed.is_main:
            resolved_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        else:
            _wait_for_path(resolved_config)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                *([logging.FileHandler(output_dir / "train.log", encoding="utf-8")] if distributed.is_main else []),
            ],
            force=True,
        )
        from transformers import set_seed

        training = config["training"]
        set_seed(int(training.get("seed", 42)))
        target = (
            torch.device("cuda", distributed.local_rank if distributed.enabled else 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if target.type != "cuda":
            LOGGER.warning("No CUDA device is visible; running on CPU (world_size=%d)", distributed.world_size)
        if target.type == "cuda" and bool(training.get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        training_args = _training_arguments(config, output_dir)

        _validate_trainer_distribution(training_args, distributed)
        tokenizer, model = _load_tokenizer_and_model(config)
        context_length = _context_length(model)
        configured_context = int(config["data"]["max_sequence_length"])
        if context_length < configured_context:
            LOGGER.info("Capping sequence length from %d to model context %d", configured_context, context_length)
        _enable_training_features(model, config)
        model.to(target)
        train_cfg = config.get("limits", {})
        dataset = CausalSummarizationDataset(
            config["data"]["train_file"],
            tokenizer,
            config["data"],
            max_examples=int(train_cfg.get("max_train_examples", 0)),
            model_context_length=context_length,
        )
        collator = CausalCollator(tokenizer.pad_token_id)
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": dataset,
            "data_collator": collator,
        }
        from transformers import Trainer

        trainer_class = Trainer
        if bool(training.get("length_bucketing", True)):
            from transformers.trainer_pt_utils import LengthGroupedSampler

            class LengthGroupedTrainer(Trainer):
                def _get_train_sampler(self, train_dataset=None):
                    dataset_for_sampler = train_dataset if train_dataset is not None else self.train_dataset
                    lengths = getattr(dataset_for_sampler, "length_estimates", None)
                    if not lengths:
                        return super()._get_train_sampler(dataset_for_sampler)
                    generator = torch.Generator()
                    generator.manual_seed(int(self.args.seed))
                    grouped_batch_size = max(
                        1,
                        int(self.args.train_batch_size) * int(self.args.gradient_accumulation_steps),
                    )
                    return LengthGroupedSampler(
                        grouped_batch_size,
                        lengths=list(lengths),
                        generator=generator,
                    )

            trainer_class = LengthGroupedTrainer

        trainer_parameters = __import__("inspect").signature(Trainer.__init__).parameters
        if "processing_class" in trainer_parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_parameters:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = trainer_class(**trainer_kwargs)
        total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        LOGGER.info(
            "run=%s rank=%d world_size=%d model=%s device=%s examples=%d epochs=%d per_device_batch=%d "
            "global_batch=%d accumulation=%d length_bucketing=%s parameters=%d",
            config["run"]["name"],
            distributed.rank,
            distributed.world_size,
            config["model"].get("model_id", config["model"].get("name_or_path")),
            target,
            len(dataset),
            int(training["num_train_epochs"]),
            int(training["per_device_train_batch_size"]),
            int(training["per_device_train_batch_size"])
            * distributed.world_size
            * int(training["gradient_accumulation_steps"]),
            int(training["gradient_accumulation_steps"]),
            bool(training.get("length_bucketing", True)),
            total_parameters,
        )
        result = trainer.train()
        final_dir = output_dir / "final_model"
        if distributed.is_main:
            final_dir.mkdir(parents=True, exist_ok=True)
            model.config.use_cache = True
            model.save_pretrained(final_dir, safe_serialization=True)
            tokenizer.save_pretrained(final_dir)
            trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
            _write_json(
                output_dir / "run_manifest.json",
                {
                    "run": config["run"]["name"],
                    "model_id": config["model"].get("model_id", config["model"].get("name_or_path")),
                    "model_path": config["model"].get("name_or_path"),
                    "dataset": config["data"].get("dataset", ""),
                    "num_train_examples": len(dataset),
                    "num_epochs": int(training["num_train_epochs"]),
                    "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
                    "world_size": distributed.world_size,
                    "global_batch_size": int(training["per_device_train_batch_size"])
                    * distributed.world_size
                    * int(training["gradient_accumulation_steps"]),
                    "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
                    "trainable_parameter_elements": total_parameters,
                    "train_metrics": result.metrics,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "prompt_protocol": "t5gemma_source_prefix_plus_causal_target_masking",
                },
            )
        if distributed.is_main:
            marker.unlink(missing_ok=True)
            (output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
            LOGGER.info("completed run=%s elapsed_seconds=%.1f", config["run"]["name"], time.time() - started)
        else:
            _wait_for_path(output_dir / "COMPLETE")
        return final_dir
    except Exception:
        LOGGER.exception("decoder-baseline run failed: %s", config["run"]["name"])
        raise
    finally:
        if distributed.is_main:
            marker.unlink(missing_ok=True)
        _destroy_distributed(distributed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full fine-tune one decoder-only summarization baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    train(args.config, overwrite_output_dir=args.overwrite_output_dir)


if __name__ == "__main__":
    main()
