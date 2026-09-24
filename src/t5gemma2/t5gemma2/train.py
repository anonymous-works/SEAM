from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import Dataset
from .data import load_summarization_jsonl, resolve_path

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

T5GEMMA_ROOT = Path(__file__).resolve().parents[1]


def load_env_file() -> None:
    env_file = Path(os.environ.get("ENV_FILE", T5GEMMA_ROOT / "env.txt"))
    if not env_file.is_absolute():
        env_file = T5GEMMA_ROOT.parents[1] / env_file
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if distributed_world_size() > 1 and not is_main_process():
        logging.getLogger().setLevel(logging.WARNING)


def distributed_world_size() -> int:

    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer") from exc
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    return world_size


def distributed_rank() -> int:

    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise ValueError("RANK must be an integer") from exc
    if rank < 0:
        raise ValueError(f"RANK must be non-negative, got {rank}")
    return rank


def is_main_process() -> bool:
    return distributed_world_size() == 1 or distributed_rank() == 0


def initialize_distributed() -> bool:

    world_size = distributed_world_size()
    if world_size == 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("Two-GPU T5Gemma fine-tuning requires CUDA")
    if not dist.is_available():
        raise RuntimeError("This PyTorch build does not include torch.distributed")
    local_rank = int(os.environ.get("LOCAL_RANK", distributed_rank()))
    device_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(f"LOCAL_RANK={local_rank} is outside the {device_count} visible CUDA devices")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout_minutes = int(os.environ.get("T5GEMMA_DDP_TIMEOUT_MINUTES", "30"))
        if timeout_minutes <= 0:
            raise ValueError("T5GEMMA_DDP_TIMEOUT_MINUTES must be positive")
        init_kwargs: Dict[str, Any] = {
            "backend": os.environ.get("T5GEMMA_DDP_BACKEND", "nccl"),
            "timeout": timedelta(minutes=timeout_minutes),
        }

        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_kwargs)
        return True
    return False


def distributed_barrier() -> None:
    if distributed_world_size() > 1 and dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", distributed_rank()))
        barrier_kwargs: Dict[str, Any] = {}
        if "device_ids" in inspect.signature(dist.barrier).parameters:
            barrier_kwargs["device_ids"] = [local_rank]
        dist.barrier(**barrier_kwargs)


def destroy_distributed(initialized_here: bool) -> None:
    if initialized_here and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SummarizationDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        source_prefix: str,
        max_source_length: int,
        max_target_length: int,
        *,
        data_config: Dict[str, Any] | None = None,
    ) -> None:

        loader_config = dict(data_config or {})
        loader_config.setdefault("source_prefix", source_prefix)
        self.examples = load_summarization_jsonl(path, loader_config)
        self.tokenizer = tokenizer
        self.source_prefix = str(loader_config.get("source_prefix", source_prefix))
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.examples[index]
        model_inputs = self.tokenizer(
            self.source_prefix + row.source,
            max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=row.target,
            max_length=self.max_target_length - 1,
            truncation=True,
        )
        label_ids = labels["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            if not label_ids or label_ids[-1] != self.tokenizer.eos_token_id:
                label_ids.append(self.tokenizer.eos_token_id)

        model_inputs["labels"] = label_ids
        return model_inputs


def torch_dtype_from_config(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if lowered in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def make_training_arguments(cfg: Dict[str, Any], output_dir: Path) -> Seq2SeqTrainingArguments:
    train_cfg = cfg["training"]
    world_size = distributed_world_size()
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_state"),
        "num_train_epochs": int(train_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(train_cfg["learning_rate"]),
        "adam_beta1": float(train_cfg.get("adam_beta1", 0.9)),
        "adam_beta2": float(train_cfg.get("adam_beta2", 0.95)),
        "adam_epsilon": float(train_cfg.get("adam_epsilon", 1e-8)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "optim": str(train_cfg.get("optim", "adamw_torch")),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", False)),
        "tf32": bool(train_cfg.get("tf32", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "logging_strategy": "steps",
        "save_strategy": "no",
        "save_safetensors": True,
        "report_to": [],
        "predict_with_generate": False,
        "group_by_length": bool(train_cfg.get("group_by_length", False)),
        "remove_unused_columns": True,
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": True,
        "seed": int(train_cfg.get("seed", 42)),
    }
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    valid_kwargs = {k: v for k, v in kwargs.items() if k in params}
    if world_size > 1:
        if "ddp_find_unused_parameters" in params:
            valid_kwargs["ddp_find_unused_parameters"] = bool(train_cfg.get("ddp_find_unused_parameters", True))
        if "ddp_backend" in params and train_cfg.get("ddp_backend"):
            valid_kwargs["ddp_backend"] = str(train_cfg["ddp_backend"])
        if "ddp_timeout" in params and train_cfg.get("ddp_timeout") is not None:
            valid_kwargs["ddp_timeout"] = int(train_cfg["ddp_timeout"])
        if "gradient_checkpointing_kwargs" in params and bool(train_cfg.get("gradient_checkpointing", False)):
            checkpointing_kwargs = train_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
            valid_kwargs["gradient_checkpointing_kwargs"] = dict(checkpointing_kwargs)

    eval_strat = str(train_cfg.get("eval_strategy", "no"))
    if "eval_strategy" in params:
        valid_kwargs["eval_strategy"] = eval_strat
    else:
        valid_kwargs["evaluation_strategy"] = eval_strat
    return Seq2SeqTrainingArguments(**valid_kwargs)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_resolved_config(config: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def assert_hf_uploads_disabled(cfg: Dict[str, Any]) -> None:
    hf_cfg = cfg.get("huggingface", {})
    if bool(hf_cfg.get("enabled", False)):
        raise RuntimeError("Hugging Face uploads are disabled for this local-only experiment")


def log_model_summary(model: torch.nn.Module, cfg: Dict[str, Any], train_size: int, eval_size: int) -> None:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    ratio = 100.0 * trainable / max(1, total)
    logging.info("T5Gemma Full Fine-tuning Summary")
    logging.info("=" * 50)
    logging.info("Model:            %s", cfg["model"]["model_name_or_path"])
    logging.info("Source/Target:    %s / %s tokens", cfg["data"]["max_source_length"], cfg["data"]["max_target_length"])
    logging.info("Train examples:   %s", train_size)
    logging.info("Eval examples:    %s", eval_size)
    logging.info("Epochs:           %s", cfg["training"]["num_train_epochs"])
    logging.info("Batch size:       %s", cfg["training"]["per_device_train_batch_size"])
    logging.info("Grad accum:       %s", cfg["training"]["gradient_accumulation_steps"])
    logging.info(
        "Effective batch:  %s",
        int(cfg["training"]["per_device_train_batch_size"]) * int(cfg["training"]["gradient_accumulation_steps"]),
    )
    logging.info("DDP world size:   %s", distributed_world_size())
    logging.info(
        "Global effective batch: %s",
        int(cfg["training"]["per_device_train_batch_size"])
        * int(cfg["training"]["gradient_accumulation_steps"])
        * distributed_world_size(),
    )
    logging.info("Learning rate:    %s", cfg["training"]["learning_rate"])
    logging.info("Trainable params: %s", f"{trainable:,}")
    logging.info("Total params:     %s", f"{total:,}")
    logging.info("Trainable ratio:  %.4f%%", ratio)
    logging.info("=" * 50)
    if trainable != total:
        frozen = [name for name, param in model.named_parameters() if not param.requires_grad]
        raise RuntimeError(
            "Full fine-tuning requires 100% trainable parameters, but found "
            f"{len(frozen)} frozen tensors: {frozen[:20]}"
        )


def main() -> None:
    load_env_file()
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()

    project_root = T5GEMMA_ROOT.parents[1]
    config_path = resolve_path(args.config, base=project_root)
    with config_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    assert_hf_uploads_disabled(cfg)
    model_name = os.environ.get("T5GEMMA_MODEL_PATH", cfg["model"]["model_name_or_path"])
    model_path = resolve_path(model_name, base=project_root)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local T5Gemma model directory not found: {model_path}; set T5GEMMA_MODEL_PATH")
    model_name = str(model_path)
    cfg["model"]["model_name_or_path"] = model_name

    if "lora" in cfg:
        raise ValueError("This is a full fine-tuning pipeline; remove the obsolete 'lora' config block.")
    mode = str(cfg.get("training", {}).get("mode", "full_finetune"))
    if mode != "full_finetune":
        raise ValueError(f"training.mode must be 'full_finetune', got {mode!r}")

    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)
    if bool(cfg["training"].get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = resolve_path(cfg["project"]["output_dir"], base=project_root)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite_output_dir:
            raise FileExistsError(
                f"Refusing to mix a new T5Gemma run with existing artifacts in {output_dir}. "
                "Use a fresh project.output_dir or pass --overwrite-output-dir."
            )
    distributed_initialized = initialize_distributed()
    main_process = is_main_process()
    if main_process:
        if args.overwrite_output_dir and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier()
    running_marker = output_dir / "RUNNING"
    if main_process:
        running_marker.write_text(
            json.dumps(
                {
                    "config": str(config_path.resolve()),
                    "status": "running",
                    "world_size": distributed_world_size(),
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        save_resolved_config(cfg, output_dir)
    distributed_barrier()

    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))
    dtype = torch_dtype_from_config(cfg["model"].get("torch_dtype", "bfloat16"))

    train_file = resolve_path(cfg["data"]["train_file"], base=project_root)
    eval_file_str = cfg["data"].get("eval_file") or cfg["data"].get("validation_file")
    eval_file = resolve_path(eval_file_str, base=project_root) if eval_file_str else None
    if not train_file.exists():
        raise FileNotFoundError(train_file)
    if eval_file and not eval_file.exists():
        raise FileNotFoundError(eval_file)
    logging.info("Loading train split: %s", train_file)
    logging.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    logging.info("Loading base model: %s", model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    if dtype == torch.float32 and any(
        parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()
    ):
        logging.info("Converting locally loaded base weights to FP32 master parameters")
        model.to(dtype=torch.float32)
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    unique_parameter_elements = int(sum(parameter.numel() for parameter in model.parameters()))
    logging.info("Unique parameter elements: %d", unique_parameter_elements)
    if bool(cfg["training"].get("require_fp32_master_weights", True)):
        low_precision = [
            f"{name}:{parameter.dtype}"
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if low_precision:
            raise RuntimeError(
                "Full fine-tuning requires FP32 master parameters; set "
                "model.torch_dtype=float32. Low-precision tensors: " + ", ".join(low_precision[:20])
            )

    logging.info("Parsing train examples (detokenize=%s)...", bool(cfg["data"].get("detokenize", False)))
    train_dataset = SummarizationDataset(
        train_file,
        tokenizer,
        cfg["data"].get("source_prefix", ""),
        int(cfg["data"]["max_source_length"]),
        int(cfg["data"]["max_target_length"]),
        data_config=cfg["data"],
    )
    logging.info("Train examples loaded: %d", len(train_dataset))
    eval_dataset = None
    if eval_file:
        logging.info("Parsing validation examples: %s", eval_file)
        eval_dataset = SummarizationDataset(
            eval_file,
            tokenizer,
            cfg["data"].get("source_prefix", ""),
            int(cfg["data"]["max_source_length"]),
            int(cfg["data"]["max_target_length"]),
            data_config=cfg["data"],
        )
        logging.info("Validation examples loaded: %d", len(eval_dataset))

    distributed_barrier()
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    if main_process:
        log_model_summary(model, cfg, len(train_dataset), len(eval_dataset) if eval_dataset else 0)
    logging.info("Building Seq2SeqTrainingArguments and Trainer...")
    training_args = make_training_arguments(cfg, output_dir)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    distributed_barrier()
    logging.info("Trainer ready; entering training loop...")
    logging.info("Starting full fine-tuning of all T5Gemma parameters...")
    train_result = trainer.train()
    distributed_barrier()
    if main_process:
        logging.info("Training complete: %s", train_result.metrics)

        final_folder = output_dir / "final_model"
        final_folder.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_folder))
        tokenizer.save_pretrained(final_folder)
        save_resolved_config(cfg, final_folder)
        write_json(
            final_folder / "checkpoint_manifest.json",
            {
                "tag": "final",
                "global_step": int(trainer.state.global_step),
                "epoch": float(trainer.state.epoch or 0.0),
                "base_model": model_name,
                "stores_base_model_weights": True,
                "checkpoint_type": "full_finetuned_seq2seq_model",
                "trainable_ratio_percent": 100.0,
                "unique_parameter_elements": unique_parameter_elements,
                "metrics": train_result.metrics,
                "data": cfg.get("data", {}),
                "generation": cfg.get("generation", {}),
                "world_size": distributed_world_size(),
            },
        )
        write_json(output_dir / "train_metrics.json", train_result.metrics)
        running_marker.unlink(missing_ok=True)

        epochs_dir = output_dir / "epochs"
        if epochs_dir.exists():
            shutil.rmtree(epochs_dir, ignore_errors=True)
            logging.info("Deleted all epoch checkpoints after phase completion to save disk space.")
    distributed_barrier()
    destroy_distributed(distributed_initialized)


if __name__ == "__main__":
    main()
