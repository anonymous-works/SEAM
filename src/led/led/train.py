from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer

from led.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from t5gemma2.data import load_summarization_jsonl


class LEDSummarizationDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, data: dict[str, Any]) -> None:
        self.records = load_summarization_jsonl(path, data)
        self.tokenizer = tokenizer
        self.prefix = str(data.get("source_prefix", ""))
        self.source_limit = int(data["max_source_length"])
        self.target_limit = int(data["max_target_length"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        encoded = self.tokenizer(
            self.prefix + row.source,
            max_length=self.source_limit,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=row.target,
            max_length=max(1, self.target_limit - 1),
            truncation=True,
        )["input_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and (not labels or labels[-1] != eos_token_id):
            labels.append(eos_token_id)
        encoded["labels"] = labels
        return encoded


class LEDDataCollator:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.base = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.base(features)
        attention_mask = batch["attention_mask"]
        first_visible = attention_mask.to(torch.int64).argmax(dim=1)
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask.scatter_(1, first_visible.unsqueeze(1), 1)
        batch["global_attention_mask"] = global_attention_mask
        return batch


def _dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}
    try:
        return mapping[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported LED dtype: {name}") from exc


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _arguments(config: dict[str, Any], output_dir: Path) -> Any:
    from transformers import Seq2SeqTrainingArguments

    train = config["training"]
    values = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": int(train["num_train_epochs"]),
        "per_device_train_batch_size": int(train["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(train["gradient_accumulation_steps"]),
        "learning_rate": float(train["learning_rate"]),
        "adam_beta1": float(train["adam_beta1"]),
        "adam_beta2": float(train["adam_beta2"]),
        "adam_epsilon": float(train["adam_epsilon"]),
        "warmup_ratio": float(train["warmup_ratio"]),
        "weight_decay": float(train["weight_decay"]),
        "max_grad_norm": float(train["max_grad_norm"]),
        "lr_scheduler_type": str(train["lr_scheduler_type"]),
        "optim": str(train["optim"]),
        "bf16": bool(train["bf16"]),
        "fp16": bool(train["fp16"]),
        "tf32": bool(train["tf32"]),
        "gradient_checkpointing": bool(train["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "save_safetensors": True,
        "logging_strategy": "steps",
        "logging_steps": int(train["logging_steps"]),
        "report_to": [],
        "predict_with_generate": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": int(train["dataloader_num_workers"]),
        "dataloader_pin_memory": True,
        "seed": int(train["seed"]),
        "data_seed": int(train["seed"]),
        "ddp_find_unused_parameters": False,
    }
    parameters = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    values = {key: value for key, value in values.items() if key in parameters}
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        values["evaluation_strategy"] = "epoch"
    return Seq2SeqTrainingArguments(**values)


def train(config: dict[str, Any], *, overwrite_output_dir: bool = False) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The reported LED recipe requires a CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("The reported LED recipe uses one GPU; launch one process only")

    output_dir = Path(config["run"]["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite_output_dir:
        raise FileExistsError(f"Run directory is not empty: {output_dir}; choose another or pass --overwrite-output-dir")
    if overwrite_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(config["model"]["name_or_path"])
    model_path = Path(model_name).expanduser()
    if model_path.exists() and not model_path.is_dir():
        raise ValueError(f"LED_MODEL_PATH must point to a model directory: {model_path}")
    if model_path.is_absolute() and not model_path.is_dir():
        raise FileNotFoundError(f"Local LED checkpoint not found: {model_path}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _set_seed(int(config["training"]["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"]["tf32"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        local_files_only=True,
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        torch_dtype=_dtype(config["model"]["torch_dtype"]),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    train_data = LEDSummarizationDataset(config["data"]["train_file"], tokenizer, config["data"])
    valid_data = LEDSummarizationDataset(config["data"]["validation_file"], tokenizer, config["data"])
    trainer = Seq2SeqTrainer(
        model=model,
        args=_arguments(config, output_dir),
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=LEDDataCollator(tokenizer, model),
        processing_class=tokenizer,
    )

    resolved = output_dir / "run_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    result = trainer.train()
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(best_dir)
    best_checkpoint = trainer.state.best_model_checkpoint
    try:
        best_step = int(Path(best_checkpoint).name.rsplit("-", 1)[1]) if best_checkpoint else None
    except (IndexError, ValueError):
        best_step = None
    manifest = {
        "base_model": model_name,
        "checkpoint_type": "full_finetuned_seq2seq_model",
        "selection": "minimum epoch-end eval_loss",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "best_global_step": best_step,
        "epochs_requested": config["training"]["num_train_epochs"],
        "effective_batch_size": config["training"]["effective_batch_size"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "git_revision": _git_revision(),
        "train_metrics": result.metrics,
    }
    (output_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return best_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Full fine-tuning for the LED-large-16k summarization baseline")
    parser.add_argument("--dataset", required=True, choices=("pubmed", "arxiv", "booksum", "govreport"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-train-epochs", type=int, required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    config = load_config(args.dataset, config_path=args.config, num_train_epochs=args.num_train_epochs)
    best = train(config, overwrite_output_dir=args.overwrite_output_dir)
    print(f"Selected LED checkpoint: {best}")


if __name__ == "__main__":
    main()
