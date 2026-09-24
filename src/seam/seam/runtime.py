from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from .config import load_config, resolve_path
from .data.collate import SummarizationCollator
from .data.copy_alignment import COPY_INPUT_KEYS
from .data.dataset import JsonlSummarizationDataset
from .data.sampling import DistributedBatchSampler, LengthBucketBatchSampler
from .modeling.model import SEAMModel
from .training.checkpoint import load_checkpoint
from .training.engine import SEAMTrainer, seed_everything

LOGGER = logging.getLogger("seam.runtime")


def _configure_precision(config: dict[str, Any]) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("tf32", False))


def _distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _distributed_rank() -> int:
    return dist.get_rank() if _distributed_active() else 0


def _distributed_world_size() -> int:
    return dist.get_world_size() if _distributed_active() else 1


def _is_main_process() -> bool:
    return _distributed_rank() == 0


def _barrier() -> None:
    if _distributed_active():
        dist.barrier()


def _init_distributed(device: str | None) -> tuple[torch.device, bool]:

    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if requested_world_size <= 1:
        return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")), False
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA; launch without torchrun for a CPU run")
    if device is not None and str(device).startswith("cpu"):
        raise ValueError("--device=cpu cannot be used with a multi-process CUDA run")
    local_rank = int(os.environ.get("LOCAL_RANK", _distributed_rank()))
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside the visible CUDA devices (count={torch.cuda.device_count()})"
        )
    torch.cuda.set_device(local_rank)
    initialized_here = False
    if not _distributed_active():
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
    return torch.device("cuda", local_rank), initialized_here


def _destroy_distributed(initialized_here: bool) -> None:
    if initialized_here and _distributed_active():
        dist.destroy_process_group()


class _TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping=False,
        truncation=False,
        max_length=None,
    ) -> dict:
        matches = list(re.finditer(r"\S+", str(text)))
        tokens = [match.group() for match in matches]
        offsets = [(match.start(), match.end()) for match in matches]
        ids = []
        for token in tokens:
            value = sum((position + 1) * ord(character) for position, character in enumerate(token))
            ids.append(4 + value % 124)
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
            offsets = [(0, 0), *offsets, (0, 0)]
        if truncation and max_length is not None:
            ids, offsets = ids[:max_length], offsets[:max_length]
        return {"input_ids": ids, **({"offset_mapping": offsets} if return_offsets_mapping else {})}

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = True) -> list[str]:
        rows = sequences.tolist() if isinstance(sequences, torch.Tensor) else sequences
        output = []
        for row in rows:
            tokens = []
            for value in row:
                value = int(value)
                if skip_special_tokens and value in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
                    continue
                tokens.append(f"tok{value}")
            output.append(" ".join(tokens))
        return output


def _tokenizers(config: dict[str, Any]):
    model = config["model"]
    kwargs = {
        "use_fast": bool(model.get("tokenizer_use_fast", True)),
        "trust_remote_code": bool(model.get("trust_remote_code", True)),
    }
    from transformers import AutoTokenizer

    encoder = (
        _TinyTokenizer()
        if str(model["encoder_name"]) == "__tiny__"
        else AutoTokenizer.from_pretrained(str(model["encoder_name"]), **kwargs)
    )
    decoder = (
        _TinyTokenizer()
        if str(model["decoder_name"]) == "__tiny__"
        else AutoTokenizer.from_pretrained(str(model["decoder_name"]), **kwargs)
    )
    if decoder.pad_token_id is None:
        decoder.pad_token = decoder.eos_token or decoder.unk_token
    if encoder.pad_token_id is None:
        encoder.pad_token = encoder.eos_token or encoder.unk_token
    return encoder, decoder


def build_loaders(
    config: dict[str, Any],
    split: str | None = None,
    batch_size_override: int | None = None,
    max_train_examples: int = 0,
    max_validation_examples: int = 0,
):
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    data = config["data"]
    paths = {
        "train": data["train_file"],
        "validation": data.get("validation_file"),
        "test": data.get("test_file"),
    }
    selected = (split,) if split else ("train", "validation")
    loaders = {}
    distributed = _distributed_active()
    rank = _distributed_rank()
    world_size = _distributed_world_size()
    for name in selected:
        if not paths[name]:
            raise ValueError(f"data.{name}_file is required to load the {name} split")
        split_data = copy.deepcopy(data)
        limit = max_train_examples if name == "train" else max_validation_examples if name == "validation" else 0
        dataset = JsonlSummarizationDataset(resolve_path(paths[name], config), split_data, max_examples=limit)
        collator = SummarizationCollator(
            encoder_tokenizer,
            decoder_tokenizer,
            split_data,
            source_copy=bool(config["decoder"].get("source_copy", {}).get("enabled", False)),
        )
        batch_size = (
            int(config["training"].get("validation_batch_size", 4))
            if name != "train"
            else int(config["training"].get("batch_size", 4))
        )
        if batch_size_override is not None:
            batch_size = int(batch_size_override)
        workers = (
            int(config["training"].get("num_workers", 0))
            if name == "train"
            else int(config["training"].get("validation_num_workers", 0))
        )
        if distributed:
            multiplier = (
                int(config["training"].get("length_bucket_multiplier", 50))
                if config["training"].get("length_bucketing", False)
                else 1
            )
            sampling = {
                "batch_sampler": DistributedBatchSampler(
                    dataset.length_estimates,
                    batch_size,
                    world_size,
                    rank,
                    seed=int(config["training"].get("seed", 42)),
                    multiplier=multiplier,
                    shuffle=name == "train",
                )
            }
        else:
            sampling = {"batch_size": batch_size, "shuffle": name == "train"}
            if name == "train" and config["training"].get("length_bucketing", False):
                sampling = {
                    "batch_sampler": LengthBucketBatchSampler(
                        dataset.length_estimates,
                        batch_size,
                        int(config["training"].get("seed", 42)),
                        int(config["training"].get("length_bucket_multiplier", 50)),
                    )
                }
        loaders[name] = DataLoader(
            dataset,
            **sampling,
            num_workers=workers,
            persistent_workers=workers > 0 and bool(config["training"].get("persistent_workers", True)),
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        if _is_main_process():
            LOGGER.info(
                "[data] split=%s | examples=%d | batch=%d | workers=%d | length_bucketing=%s | rank=%d/%d",
                name,
                len(dataset),
                batch_size,
                workers,
                "batch_sampler" in sampling,
                rank,
                world_size,
            )
    return loaders


def _write_resolved_config(config: dict[str, Any], output_dir: Path) -> None:
    resolved = copy.deepcopy(config)
    resolved.pop("_meta", None)
    for key in ("train_file", "validation_file", "test_file"):
        if resolved["data"].get(key):
            resolved["data"][key] = str(resolve_path(resolved["data"][key], config))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)


def _clear_run_artifacts(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("*.pt", "*.jsonl", "*.json", "resolved_config.yaml"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def train(
    config_path: str | Path,
    *,
    device: str | None = None,
    resume_checkpoint: str | None = None,
    train_file: str | Path | None = None,
    train_only: bool = False,
    max_train_examples: int = 0,
    max_validation_examples: int = 0,
    overwrite_output_dir: bool = False,
    output_dir_override: str | None = None,
) -> None:
    selected_device, initialized_here = _init_distributed(device)
    try:
        config = load_config(config_path, train_only=train_only)
        config["model"]["dtype"] = "float32"
        config["model"].setdefault("compute_dtype", "bfloat16")
        _configure_precision(config)
        if output_dir_override:
            config["experiment"]["output_dir"] = output_dir_override
        if train_file is not None:
            config["data"]["train_file"] = str(Path(train_file).expanduser().resolve())
        checkpoint = resume_checkpoint or str(config["training"].get("resume_checkpoint", "")).strip()
        if overwrite_output_dir and checkpoint:
            raise ValueError("--overwrite-output-dir cannot be combined with --resume-checkpoint")
        output_dir = resolve_path(config["experiment"]["output_dir"], config)
        config["experiment"]["output_dir"] = str(output_dir)
        if not overwrite_output_dir and not checkpoint and output_dir.exists() and any(output_dir.glob("*.pt")):
            raise FileExistsError(
                f"Existing checkpoints in {output_dir}; resume or explicitly use --overwrite-output-dir"
            )
        if _is_main_process():
            if overwrite_output_dir:
                _clear_run_artifacts(output_dir)
        _barrier()
        seed_everything(int(config["training"].get("seed", 42)) + _distributed_rank())
        if _is_main_process():
            _write_resolved_config(config, output_dir)
        _barrier()
        loaders = build_loaders(
            config,
            split="train" if train_only else None,
            max_train_examples=max_train_examples,
            max_validation_examples=max_validation_examples,
        )
        model = SEAMModel(config).to(device=selected_device, dtype=torch.float32)
        counts = {
            name: sum(p.numel() for p in module.parameters())
            for name, module in (("encoder", model.encoder), ("bridge", model.bridge), ("decoder", model.decoder))
        }
        if _is_main_process():
            LOGGER.info(
                "model parameters=%s total=%d | distributed=%s world_size=%d",
                counts,
                sum(p.numel() for p in model.parameters()),
                _distributed_active(),
                _distributed_world_size(),
            )
        if _distributed_active():
            model = DistributedDataParallel(
                model,
                device_ids=[selected_device.index],
                output_device=selected_device.index,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        trainer = SEAMTrainer(model, config, selected_device)
        if checkpoint and _is_main_process():
            LOGGER.info("resumed SEAM checkpoint: %s", checkpoint)
        trainer.fit(
            loaders["train"],
            None if train_only else loaders.get("validation"),
            resume_checkpoint=checkpoint or None,
        )
        _barrier()
    finally:
        _destroy_distributed(initialized_here)


def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "test",
    batch_size: int | None = None,
    device: str | None = None,
    max_examples: int = 0,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    shard_rank: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "Distributed evaluation is not enabled; run evaluation once on one visible GPU after DDP training"
        )
    if num_shards <= 0 or not 0 <= shard_rank < num_shards:
        raise ValueError("shard_rank must lie in [0, num_shards) and num_shards must be positive")
    config = load_config(config_path)
    LOGGER.info(
        "[model] config=%s | encoder=%s | decoder=%s",
        config.get("_meta", {}).get("config_path", config_path),
        config["model"]["encoder_name"],
        config["model"]["decoder_name"],
    )
    _configure_precision(config)
    selected_batch_size = int(batch_size if batch_size is not None else config["generation"]["batch_size"])
    if selected_batch_size <= 0:
        raise ValueError("Evaluation batch size must be positive")
    loaders = build_loaders(
        config,
        split=split,
        batch_size_override=selected_batch_size,
    )
    from .evaluation.generate import append_jsonl, generate_greedy, generate_sampled

    loader = loaders[split]
    loader.collate_fn.include_targets = False
    active_indices = list(range(int(shard_rank), len(loader.dataset), int(num_shards)))
    active_dataset = loader.dataset if num_shards == 1 else Subset(loader.dataset, active_indices)
    seen = set()
    predictions: list[str] = []
    references: list[str] = []
    started = time.monotonic()
    output_file = Path(output_path)
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                index = len(predictions)
                if index >= len(active_dataset):
                    raise ValueError("Resume file contains more rows than the active split")
                expected = active_dataset[index]
                example_id = str(row.get("id", ""))
                if example_id in seen or example_id != expected.example_id or row.get("reference") != expected.target:
                    raise ValueError(f"Evaluation resume must match the exact ID/reference prefix; mismatch at {index}")
                if num_shards > 1 and int(row.get("index", -1)) != active_indices[index]:
                    raise ValueError(f"Evaluation shard index mismatch at local row {index}")
                if not isinstance(row.get("prediction"), str):
                    raise ValueError(f"Missing prediction at resume row {index}")
                seen.add(example_id)
                predictions.append(row["prediction"])
                references.append(row["reference"])
    resumed_count = len(predictions)
    total = min(len(active_dataset), max_examples) if max_examples > 0 else len(active_dataset)
    if resumed_count > total:
        raise ValueError("Resume file exceeds requested max_examples")
    LOGGER.info(
        "resuming evaluation: %d/%d predictions already present; batch_size=%d",
        resumed_count,
        total,
        selected_batch_size,
    )
    if resumed_count == total:
        from .evaluation.metrics import summarization_metrics

        result = summarization_metrics(predictions, references)
        Path(str(output_path) + ".metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "[evaluation] complete | split=%s | examples=%d | ROUGE-1=%.3f | ROUGE-2=%.3f | ROUGE-L=%.3f",
            split,
            total,
            result["rouge1"],
            result["rouge2"],
            result["rougeL"],
        )
        return result
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inference_config = copy.deepcopy(config)
    if device_obj.type == "cuda":
        inference_config["model"]["dtype"] = config["model"].get(
            "compute_dtype", config["model"].get("dtype", "float32")
        )
    model = SEAMModel(inference_config).to(device_obj)
    load_checkpoint(checkpoint_path, model, config=config, restore_rng=False)
    model.eval()
    generation = config["generation"]
    requested_do_sample = bool(generation.get("do_sample", False) if do_sample is None else do_sample)
    requested_temperature = float(generation.get("temperature", 1.0) if temperature is None else temperature)
    requested_top_k = int(generation.get("top_k", 0) if top_k is None else top_k)
    requested_top_p = float(generation.get("top_p", 1.0) if top_p is None else top_p)
    if requested_do_sample and resumed_count:
        raise ValueError("Sampling evaluation requires a fresh output JSONL; do not resume a greedy prefix")
    generator = None
    if requested_do_sample:
        generator = torch.Generator(device=device_obj).manual_seed(int(config["training"].get("seed", 42)))
    loader = DataLoader(
        Subset(active_dataset, range(resumed_count, total)),
        batch_size=selected_batch_size,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=device_obj.type == "cuda",
    )
    started = time.monotonic()
    generated_this_run = 0
    pending_batches = []
    for batch in loader:
        pending_batches.append(batch)
        while pending_batches:
            batch = pending_batches.pop(0)
            if len(set(batch["ids"])) != len(batch["ids"]) or any(str(value) in seen for value in batch["ids"]):
                raise ValueError("Evaluation requires unique example IDs; repeated ID in active split")
            device_obj = next(model.parameters()).device
            narrowed = {
                key: value.to(device_obj, non_blocking=True) if isinstance(value, torch.Tensor) else list(value)
                for key, value in batch.items()
                if key
                in {
                    "input_ids",
                    "attention_mask",
                    "source_content_mask",
                    "decoder_prompt_ids",
                    "decoder_prompt_mask",
                    "ids",
                    "references",
                    *COPY_INPUT_KEYS,
                }
            }
            try:
                generate = generate_sampled if requested_do_sample else generate_greedy
                texts, _ = generate(
                    model,
                    narrowed,
                    loaders[split].collate_fn.decoder_tokenizer,
                    int(config["generation"]["max_new_tokens"]),
                    int(config["generation"].get("min_new_tokens", 0)),
                    float(config["generation"].get("repetition_penalty", 1.0)),
                    int(config["generation"].get("no_repeat_ngram_size", 0)),
                    bool(config["generation"].get("compact_finished", True)),
                    **(
                        {
                            "temperature": requested_temperature,
                            "top_k": requested_top_k,
                            "top_p": requested_top_p,
                            "generator": generator,
                        }
                        if requested_do_sample
                        else {}
                    ),
                )
            except torch.cuda.OutOfMemoryError:
                size = len(narrowed["ids"])
                if device_obj.type != "cuda" or size <= 1:
                    raise
                half = size // 2
                LOGGER.warning(
                    "[evaluation] CUDA OOM | retrying batch=%d as %d+%d",
                    size,
                    half,
                    size - half,
                )
                torch.cuda.empty_cache()
                left = {
                    key: value[:half].detach().cpu() if isinstance(value, torch.Tensor) else value[:half]
                    for key, value in narrowed.items()
                }
                right = {
                    key: value[half:].detach().cpu() if isinstance(value, torch.Tensor) else value[half:]
                    for key, value in narrowed.items()
                }
                del narrowed, batch
                torch.cuda.empty_cache()
                pending_batches[0:0] = [left, right]
                continue
            row_start = resumed_count + generated_this_run
            row_indices = active_indices[row_start : row_start + len(texts)]
            if len(row_indices) != len(texts):
                raise RuntimeError("Evaluation shard produced more rows than its active dataset")
            append_jsonl(
                output_path,
                (
                    {
                        "id": example_id,
                        "prediction": text,
                        "reference": reference,
                        **({"index": global_index} if num_shards > 1 else {}),
                    }
                    for global_index, example_id, text, reference in zip(
                        row_indices, narrowed["ids"], texts, narrowed["references"]
                    )
                ),
            )
            seen.update(str(value) for value in narrowed["ids"])
            predictions.extend(texts)
            references.extend(narrowed["references"])
            generated_this_run += len(texts)
            processed = resumed_count + generated_this_run
            elapsed = time.monotonic() - started
            eta = elapsed * max(0, total - processed) / max(1, generated_this_run)
            peak = torch.cuda.max_memory_allocated(device_obj) / 2**30 if device_obj.type == "cuda" else 0
            message = (
                "[evaluation] split=%s | processed=%d/%d | %.2f ex/s | elapsed=%.1fs | ETA=%.1fs | peak=%.2f GiB"
                % (
                    split,
                    processed,
                    total,
                    generated_this_run / max(elapsed, 1e-6),
                    elapsed,
                    eta,
                    peak,
                )
            )
            LOGGER.info(message)
            print(message, flush=True)
    from .evaluation.metrics import summarization_metrics

    result = summarization_metrics(predictions, references)
    Path(str(output_path) + ".metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "[evaluation] complete | split=%s | examples=%d | ROUGE-1=%.3f | ROUGE-2=%.3f | ROUGE-L=%.3f",
        split,
        total,
        result["rouge1"],
        result["rouge2"],
        result["rougeL"],
    )
    return result
