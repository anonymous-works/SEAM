from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import load_config, validate_config
from .data import encode_prompt, left_pad_prompts, read_jsonl, record_texts
from .metrics import rouge_scores
from .train import _context_length, _load_tokenizer_and_model
from .vllm_service import VLLMClient, VLLMServer, iter_batches


def _filter_logits(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:

    filtered = logits.clone()
    if top_k > 0:
        k = min(int(top_k), filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        filtered.scatter_(
            -1,
            sorted_indices,
            sorted_logits.masked_fill(remove, -torch.inf),
        )
    return filtered


def _no_repeat_ngram_mask(logits: torch.Tensor, generated: torch.Tensor, ngram_size: int) -> torch.Tensor:

    if ngram_size <= 1 or generated.shape[1] < ngram_size:
        return logits

    ngrams = generated.unfold(dimension=1, size=ngram_size, step=1)
    suffix = generated[:, -(ngram_size - 1) :]
    matches = (ngrams[..., :-1] == suffix.unsqueeze(1)).all(dim=-1)
    rows, columns = matches.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return logits
    result = logits.clone()
    banned_tokens = ngrams[..., -1][rows, columns]
    result[rows, banned_tokens] = -torch.inf
    return result


def _next_token(
    logits: torch.Tensor,
    *,
    generation: dict[str, Any],
    generated: torch.Tensor,
    eos_token_id: int | None,
    step: int,
) -> torch.Tensor:
    scores = logits.float()
    penalty = float(generation.get("repetition_penalty", 1.0))
    if penalty != 1.0 and generated.numel():
        for row in range(scores.shape[0]):
            seen = torch.unique(generated[row])
            values = scores[row, seen]
            scores[row, seen] = torch.where(values < 0, values * penalty, values / penalty)
    scores = _no_repeat_ngram_mask(scores, generated, int(generation.get("no_repeat_ngram_size", 0)))
    if eos_token_id is not None and step < int(generation.get("min_new_tokens", 0)):
        scores[:, int(eos_token_id)] = -torch.inf
    do_sample = bool(generation.get("do_sample", False))
    if not do_sample:
        return scores.argmax(dim=-1, keepdim=True)
    temperature = float(generation.get("temperature", 0.0))
    if temperature <= 0:
        raise ValueError("Sampling requires generation.temperature > 0")
    scores = _filter_logits(
        scores / temperature, top_k=int(generation.get("top_k", 0)), top_p=float(generation.get("top_p", 1.0))
    )
    invalid = ~torch.isfinite(scores).any(dim=-1)
    if invalid.any():
        scores[invalid] = logits[invalid]
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)


@torch.inference_mode()
def generate_nemotron_ar(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    tokenizer: Any,
    generation: dict[str, Any],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] < 1 or prompt_ids.shape[1] < 1:
        raise ValueError("Nemotron AR generation expects a non-empty [batch, prompt_length] tensor")
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:
        raise RuntimeError("Nemotron AR generation requires transformers.cache_utils.DynamicCache") from exc
    for layer in getattr(model.encoder, "layers", []):
        attention = getattr(layer, "self_attn", None)
        if hasattr(attention, "diffusion_lm"):
            attention.diffusion_lm = False
    device = prompt_ids.device
    batch_size = int(prompt_ids.shape[0])
    prompt_length = int(prompt_ids.shape[1])
    padded = attention_mask is not None
    if padded:
        attention_mask = attention_mask.to(device=device)
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(prompt_ids.shape):
            raise ValueError("Nemotron attention_mask must have the same [batch, prompt_length] shape as prompt_ids")
        prompt_lengths = attention_mask.to(dtype=torch.long).sum(dim=-1)
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    else:
        prompt_lengths = None
        position_ids = None
    cache = DynamicCache()
    positions = torch.arange(prompt_length, device=device)
    prefill_kwargs: dict[str, Any] = {
        "input_ids": prompt_ids,
        "position_ids": position_ids if position_ids is not None else positions.unsqueeze(0).expand(batch_size, -1),
        "past_key_values": cache,
        "use_cache": True,
        "cache_position": positions,
    }
    if padded:
        prefill_kwargs.update({"attention_mask": attention_mask, "use_causal_mask": True})
    encoded = model.encoder(**prefill_kwargs)
    cache = encoded.past_key_values
    logits = model.diffusion_head(encoded.last_hidden_state[:, -1, :])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        raise ValueError("Nemotron AR generation requires tokenizer.pad_token_id or tokenizer.eos_token_id")
    max_new_tokens = int(generation["max_new_tokens"])
    generated = torch.empty((batch_size, max_new_tokens), dtype=torch.long, device=device)
    if padded:
        decode_attention = torch.zeros(
            (batch_size, prompt_length + max_new_tokens), dtype=attention_mask.dtype, device=device
        )
        decode_attention[:, :prompt_length] = attention_mask
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated_length = 0
    for step in range(max_new_tokens):
        token = _next_token(
            logits,
            generation=generation,
            generated=generated[:, :generated_length],
            eos_token_id=eos_token_id,
            step=step,
        )
        was_finished = finished.clone()
        if was_finished.any():
            token = token.masked_fill(was_finished.unsqueeze(1), int(pad_token_id))
        generated[:, step : step + 1] = token
        generated_length = step + 1
        if eos_token_id is not None and step + 1 >= int(generation.get("min_new_tokens", 0)):
            finished |= (~was_finished) & (token.squeeze(1) == int(eos_token_id))
        if eos_token_id is not None and bool(finished.all()):
            break
        if step + 1 >= int(generation["max_new_tokens"]):
            break
        cache_position = torch.tensor([prompt_length + step], device=device)
        step_kwargs: dict[str, Any] = {
            "input_ids": token,
            "past_key_values": cache,
            "use_cache": True,
            "cache_position": cache_position,
        }
        if padded:
            decode_attention[:, prompt_length + step] = 1
            step_kwargs.update(
                {
                    "attention_mask": decode_attention[:, : prompt_length + step + 1],
                    "position_ids": (prompt_lengths + step).unsqueeze(1),
                    "use_causal_mask": True,
                }
            )
        else:
            step_kwargs["position_ids"] = cache_position.unsqueeze(0).expand(batch_size, -1)
        encoded = model.encoder(**step_kwargs)
        cache = encoded.past_key_values
        logits = model.diffusion_head(encoded.last_hidden_state[:, -1, :])
    return torch.cat([prompt_ids, generated[:, :generated_length]], dim=1)


def _generation_kwargs(generation: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    values = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "min_new_tokens": int(generation.get("min_new_tokens", 0)),
        "num_beams": 1,
        "do_sample": bool(generation.get("do_sample", False)),
        "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
        "no_repeat_ngram_size": int(generation.get("no_repeat_ngram_size", 0)),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if values["do_sample"]:
        values.update(
            {
                "temperature": float(generation["temperature"]),
                "top_k": int(generation.get("top_k", 0)),
                "top_p": float(generation.get("top_p", 1.0)),
            }
        )
    return values


def _config_from_suite(suite_path: str | Path, model_name: str, dataset_name: str) -> dict[str, Any]:

    from .benchmark import build_run_config

    path = Path(suite_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle) or {}
    if not isinstance(suite, dict):
        raise ValueError(f"Suite config must be a mapping: {path}")
    models = suite.get("models")
    datasets = suite.get("datasets")
    if not isinstance(models, dict) or not isinstance(datasets, dict):
        raise ValueError(f"Suite config requires models and datasets mappings: {path}")
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}; available={sorted(models)}")
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset {dataset_name!r}; available={sorted(datasets)}")
    config, _ = build_run_config(suite, path, model_name, dataset_name)
    validate_config(config)
    return config


def _resolve_backend(config: dict[str, Any], requested: str) -> str:
    backend = str(requested).strip().lower()
    if backend not in {"auto", "local", "vllm"}:
        raise ValueError("Evaluation backend must be auto, local, or vllm")
    family = str(config["model"].get("family", "causal_lm"))
    if backend == "auto":
        return "local" if family == "nemotron_diffusion" else "vllm"
    if backend == "vllm" and family == "nemotron_diffusion":
        return "local"
    return backend


def _load_eval_records(
    tokenizer: Any,
    config: dict[str, Any],
    *,
    split: str,
    max_examples: int,
    context_length: int,
) -> tuple[list[dict[str, str]], list[list[int]]]:
    data = config["data"]
    configured_limit = int(config.get("limits", {}).get(f"max_{split}_examples", 0))
    limit = int(max_examples) if int(max_examples) > 0 else configured_limit
    rows = read_jsonl(data[f"{split}_file"], max_examples=limit)
    generation = config["generation"]
    records: list[dict[str, str]] = []
    prompts: list[list[int]] = []
    max_total_length = max(1, int(context_length) - int(generation["max_new_tokens"]))
    for row in rows:
        identifier, source, reference = record_texts(row, data)
        prompts.append(encode_prompt(tokenizer, source, data, max_total_length=max_total_length))
        records.append({"id": identifier, "source": source, "reference": reference})
    return records, prompts


def _write_prediction_batch(handle: Any, records: list[dict[str, str]], predictions: list[str]) -> None:
    if len(records) != len(predictions):
        raise RuntimeError(f"Prediction count mismatch: records={len(records)}, predictions={len(predictions)}")
    for record, prediction in zip(records, predictions, strict=True):
        handle.write(json.dumps({**record, "prediction": prediction.strip()}, ensure_ascii=False) + "\n")

        handle.flush()


def _log_progress(
    *,
    started: float,
    processed: int,
    total: int,
    batch_index: int,
    batch_total: int,
    every: int,
) -> None:
    if batch_index != 1 and batch_index != batch_total and batch_index % max(1, every) != 0:
        return
    elapsed = max(0.001, time.perf_counter() - started)
    rate = processed / elapsed if processed else 0.0
    remaining = max(0, total - processed)
    eta = remaining / rate if rate > 0 else float("inf")
    eta_text = "unknown" if eta == float("inf") else f"{eta / 60:.1f}m"
    percent = (100.0 * processed / total) if total else 100.0
    print(
        f"[eval] batch {batch_index}/{batch_total} | {processed}/{total} ({percent:.1f}%) | "
        f"{rate:.2f} examples/s | elapsed={elapsed / 60:.1f}m | ETA={eta_text}",
        flush=True,
    )


def _finish_metrics(
    config: dict[str, Any],
    *,
    output_path: Path,
    split: str,
    generation: dict[str, Any],
    predictions: list[str],
    references: list[str],
    backend: str,
    started: float,
    started_at: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed = max(0.0, time.perf_counter() - started)
    metrics = {
        **rouge_scores(predictions, references),
        "model_id": config["model"].get("model_id", config["model"].get("name_or_path")),
        "model_family": config["model"].get("family", "causal_lm"),
        "split": split,
        "num_examples": len(predictions),
        "predictions_file": str(output_path),
        "generation": generation,
        "eval_backend": backend,
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "examples_per_second": round(len(predictions) / elapsed, 4) if elapsed else 0.0,
        "prompt_protocol": "t5gemma_source_prefix_plus_causal_target_masking",
    }
    if extra:
        metrics.update(extra)
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[eval] COMPLETE | backend={backend} | examples={len(predictions)} | "
        f"elapsed={elapsed / 60:.1f}m | output={output_path} | finished={metrics['finished_at']}",
        flush=True,
    )
    return metrics


def _evaluate_local(
    config: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    *,
    split: str,
    max_examples: int,
    progress_every: int,
    started: float,
    started_at: str,
) -> dict[str, Any]:
    tokenizer, model = _load_tokenizer_and_model(
        {**config, "model": {**config["model"], "name_or_path": str(checkpoint_path)}}, evaluation=True
    )
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target).eval()
    records, prompts = _load_eval_records(
        tokenizer,
        config,
        split=split,
        max_examples=max_examples,
        context_length=_context_length(model),
    )
    generation = dict(config["generation"])
    batch_size = int(generation["batch_size"])
    family = str(config["model"].get("family", "causal_lm"))
    batch_indices = [
        list(range(start, min(start + batch_size, len(prompts)))) for start in range(0, len(prompts), batch_size)
    ]
    predictions: list[str] = []
    batch_total = len(batch_indices)
    batch_note = ""
    if family == "nemotron_diffusion":
        average_batch = len(prompts) / batch_total if batch_total else 0.0
        batch_note = f" | actual_batches={batch_total} | avg_actual_batch={average_batch:.2f}"
    print(
        f"[eval] START | backend=local | split={split} | examples={len(prompts)} | "
        f"batch_size={batch_size}{batch_note} | started={started_at}",
        flush=True,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        generated_count = 0
        for batch_index, indices in enumerate(batch_indices, start=1):
            batch_prompts = [prompts[index] for index in indices]
            if family == "nemotron_diffusion":
                if len({len(prompt) for prompt in batch_prompts}) == 1:
                    input_ids = torch.tensor(batch_prompts, dtype=torch.long, device=target)
                    attention = None
                else:
                    input_ids, attention = left_pad_prompts(batch_prompts, tokenizer.pad_token_id)
                    input_ids = input_ids.to(target)
                    attention = attention.to(target)
            else:
                input_ids, attention = left_pad_prompts(batch_prompts, tokenizer.pad_token_id)
                input_ids = input_ids.to(target)
                attention = attention.to(target)
            if family == "nemotron_diffusion":
                outputs = generate_nemotron_ar(
                    model,
                    input_ids,
                    tokenizer,
                    generation,
                    attention_mask=attention,
                )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention,
                    **_generation_kwargs(generation, tokenizer),
                )
            width = int(input_ids.shape[1])
            decoded = [value.strip() for value in tokenizer.batch_decode(outputs[:, width:], skip_special_tokens=True)]
            if len(decoded) != len(indices):
                raise RuntimeError(f"Prediction count mismatch: indices={len(indices)}, decoded={len(decoded)}")
            _write_prediction_batch(handle, records[indices[0] : indices[-1] + 1], decoded)
            predictions.extend(decoded)
            generated_count += len(indices)
            _log_progress(
                started=started,
                processed=generated_count,
                total=len(prompts),
                batch_index=batch_index,
                batch_total=batch_total,
                every=progress_every,
            )
    return _finish_metrics(
        config,
        output_path=output_path,
        split=split,
        generation=generation,
        predictions=predictions,
        references=[record["reference"] for record in records],
        backend="local",
        started=started,
        started_at=started_at,
        extra=(
            {
                "requested_batch_size": batch_size,
                "actual_batch_count": batch_total,
                "average_actual_batch_size": round(len(prompts) / batch_total, 4) if batch_total else 0.0,
            }
            if family == "nemotron_diffusion"
            else None
        ),
    )


def _checkpoint_context_length(checkpoint_path: Path, config: dict[str, Any]) -> int:
    configured = int(config["data"]["max_sequence_length"])
    try:
        from transformers import AutoConfig

        raw_config = AutoConfig.from_pretrained(
            str(checkpoint_path),
            local_files_only=True,
            trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
        )
        for name in ("max_position_embeddings", "max_seq_len", "n_positions", "max_sequence_length"):
            value = getattr(raw_config, name, None)
            if value is not None and int(value) > 0:
                return min(configured, int(value))
    except Exception as exc:
        print(f"[eval] warning: could not read model context length ({exc}); using config={configured}", flush=True)
    return configured


def _vllm_dtype(config: dict[str, Any]) -> str:
    value = str(config["model"].get("eval_torch_dtype", "bfloat16")).lower()
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(value, value)


def _evaluate_vllm(
    config: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    *,
    split: str,
    max_examples: int,
    progress_every: int,
    vllm_base_url: str,
    vllm_model: str | None,
    vllm_batch_size: int,
    start_vllm_service: bool,
    vllm_startup_timeout: float,
    started: float,
    started_at: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    common = {
        "local_files_only": True,
        "trust_remote_code": bool(config["model"].get("trust_remote_code", True)),
    }
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), **common)
    context_length = _checkpoint_context_length(checkpoint_path, config)
    records, prompts = _load_eval_records(
        tokenizer,
        config,
        split=split,
        max_examples=max_examples,
        context_length=context_length,
    )
    generation = dict(config["generation"])
    batch_size = int(vllm_batch_size) if int(vllm_batch_size) > 0 else int(generation["batch_size"])
    batch_total = (len(prompts) + batch_size - 1) // batch_size if prompts else 0
    print(
        f"[eval] START | backend=vllm | split={split} | examples={len(prompts)} | "
        f"request_batch_size={batch_size} | started={started_at} | base_url={vllm_base_url}",
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    server: VLLMServer | None = None
    try:
        if start_vllm_service:
            server = VLLMServer.start(
                checkpoint_path,
                base_url=vllm_base_url,
                max_model_len=context_length,
                dtype=_vllm_dtype(config),
                trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
                startup_timeout=vllm_startup_timeout,
                log_path=output_path.with_name(f"{output_path.stem}.vllm.log"),
            )
            client = server.client
        else:
            client = VLLMClient(
                vllm_base_url,
                api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                timeout=max(30.0, float(vllm_startup_timeout)),
            )
            discovered = client.wait_until_ready(timeout_seconds=vllm_startup_timeout)
            print(f"[vllm] using existing service: url={client.base_url} model={discovered}", flush=True)
        served_model = client.model_name(vllm_model)
        predictions: list[str] = []
        with output_path.open("w", encoding="utf-8") as handle:
            for batch_index, batch_prompts in enumerate(iter_batches(prompts, batch_size), start=1):
                decoded = client.complete(batch_prompts, generation, model=served_model)
                predictions.extend(decoded)
                start = (batch_index - 1) * batch_size
                _write_prediction_batch(handle, records[start : start + len(decoded)], decoded)
                _log_progress(
                    started=started,
                    processed=len(predictions),
                    total=len(prompts),
                    batch_index=batch_index,
                    batch_total=batch_total,
                    every=progress_every,
                )
        return _finish_metrics(
            config,
            output_path=output_path,
            split=split,
            generation=generation,
            predictions=predictions,
            references=[record["reference"] for record in records],
            backend="vllm",
            started=started,
            started_at=started_at,
            extra={
                "vllm_base_url": client.base_url,
                "vllm_model": served_model,
                "vllm_request_batch_size": batch_size,
                "vllm_unsupported_generation_controls": ["no_repeat_ngram_size"]
                if int(generation.get("no_repeat_ngram_size", 0)) > 0
                else [],
            },
        )
    finally:
        if server is not None:
            server.stop()


def evaluate(
    config_path: str | Path | dict[str, Any],
    checkpoint: str | Path,
    output: str | Path,
    *,
    split: str = "test",
    max_examples: int = 0,
    backend: str = "auto",
    vllm_base_url: str | None = None,
    vllm_model: str | None = None,
    vllm_batch_size: int = 0,
    start_vllm_service: bool | None = None,
    vllm_startup_timeout: float = 900.0,
    progress_every: int = 10,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    config = config_path if isinstance(config_path, dict) else load_config(config_path)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_path}")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.touch(exist_ok=True)
    selected_backend = _resolve_backend(config, backend)
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(
        f"[eval] initializing | backend={selected_backend} | checkpoint={checkpoint_path} | "
        f"output={output_path} | started={started_at}",
        flush=True,
    )
    if (
        selected_backend == "local"
        and str(config["model"].get("family", "causal_lm")) == "nemotron_diffusion"
        and str(backend).strip().lower() == "vllm"
    ):
        print("[eval] Nemotron uses native local AR generation; ignoring the stale vLLM backend flag", flush=True)
    if selected_backend == "local":
        return _evaluate_local(
            config,
            checkpoint_path,
            output_path,
            split=split,
            max_examples=max_examples,
            progress_every=progress_every,
            started=started,
            started_at=started_at,
        )
    if start_vllm_service is None:
        start_vllm_service = not bool(os.environ.get("VLLM_BASE_URL"))
    return _evaluate_vllm(
        config,
        checkpoint_path,
        output_path,
        split=split,
        max_examples=max_examples,
        progress_every=progress_every,
        vllm_base_url=vllm_base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        vllm_model=vllm_model or os.environ.get("VLLM_MODEL"),
        vllm_batch_size=vllm_batch_size,
        start_vllm_service=bool(start_vllm_service),
        vllm_startup_timeout=vllm_startup_timeout,
        started=started,
        started_at=started_at,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one decoder-only baseline checkpoint")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Materialized per-run YAML config")
    config_group.add_argument("--suite", help="Decoder-only benchmark suite YAML")
    parser.add_argument("--model", help="Model key from --suite, for example qwen3_4b")
    parser.add_argument("--dataset", help="Dataset key from --suite, for example pubmed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--backend",
        "--eval-backend",
        dest="backend",
        choices=("auto", "vllm", "local"),
        default=os.environ.get("DECODER_EVAL_BACKEND", "auto"),
        help="auto uses vLLM for standard decoder-only LMs and local native AR generation for Nemotron; use local to force in-process Transformers",
    )
    parser.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--vllm-batch-size", type=int, default=int(os.environ.get("VLLM_BATCH_SIZE", "0")))
    parser.add_argument("--vllm-startup-timeout", type=float, default=900.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--start-vllm-service",
        dest="start_vllm_service",
        action="store_true",
        help="Start vllm serve from --checkpoint and stop it after evaluation",
    )
    parser.add_argument(
        "--no-start-vllm-service",
        dest="start_vllm_service",
        action="store_false",
        help="Use an already running VLLM_BASE_URL service",
    )
    parser.set_defaults(start_vllm_service=not bool(os.environ.get("VLLM_BASE_URL")))
    args = parser.parse_args()
    if args.suite and (not args.model or not args.dataset):
        parser.error("--suite requires both --model and --dataset")
    if args.suite:
        config: str | Path | dict[str, Any] = _config_from_suite(args.suite, args.model, args.dataset)
    else:
        config = args.config
    print(
        json.dumps(
            evaluate(
                config,
                args.checkpoint,
                args.output,
                split=args.split,
                max_examples=args.max_examples,
                backend=args.backend,
                vllm_base_url=args.vllm_base_url,
                vllm_model=args.vllm_model,
                vllm_batch_size=args.vllm_batch_size,
                start_vllm_service=args.start_vllm_service,
                vllm_startup_timeout=args.vllm_startup_timeout,
                progress_every=args.progress_every,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
