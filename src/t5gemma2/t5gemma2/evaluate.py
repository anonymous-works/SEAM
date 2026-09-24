from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import load_summarization_jsonl, resolve_path
from .metrics import heter_sum_graph_rouge

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


def torch_dtype_from_config(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if lowered in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def word_count(text: str) -> int:
    return len(text.split())


def repeated_ngram_rate(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.ceil((pct / 100.0) * len(sorted_values)) - 1))
    return float(sorted_values[idx])


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_metrics(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    compute_bertscore: bool = False,
    bertscore_model_type: str = "bert-base-multilingual-cased",
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = heter_sum_graph_rouge(
        predictions,
        references,
        include_rouge_lsum=True,
    )
    metrics["rouge_backend"] = "rouge==1.0.0 (HeterSumGraph)"
    metrics["rouge_preprocessing"] = "NFC + lowercase + stored whitespace tokenization"
    bleu = corpus_bleu(predictions, [references])
    chrf = corpus_chrf(predictions, [references])
    metrics.update(
        {
            "bleu": round(bleu.score, 4),
            "bleu_bp": round(bleu.bp, 6),
            "chrf": round(chrf.score, 4),
            "num_examples": len(predictions),
        }
    )

    pred_words = [word_count(pred) for pred in predictions]
    ref_words = [word_count(ref) for ref in references]
    length_ratios = [pred_len / max(1, ref_len) for pred_len, ref_len in zip(pred_words, ref_words)]
    repeat_rates = [repeated_ngram_rate(pred, n=3) for pred in predictions]
    normalized_predictions = [" ".join(pred.lower().split()) for pred in predictions]
    nonempty_predictions = [pred for pred in normalized_predictions if pred]
    prefixes = Counter(" ".join(pred.split()[:5]) for pred in nonempty_predictions)
    metrics.update(
        {
            "prediction_words_mean": round(safe_mean(pred_words), 4),
            "reference_words_mean": round(safe_mean(ref_words), 4),
            "length_ratio_mean": round(safe_mean(length_ratios), 6),
            "empty_prediction_rate": round(
                100.0 * safe_mean([1.0 if not pred.strip() else 0.0 for pred in predictions]), 4
            ),
            "too_short_rate": round(100.0 * safe_mean([1.0 if ratio < 0.5 else 0.0 for ratio in length_ratios]), 4),
            "too_long_rate": round(100.0 * safe_mean([1.0 if ratio > 1.5 else 0.0 for ratio in length_ratios]), 4),
            "repeated_trigram_rate_mean": round(100.0 * safe_mean(repeat_rates), 4),
            "unique_prediction_rate": round(
                100.0 * len(set(nonempty_predictions)) / max(1, len(nonempty_predictions)),
                4,
            ),
            "dominant_prefix_5gram_rate": round(
                100.0 * max(prefixes.values(), default=0) / max(1, len(nonempty_predictions)),
                4,
            ),
        }
    )

    if sources is not None:
        source_words = [word_count(src) for src in sources]
        compression_ratios = [pred_len / max(1, src_len) for pred_len, src_len in zip(pred_words, source_words)]
        metrics.update(
            {
                "source_words_mean": round(safe_mean(source_words), 4),
                "compression_ratio_mean": round(safe_mean(compression_ratios), 6),
            }
        )

    if compute_bertscore:
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(
            predictions,
            references,
            model_type=bertscore_model_type,
            verbose=True,
        )
        metrics.update(
            {
                "bertscore_model_type": bertscore_model_type,
                "bertscore_precision": round(float(precision.mean().item()) * 100.0, 4),
                "bertscore_recall": round(float(recall.mean().item()) * 100.0, 4),
                "bertscore_f1": round(float(f1.mean().item()) * 100.0, 4),
            }
        )
    return metrics


def generation_value(args: argparse.Namespace, raw_cfg: Dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return raw_cfg.get("generation", {}).get(name, default)


def resolve_checkpoint_source(
    raw_cfg: Dict[str, Any], checkpoint: Optional[str], *, base: Optional[Path] = None
) -> Tuple[str, Optional[str], str]:
    local_candidates: List[Path] = []
    if checkpoint:
        explicit_path = Path(checkpoint).expanduser()
        if explicit_path.is_absolute() and explicit_path.exists():
            return str(explicit_path.resolve()), None, str(explicit_path.resolve())
        if not explicit_path.is_absolute():
            candidates = [(Path.cwd() / explicit_path).resolve()]
            if base is not None:
                candidates.append((base / explicit_path).resolve())
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate), None, str(candidate)

        raise FileNotFoundError(f"Full checkpoint not found: {checkpoint}")
    output_dir = resolve_path(raw_cfg["project"]["output_dir"], base=base or T5GEMMA_ROOT.parents[1])
    local_candidates.append(output_dir / "final_model")
    for path in local_candidates:
        if path.exists():
            return str(path), None, str(path)

    raise FileNotFoundError(f"No local full checkpoint found in {[str(p) for p in local_candidates]}")


def maybe_upload_eval_outputs(raw_cfg: Dict[str, Any], output_dir: Path) -> None:
    del output_dir
    hf_cfg = raw_cfg.get("huggingface", {})
    if not bool(hf_cfg.get("enabled", False)):
        return
    raise RuntimeError("Hugging Face uploads are hard-disabled; set huggingface.enabled=false")


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--output_dir", default="src/t5gemma2/eval_outputs/full_test")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--compute_bertscore", action="store_true")
    parser.add_argument("--bertscore_model_type", default=None)
    args = parser.parse_args()

    project_root = T5GEMMA_ROOT.parents[1]
    config_path = resolve_path(args.config, base=project_root)
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    test_file = resolve_path(
        args.test_file or raw_cfg["data"].get("test_file", "src/seam/datasets/test.jsonl"),
        base=project_root,
    )
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    output_dir = resolve_path(args.output_dir, base=project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    run_info_path = output_dir / "eval_run_info.json"
    print(f"Prediction output: {predictions_path.resolve()}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = raw_cfg["model"]["model_name_or_path"]
    trust_remote_code = bool(raw_cfg["model"].get("trust_remote_code", True))
    dtype = torch_dtype_from_config(
        raw_cfg["model"].get(
            "eval_torch_dtype",
            raw_cfg["model"].get("torch_dtype", "bfloat16"),
        )
    )

    checkpoint_source, checkpoint_subfolder, checkpoint_label = resolve_checkpoint_source(
        raw_cfg, args.checkpoint, base=project_root
    )
    local_checkpoint = Path(checkpoint_source)
    checkpoint_manifest: Dict[str, Any] = {}
    if local_checkpoint.exists():
        running_marker = local_checkpoint.parent / "RUNNING"
        if running_marker.exists():
            raise RuntimeError(f"Refusing to evaluate {local_checkpoint}: {running_marker} indicates an incomplete run")
        checkpoint_manifest_path = local_checkpoint / "checkpoint_manifest.json"
        if checkpoint_manifest_path.exists():
            checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    print(f"Loading full fine-tuned checkpoint: {checkpoint_label}")
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": True,
    }

    if checkpoint_subfolder is not None:
        load_kwargs["subfolder"] = checkpoint_subfolder
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_source,
        **load_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_source,
        dtype=dtype,
        **load_kwargs,
    )
    model.config.use_cache = bool(raw_cfg["model"].get("use_cache_for_eval", True))
    model.to(device)
    model.eval()

    unique_parameter_elements = int(sum(parameter.numel() for parameter in model.parameters()))
    checkpoint_parameter_elements = checkpoint_manifest.get("unique_parameter_elements")
    checkpoint_parameters_match_model = (
        int(checkpoint_parameter_elements) == unique_parameter_elements
        if checkpoint_parameter_elements is not None
        else None
    )

    examples = load_summarization_jsonl(test_file, raw_cfg["data"], limit=args.limit)
    batch_size = int(args.batch_size or raw_cfg.get("generation", {}).get("eval_batch_size", 1))
    generation_settings = {
        "max_new_tokens": int(generation_value(args, raw_cfg, "max_new_tokens", 256)),
        "min_new_tokens": int(generation_value(args, raw_cfg, "min_new_tokens", 16)),
        "num_beams": int(generation_value(args, raw_cfg, "num_beams", 1)),
        "do_sample": bool(generation_value(args, raw_cfg, "do_sample", False)),
        "temperature": float(generation_value(args, raw_cfg, "temperature", 0.0)),
        "top_k": int(generation_value(args, raw_cfg, "top_k", 0)),
        "top_p": float(generation_value(args, raw_cfg, "top_p", 1.0)),
        "repetition_penalty": float(generation_value(args, raw_cfg, "repetition_penalty", 1.05)),
        "no_repeat_ngram_size": int(generation_value(args, raw_cfg, "no_repeat_ngram_size", 3)),
    }
    generate_kwargs = {
        "max_new_tokens": generation_settings["max_new_tokens"],
        "min_new_tokens": generation_settings["min_new_tokens"],
        "num_beams": generation_settings["num_beams"],
        "do_sample": generation_settings["do_sample"],
        "repetition_penalty": generation_settings["repetition_penalty"],
        "no_repeat_ngram_size": generation_settings["no_repeat_ngram_size"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_settings["do_sample"]:
        generate_kwargs.update(
            {
                "temperature": max(1e-5, generation_settings["temperature"]),
                "top_k": generation_settings["top_k"],
                "top_p": generation_settings["top_p"],
            }
        )

    source_prefix = raw_cfg.get("data", {}).get("source_prefix", "")
    predictions: List[str] = []
    references: List[str] = []
    sources: List[str] = []
    latencies: List[float] = []
    new_token_counts: List[float] = []
    total_new_tokens = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with predictions_path.open("w", encoding="utf-8") as out_f:
        for offset in tqdm(range(0, len(examples), batch_size), desc="Generating"):
            batch = examples[offset : offset + batch_size]
            batch_sources = [row.source for row in batch]
            batch_refs = [row.target for row in batch]
            enc = tokenizer(
                [source_prefix + source for source in batch_sources],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=int(raw_cfg["data"]["max_source_length"]),
            ).to(device)

            sync_device(device)
            generation_start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(**enc, **generate_kwargs)
            sync_device(device)
            batch_elapsed = time.perf_counter() - generation_start

            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for row, source, reference, prediction, ids in zip(batch, batch_sources, batch_refs, decoded, output_ids):
                if tokenizer.pad_token_id is None:
                    new_tokens = int(ids.numel())
                else:
                    new_tokens = int((ids != tokenizer.pad_token_id).sum().item())
                total_new_tokens += new_tokens
                predictions.append(prediction)
                references.append(reference)
                sources.append(source)
                latencies.append(batch_elapsed / max(1, len(batch)))
                new_token_counts.append(float(new_tokens))

                out_f.write(
                    json.dumps(
                        {
                            "id": row.identifier,
                            "source": source,
                            "reference": reference,
                            "prediction": prediction,
                            "generated_tokens": new_tokens,
                            "latency_seconds": batch_elapsed / max(1, len(batch)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out_f.flush()

    elapsed = time.perf_counter() - start
    metrics = compute_metrics(
        predictions,
        references,
        sources=sources,
        compute_bertscore=args.compute_bertscore or bool(raw_cfg.get("evaluation", {}).get("compute_bertscore", False)),
        bertscore_model_type=args.bertscore_model_type
        or raw_cfg.get("evaluation", {}).get("bertscore_model_type", "bert-base-multilingual-cased"),
    )
    decode_elapsed = max(1e-9, sum(latencies))
    metrics.update(
        {
            "elapsed_seconds": round(elapsed, 3),
            "examples_per_second": round(len(predictions) / max(1e-9, elapsed), 6),
            "generated_tokens_per_second": round(total_new_tokens / max(1e-9, elapsed), 6),
            "decode_mode": "autoregressive_generate",
            "decode_elapsed_seconds": round(decode_elapsed, 3),
            "decode_examples_per_second": round(len(predictions) / max(1e-9, decode_elapsed), 6),
            "decode_generated_tokens_per_second": round(total_new_tokens / max(1e-9, decode_elapsed), 6),
            "seconds_per_generated_token": round(decode_elapsed / max(1, total_new_tokens), 8),
            "latency_seconds_mean": round(safe_mean(latencies), 6),
            "latency_seconds_median": round(safe_median(latencies), 6),
            "latency_seconds_p95": round(percentile(latencies, 95), 6),
            "latency_seconds_min": round(min(latencies) if latencies else 0.0, 6),
            "latency_seconds_max": round(max(latencies) if latencies else 0.0, 6),
            "generated_tokens_total": int(total_new_tokens),
            "generated_tokens_mean": round(safe_mean(new_token_counts), 4),
            "decode_steps_total": float(total_new_tokens),
            "decode_steps_mean": round(safe_mean(new_token_counts), 6),
            "tokens_per_decode_step": 1.0,
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else 0.0,
            "checkpoint": checkpoint_label,
            "base_model": model_name,
            "checkpoint_base_model": checkpoint_manifest.get("base_model"),
            "unique_parameter_elements": unique_parameter_elements,
            "total_parameters": unique_parameter_elements,
            "checkpoint_parameter_elements": checkpoint_parameter_elements,
            "checkpoint_parameters_match_model": checkpoint_parameters_match_model,
            "config": str(config_path),
            "test_file": str(test_file),
            "predictions_file": str(predictions_path),
            "evaluation_split": "test",
            "generation": generation_settings,
            "source_prefix": source_prefix,
            "max_source_length": int(raw_cfg["data"]["max_source_length"]),
            "max_target_length": int(raw_cfg["data"]["max_target_length"]),
            "eval_batch_size": batch_size,
        }
    )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with run_info_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": checkpoint_label,
                "base_model": model_name,
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "generation": generation_settings,
                "metrics_file": str(metrics_path),
                "predictions_file": str(predictions_path),
                "evaluation_split": "test",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    maybe_upload_eval_outputs(raw_cfg, output_dir)


if __name__ == "__main__":
    main()
