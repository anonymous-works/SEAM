from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from led.config import DEFAULT_CONFIG, load_config
from led.train import _dtype
from t5gemma2.data import load_summarization_jsonl


def _first_token_global_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    first_visible = attention_mask.to(torch.int64).argmax(dim=1)
    mask = torch.zeros_like(attention_mask)
    mask.scatter_(1, first_visible.unsqueeze(1), 1)
    return mask


def evaluate(
    config: dict[str, Any],
    *,
    checkpoint: str | None = None,
    limit: int = -1,
    batch_size: int | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("LED generation requires a CUDA GPU")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device("cuda")
    run_dir = Path(config["run"]["output_dir"])
    checkpoint_dir = Path(checkpoint or run_dir / "best_model").expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"LED checkpoint directory not found: {checkpoint_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        torch_dtype=_dtype(config["model"]["torch_dtype"]),
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = bool(config["model"]["use_cache_for_eval"])
    model.eval()

    examples = load_summarization_jsonl(config["data"]["test_file"], config["data"], limit=limit)
    generation = config["generation"]
    batch_size = int(batch_size or generation["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_dir = run_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    source_prefix = str(config["data"].get("source_prefix", ""))
    generate_kwargs = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "min_new_tokens": int(generation["min_new_tokens"]),
        "num_beams": int(generation["num_beams"]),
        "do_sample": bool(generation["do_sample"]),
        "repetition_penalty": float(generation["repetition_penalty"]),
        "no_repeat_ngram_size": int(generation["no_repeat_ngram_size"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with prediction_path.open("w", encoding="utf-8") as output:
        for start in tqdm(range(0, len(examples), batch_size), desc=f"LED {config['dataset']} test"):
            rows = examples[start : start + batch_size]
            encoded = tokenizer(
                [source_prefix + row.source for row in rows],
                max_length=int(config["data"]["max_source_length"]),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            global_attention_mask = _first_token_global_mask(encoded["attention_mask"])
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    global_attention_mask=global_attention_mask.to(device),
                    **generate_kwargs,
                )
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for row, prediction in zip(rows, predictions):
                output.write(
                    json.dumps(
                        {
                            "id": row.identifier,
                            "source": row.source,
                            "reference": row.target,
                            "prediction": prediction,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    info = {
        "base_model": config["model"]["model_id"],
        "checkpoint": str(checkpoint_dir),
        "evaluation_split": "test",
        "num_examples": len(examples),
        "generation": generation,
        "max_source_length": config["data"]["max_source_length"],
        "max_target_length": config["data"]["max_target_length"],
        "global_attention": "first non-padding source token",
        "predictions_file": str(prediction_path),
    }
    (output_dir / "eval_run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return prediction_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LED predictions on the held-out test split")
    parser.add_argument("--dataset", required=True, choices=("pubmed", "arxiv", "booksum", "govreport"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.dataset, config_path=args.config)
    path = evaluate(config, checkpoint=args.checkpoint, limit=args.limit, batch_size=args.batch_size)
    print(f"LED predictions: {path}")


if __name__ == "__main__":
    main()
