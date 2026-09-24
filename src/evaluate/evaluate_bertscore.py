from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

try:
    from .metric_io import load_jsonl, metric_output_path
except ImportError:
    from metric_io import load_jsonl, metric_output_path


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _infer_num_layers(model_path: Path) -> int:

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    except Exception as exc:
        raise ValueError(
            "Could not infer --num-layers from the local model config; pass --num-layers explicitly"
        ) from exc

    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)
    for candidate in configs:
        for field in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
            value = getattr(candidate, field, None)
            if value is not None and int(value) > 0:
                return int(value)
    raise ValueError("The local model config does not expose a layer count; pass --num-layers explicitly")


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_context_length(scorer: Any) -> int:

    config = getattr(getattr(scorer, "_model", None), "config", None)
    value = getattr(config, "max_position_embeddings", None)
    try:
        context_length = int(value)
    except (TypeError, ValueError, OverflowError):
        context_length = 0
    if context_length <= 0 or context_length > 1_000_000:
        return 512

    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type in {"roberta", "xlm-roberta", "camembert"} and context_length > 512:
        context_length -= 2
    return max(1, context_length)


def _patch_tokenizer_max_length(scorer: Any, requested: int | None = None) -> int | None:

    tokenizer = getattr(scorer, "_tokenizer", None)
    if tokenizer is None:
        return None
    if requested is not None:
        if requested <= 0 or requested > 2**31 - 1:
            raise ValueError("BERTScore max_length must lie in [1, 2^31-1]")
        resolved = int(requested)
    else:
        current = getattr(tokenizer, "model_max_length", None)
        try:
            current_int = int(current)
        except (TypeError, ValueError, OverflowError):
            current_int = 0

        resolved = current_int if 0 < current_int <= 1_000_000 else _model_context_length(scorer)

    tokenizer.model_max_length = resolved

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs["model_max_length"] = resolved
    return resolved


def _as_unit_scores(values: Any, name: str) -> list[float]:
    scores = [float(value) for value in values.tolist()]
    if any(not 0.0 <= value <= 1.0 + 1.0e-4 for value in scores):
        raise ValueError(f"BERTScore returned an invalid {name} value outside [0, 1]")

    return [min(1.0, max(0.0, value)) for value in scores]


def _build_scorer(
    model_path: Path,
    *,
    num_layers: int,
    batch_size: int,
    device: str,
    language: str | None,
    use_fast_tokenizer: bool,
    idf: bool,
):

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError("BERTScore is unavailable; install the optional `bert-score` package") from exc
    scorer = BERTScorer(
        model_type=str(model_path),
        num_layers=num_layers,
        batch_size=batch_size,
        device=device,
        lang=language,
        idf=idf,
        rescale_with_baseline=False,
        use_fast_tokenizer=use_fast_tokenizer,
    )
    return scorer


def evaluate(
    predictions_file: Path,
    model_path: Path,
    output_file: Path | None = None,
    *,
    prediction_field: str = "prediction",
    reference_field: str = "reference",
    batch_size: int = 64,
    num_layers: int | None = None,
    device: str | None = None,
    language: str | None = None,
    use_fast_tokenizer: bool = False,
    idf: bool = False,
    max_length: int | None = None,
    details: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    predictions_file = predictions_file.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"BERTScore model path must be an existing local directory: {model_path}")
    if batch_size <= 0:
        raise ValueError("BERTScore batch_size must be positive")
    resolved_layers = int(num_layers) if num_layers is not None else _infer_num_layers(model_path)
    if resolved_layers <= 0:
        raise ValueError("BERTScore num_layers must be positive")

    rows = load_jsonl(
        predictions_file,
        prediction_field=prediction_field,
        reference_field=reference_field,
    )
    resolved_device = device or _default_device()
    scorer = _build_scorer(
        model_path,
        num_layers=resolved_layers,
        batch_size=batch_size,
        device=resolved_device,
        language=language,
        use_fast_tokenizer=use_fast_tokenizer,
        idf=idf,
    )
    tokenizer_max_length = _patch_tokenizer_max_length(scorer, max_length)
    if idf:
        scorer.compute_idf([reference for row in rows for reference in row.references])
    candidates = [row.prediction for row in rows]
    references = [list(row.references) for row in rows]
    precision, recall, f1 = scorer.score(candidates, references, verbose=verbose, batch_size=batch_size)
    values = {
        "precision": _as_unit_scores(precision, "precision"),
        "recall": _as_unit_scores(recall, "recall"),
        "f1": _as_unit_scores(f1, "f1"),
    }
    result: dict[str, Any] = {
        "schema_version": "eviseq.bertscore.v1",
        "backend": f"bert-score=={_package_version('bert-score')}",
        "model_path": str(model_path),
        "num_layers": resolved_layers,
        "device": resolved_device,
        "idf": bool(idf),
        "tokenizer_max_length": tokenizer_max_length,
        "score_scale": "0-100",
        "num_examples": len(rows),
        "prediction_field": prediction_field,
        "reference_field": reference_field,
        "predictions_file": str(predictions_file),
        "bertscore": {
            "precision": 100.0 * sum(values["precision"]) / len(rows),
            "recall": 100.0 * sum(values["recall"]) / len(rows),
            "f1": 100.0 * sum(values["f1"]) / len(rows),
        },
    }
    if details:
        result["rows"] = [
            {
                "row_index": row.row_index,
                "id": row.identifier,
                "precision": 100.0 * values["precision"][row.row_index],
                "recall": 100.0 * values["recall"][row.row_index],
                "f1": 100.0 * values["f1"][row.row_index],
            }
            for row in rows
        ]

    output_file = metric_output_path(predictions_file, output_file, ".bertscore.json")
    result["output_file"] = str(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute BERTScore from prediction JSONL. --model-path must point to a local "
            "Hugging Face encoder directory; network downloads are disabled."
        )
    )
    parser.add_argument("predictions", type=Path, help="Prediction JSONL file")
    parser.add_argument("--model-path", type=Path, required=True, help="Local BERTScore model directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-layers", type=int, help="Model layer used for BERTScore; inferred from config if omitted"
    )
    parser.add_argument("--device", help="Torch device, for example cpu or cuda:0")
    parser.add_argument("--lang", dest="language", help="Optional BERTScore language code")
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--idf", action="store_true", help="Compute IDF weights from the references")
    parser.add_argument(
        "--max-length",
        type=int,
        help="Tokenizer truncation length; inferred from the model when omitted",
    )
    parser.add_argument("--details", action="store_true", help="Include per-example scores in the JSON output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    output = evaluate(
        args.predictions,
        args.model_path,
        args.output,
        prediction_field=args.prediction_field,
        reference_field=args.reference_field,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        device=args.device,
        language=args.language,
        use_fast_tokenizer=args.use_fast_tokenizer,
        idf=args.idf,
        max_length=args.max_length,
        details=args.details,
        verbose=args.verbose,
    )
    scores = output["bertscore"]
    print(f"BERTScore-P={scores['precision']:.3f} BERTScore-R={scores['recall']:.3f} BERTScore-F1={scores['f1']:.3f}")
    print(f"Saved metrics: {Path(output['output_file']).resolve()}")


if __name__ == "__main__":
    main()
