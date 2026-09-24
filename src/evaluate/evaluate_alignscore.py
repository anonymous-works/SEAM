from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

try:
    from .metric_io import EvaluationRow, load_jsonl, metric_output_path
except ImportError:
    from metric_io import EvaluationRow, load_jsonl, metric_output_path


ALIGN_SCORE_PAPER = "Zha et al., ACL 2023"
DEFAULT_CHUNK_WORDS = 350
DEFAULT_MAX_LENGTH = 512
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[A-ZÀ-ÖØ-Þ0-9])")
_SOURCE_FIELD_FALLBACKS = ("source", "text", "document", "article_text", "article")


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sentence_split(text: str) -> list[str]:

    value = text.strip()
    if not value:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(value)
        if sentences:
            return [sentence.strip() for sentence in sentences if sentence.strip()]
    except (ImportError, LookupError):
        pass
    return [part.strip() for part in _SENTENCE_BREAK.split(value) if part.strip()]


def sentence_splitter_name() -> str:

    try:
        from nltk.data import find

        try:
            find("tokenizers/punkt_tab")
        except LookupError:
            find("tokenizers/punkt")
        return "nltk.sent_tokenize"
    except (ImportError, LookupError):
        return "regex_fallback"


def _source_chunks(source: str, chunk_words: int) -> list[str]:

    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    sentences = _sentence_split(source)
    if not sentences:
        return [""]

    n_groups = len(source.strip().split()) // chunk_words + 1
    sentences_per_group = max(len(sentences) // n_groups, 1)
    return [
        " ".join(sentences[start : start + sentences_per_group])
        for start in range(0, len(sentences), sentences_per_group)
    ]


def _source_text(value: Any, *, field: str, path: Path, line_number: int) -> str:

    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field!r} must contain strings at {path}:{line_number}")
        text = "\n".join(item.strip() for item in value if item.strip())
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ValueError(f"{field!r} must be a string or list of strings at {path}:{line_number}")
    if not text:
        raise ValueError(f"{field!r} is empty at {path}:{line_number}")
    return text


def _load_source_map(path: Path, *, source_field: str, id_field: str) -> dict[str, str]:

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    source_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")

            raw_id = row.get(id_field)
            if raw_id in (None, ""):
                raise ValueError(f"Missing {id_field!r} at {path}:{line_number}")
            identifier = str(raw_id)
            if identifier in source_by_id:
                raise ValueError(f"Duplicate source ID {identifier!r} at {path}:{line_number}")

            value = row.get(source_field)
            selected_field = source_field
            if value is None:
                for fallback in _SOURCE_FIELD_FALLBACKS:
                    if fallback in row:
                        value = row[fallback]
                        selected_field = fallback
                        break
            if value is None:
                fields = ", ".join(repr(field) for field in _SOURCE_FIELD_FALLBACKS)
                raise ValueError(f"Missing source field (tried {fields}) at {path}:{line_number}")
            source_by_id[identifier] = _source_text(
                value,
                field=selected_field,
                path=path,
                line_number=line_number,
            )
    if not source_by_id:
        raise ValueError(f"No source records found in {path}")
    return source_by_id


def _attach_sources(
    rows: Sequence[EvaluationRow],
    source_file: Path,
    *,
    source_file_field: str,
    source_id_field: str,
) -> list[EvaluationRow]:

    source_by_id = _load_source_map(
        source_file,
        source_field=source_file_field,
        id_field=source_id_field,
    )
    missing: list[str] = []
    enriched: list[EvaluationRow] = []
    for row in rows:
        identifier = str(row.identifier)
        source = source_by_id.get(identifier)
        if source is None:
            missing.append(identifier)
            continue
        enriched.append(replace(row, source=source))
    if missing:
        preview = ", ".join(repr(identifier) for identifier in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Source file {source_file} has no record for prediction IDs: {preview}{suffix}")
    return enriched


def _load_checkpoint(path: Path) -> dict[str, Tensor]:

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"AlignScore checkpoint does not contain a state dict: {path}")
    state = {str(key): value for key, value in payload.items() if isinstance(value, Tensor)}
    if not state:
        raise ValueError(f"AlignScore checkpoint state dict is empty: {path}")
    return state


def _state_value(state: dict[str, Tensor], suffix: str) -> Tensor | None:

    for key, value in state.items():
        if key == suffix or key.endswith("." + suffix):
            return value
    return None


def _base_model_state(state: dict[str, Tensor]) -> dict[str, Tensor]:

    extracted: dict[str, Tensor] = {}
    marker = "base_model."
    for key, value in state.items():
        if marker in key:
            extracted[key.split(marker, 1)[1]] = value
    return extracted


class LocalAlignScore:
    def __init__(
        self,
        model_path: Path,
        checkpoint_path: Path,
        *,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        verbose: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if chunk_words <= 0:
            raise ValueError("chunk_words must be positive")
        if not model_path.is_dir():
            raise FileNotFoundError(f"AlignScore backbone path must be a local directory: {model_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AlignScore checkpoint must be a local file: {checkpoint_path}")

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("AlignScore evaluation requires the `transformers` package") from exc

        self.model_path = model_path.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.batch_size = batch_size
        self.device = device or _default_device()
        self.max_length = max_length
        self.chunk_words = chunk_words
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.encoder = AutoModel.from_pretrained(str(self.model_path), local_files_only=True)
        checkpoint = _load_checkpoint(self.checkpoint_path)

        encoder_state = _base_model_state(checkpoint)
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)

        tri_weight = _state_value(checkpoint, "tri_layer.weight")
        tri_bias = _state_value(checkpoint, "tri_layer.bias")
        if tri_weight is None or tri_bias is None or tri_weight.ndim != 2 or tri_weight.shape[0] != 3:
            raise ValueError(
                "The checkpoint is missing AlignScore's tri_layer; pass an AlignScore .ckpt trained for nli_sp"
            )
        hidden_size = int(tri_weight.shape[1])
        self.tri_layer = nn.Linear(hidden_size, 3)
        self.tri_layer.load_state_dict({"weight": tri_weight, "bias": tri_bias})

        self.encoder.to(self.device).eval()
        self.tri_layer.to(self.device).eval()
        self._splitter = sentence_splitter_name()

    def _encode(self, contexts: Sequence[str], claims: Sequence[str]) -> dict[str, Tensor]:
        try:
            encoded = self.tokenizer(
                list(contexts),
                list(claims),
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception:
            encoded = self.tokenizer(
                list(contexts),
                list(claims),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        return {
            key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

    def _forward(self, contexts: Sequence[str], claims: Sequence[str]) -> list[float]:
        encoded = self._encode(contexts, claims)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        try:
            with torch.inference_mode():
                output = self.encoder(**encoded)
        except TypeError:
            encoded.pop("token_type_ids", None)
            with torch.inference_mode():
                output = self.encoder(**encoded)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        with torch.inference_mode():
            logits = self.tri_layer(pooled.float())
            probabilities = torch.softmax(logits, dim=-1)[:, 0]
        return [float(value) for value in probabilities.detach().cpu().tolist()]

    def score_pairs(self, contexts: Sequence[str], claims: Sequence[str]) -> list[float]:

        if len(contexts) != len(claims):
            raise ValueError("contexts and claims must have the same length")
        scores: list[float] = []
        for start in range(0, len(contexts), self.batch_size):
            scores.extend(
                self._forward(contexts[start : start + self.batch_size], claims[start : start + self.batch_size])
            )
        return scores

    def score_document(self, source: str, prediction: str) -> float:

        claim_sentences = _sentence_split(prediction)
        if not claim_sentences:
            return 0.0
        context_chunks = _source_chunks(source, self.chunk_words)
        contexts = [chunk for chunk in context_chunks for _ in claim_sentences]
        claims = claim_sentences * len(context_chunks)
        pair_scores = self.score_pairs(contexts, claims)
        support_scores = [
            max(pair_scores[offset + sentence_index] for offset in range(0, len(pair_scores), len(claim_sentences)))
            for sentence_index in range(len(claim_sentences))
        ]
        return sum(support_scores) / len(support_scores)


def _score_rows(rows: Sequence[EvaluationRow], scorer: Any) -> list[float]:
    return [float(scorer.score_document(row.source or "", row.prediction)) for row in rows]


def evaluate(
    predictions_file: Path,
    model_path: Path,
    checkpoint_path: Path,
    output_file: Path | None = None,
    *,
    prediction_field: str = "prediction",
    reference_field: str = "reference",
    source_field: str = "source",
    source_file: Path | None = None,
    source_file_field: str = "text",
    source_id_field: str = "id",
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    details: bool = False,
    scorer: Any | None = None,
) -> dict[str, Any]:
    predictions_file = predictions_file.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    source_joined = False
    try:
        rows = load_jsonl(
            predictions_file,
            prediction_field=prediction_field,
            reference_field=reference_field,
            source_field=source_field,
        )
    except ValueError as exc:
        missing_source = str(exc).startswith(f"Missing {source_field!r}")
        if not missing_source:
            raise
        if source_file is None:
            raise ValueError(
                f"{exc}. Prediction rows do not contain a source; pass --source-file "
                "with the original test JSONL so sources can be joined by id."
            ) from exc

        rows = load_jsonl(
            predictions_file,
            prediction_field=prediction_field,
            reference_field=reference_field,
            source_field=None,
        )
        rows = _attach_sources(
            rows,
            source_file,
            source_file_field=source_file_field,
            source_id_field=source_id_field,
        )
        source_joined = True
    evaluator = scorer or LocalAlignScore(
        model_path,
        checkpoint_path,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        chunk_words=chunk_words,
    )
    consistency = [min(1.0, max(0.0, score)) for score in _score_rows(rows, evaluator)]
    hallucination = [1.0 - score for score in consistency]
    result: dict[str, Any] = {
        "schema_version": "eviseq.alignscore_hallucination.v1",
        "metric": "AlignScore-nli_sp",
        "paper": ALIGN_SCORE_PAPER,
        "definition": (
            "AlignScore factual-consistency score from source chunks to prediction sentences; "
            "hallucination_score is 1 - consistency"
        ),
        "score_direction": "hallucination_score_lower_is_better",
        "alignscore_consistency": sum(consistency) / len(consistency),
        "hallucination_score": sum(hallucination) / len(hallucination),
        "score_scale": "0-1",
        "num_examples": len(rows),
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path),
        "device": device or _default_device(),
        "batch_size": batch_size,
        "max_length": max_length,
        "chunk_words": chunk_words,
        "sentence_splitter": getattr(evaluator, "_splitter", sentence_splitter_name()),
        "prediction_field": prediction_field,
        "reference_field": reference_field,
        "source_field": source_field,
        "source_file": str(source_file.expanduser().resolve()) if source_file is not None else None,
        "source_file_field": source_file_field,
        "source_id_field": source_id_field,
        "source_joined_by_id": source_joined,
        "predictions_file": str(predictions_file),
    }
    if details:
        result["rows"] = [
            {
                "row_index": row.row_index,
                "id": row.identifier,
                "alignscore_consistency": consistency[row.row_index],
                "hallucination_score": hallucination[row.row_index],
            }
            for row in rows
        ]

    output_path = metric_output_path(predictions_file, output_file, ".alignscore.json")
    result["output_file"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source-grounded factuality with AlignScore (Zha et al., ACL 2023). "
            "Both --model-path and --checkpoint-path are local; network access is disabled. "
            "If predictions omit source, --source-file joins it by id."
        )
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="Prediction JSONL containing prediction/reference and optionally source",
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Local AlignScore backbone directory")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Local AlignScore .ckpt file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--source-field", default="source")
    parser.add_argument(
        "--source-file",
        type=Path,
        help="Original test JSONL used to join source text by prediction id when source is absent",
    )
    parser.add_argument(
        "--source-file-field",
        default="text",
        help="Source field in --source-file (fallbacks include source, document and article)",
    )
    parser.add_argument("--source-id-field", default="id", help="ID field in --source-file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="Torch device, for example cpu or cuda:0")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--details", action="store_true", help="Include per-example scores in the JSON output")
    args = parser.parse_args()
    output = evaluate(
        args.predictions,
        args.model_path,
        args.checkpoint_path,
        args.output,
        prediction_field=args.prediction_field,
        reference_field=args.reference_field,
        source_field=args.source_field,
        source_file=args.source_file,
        source_file_field=args.source_file_field,
        source_id_field=args.source_id_field,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        chunk_words=args.chunk_words,
        details=args.details,
    )
    print(
        "AlignScore="
        f"{output['alignscore_consistency']:.4f} "
        "HallucinationScore="
        f"{output['hallucination_score']:.4f} (lower is better)"
    )
    print(f"Saved metrics: {Path(output['output_file']).resolve()}")


if __name__ == "__main__":
    main()
