from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from .evaluate_alignscore import _attach_sources, _sentence_split, sentence_splitter_name
    from .metric_io import EvaluationRow, load_jsonl, metric_output_path
except ImportError:
    from evaluate_alignscore import _attach_sources, _sentence_split, sentence_splitter_name
    from metric_io import EvaluationRow, load_jsonl, metric_output_path


SCALE_PAPER = "Lattimer et al., EMNLP 2023"
SCALE_PROMPT = '{{premise}} Question: Does this imply that "{{hypothesis}}"? Yes or No?'
SCALE_SIZES = frozenset({"small", "base", "large", "xl", "xxl"})
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_WINDOW_SIZE = 0.25
DEFAULT_BATCH_SIZE = 8
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[A-ZÀ-ÖØ-Þ0-9])")


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sentence_split_fallback(text: str) -> list[str]:

    value = text.strip()
    return [part.strip() for part in _SENTENCE_BREAK.split(value) if part.strip()]


def _token_ids(tokenizer: Any, text: str) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("SCALE tokenizer must return one two-dimensional input_ids row")
    return input_ids.squeeze(0).to(dtype=torch.long)


def _build_chunks(
    tokenizer: Any,
    source: str,
    hypothesis: str,
    *,
    chunk_size: int,
    window_size: float,
) -> list[torch.Tensor]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0.0 <= window_size < 1.0:
        raise ValueError("window_size must satisfy 0 <= window_size < 1")

    prompt = SCALE_PROMPT.replace("{{hypothesis}}", hypothesis)
    full_text = prompt.replace("{{premise}}", source)
    full_tokens = _token_ids(tokenizer, full_text)
    if full_tokens.numel() < chunk_size:
        return [full_tokens.unsqueeze(0)]

    pre_prompt, post_prompt = prompt.split("{{premise}}", 1)

    pre_tokens = _token_ids(tokenizer, pre_prompt)[:-1]
    post_tokens = _token_ids(tokenizer, post_prompt)
    source_tokens = _token_ids(tokenizer, source)
    prompt_token_len = int(pre_tokens.numel() + post_tokens.numel())
    chunk_size_mod = chunk_size - prompt_token_len
    if chunk_size_mod <= 0:
        raise ValueError(
            f"chunk_size is too small for the SCALE prompt; increase --chunk-size above {prompt_token_len} tokens"
        )
    step = int(chunk_size_mod * (1.0 - window_size))
    if step <= 0:
        raise ValueError("window_size leaves no positive token step")

    num_windows = math.ceil((max(int(source_tokens.numel()) - 1, 1)) / step)
    chunks: list[torch.Tensor] = []
    for index in range(num_windows):
        begin = step * index
        end = begin + chunk_size_mod

        source_end: int | None = -1 if end >= source_tokens.numel() else end
        source_slice = source_tokens[begin:source_end]
        if pre_tokens.numel() == 0:
            chunk = torch.cat([source_slice, post_tokens])
        else:
            chunk = torch.cat([pre_tokens, source_slice, post_tokens])
        chunks.append(chunk.unsqueeze(0))
    if not chunks:
        return [full_tokens.unsqueeze(0)]
    return chunks


def _first_token_id(tokenizer: Any, text: str) -> int:
    ids = _token_ids(tokenizer, text)
    if ids.numel() == 0:
        raise ValueError(f"SCALE tokenizer produced no token for {text!r}")
    return int(ids[0].item())


class LocalSCALE:
    def __init__(
        self,
        model_path: Path,
        *,
        size: str = "large",
        tokenizer_path: Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        window_size: float = DEFAULT_WINDOW_SIZE,
    ) -> None:
        if size not in SCALE_SIZES:
            choices = ", ".join(sorted(SCALE_SIZES))
            raise ValueError(f"SCALE size must be one of {choices}; got {size!r}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= window_size < 1.0:
            raise ValueError("window_size must satisfy 0 <= window_size < 1")

        self.model_path = model_path.expanduser().resolve()
        self.tokenizer_path = (tokenizer_path or model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"SCALE model path must be a local directory: {self.model_path}")
        if not self.tokenizer_path.is_dir():
            raise FileNotFoundError(f"SCALE tokenizer path must be a local directory: {self.tokenizer_path}")

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("SCALE evaluation requires the `transformers` package") from exc

        self.size = size
        self.batch_size = batch_size
        self.device = device or _default_device()
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path), local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path), local_files_only=True)
        self.model.to(self.device).eval()
        self.yes_token_id = _first_token_id(self.tokenizer, "Yes")
        self.no_token_id = _first_token_id(self.tokenizer, "No")
        self._splitter = sentence_splitter_name()

    def _forward(self, chunks: Sequence[torch.Tensor]) -> list[float]:
        if not chunks:
            return []
        features = [{"input_ids": chunk.squeeze(0).tolist()} for chunk in chunks]
        encoded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
            )
        scores = getattr(generated, "scores", None)
        if scores is None and isinstance(generated, dict):
            scores = generated.get("scores")
        if not scores:
            raise RuntimeError("SCALE model.generate did not return one-step output scores")
        logits = scores[0]
        if logits.ndim != 2 or logits.shape[0] != len(chunks):
            raise RuntimeError("Unexpected SCALE generation score shape")
        yes_no = logits[:, [self.yes_token_id, self.no_token_id]].float()
        probabilities = torch.softmax(yes_no, dim=-1)[:, 0]
        return [float(value) for value in probabilities.detach().cpu().tolist()]

    def score_document(self, source: str, prediction: str) -> float:
        claims = _sentence_split(prediction) or _sentence_split_fallback(prediction)
        if not claims:
            return 0.0
        sentence_scores: list[float] = []
        for claim in claims:
            chunks = _build_chunks(
                self.tokenizer,
                source,
                claim,
                chunk_size=self.chunk_size,
                window_size=self.window_size,
            )
            chunk_scores: list[float] = []
            for start in range(0, len(chunks), self.batch_size):
                chunk_scores.extend(self._forward(chunks[start : start + self.batch_size]))
            sentence_scores.append(max(chunk_scores))
        return sum(sentence_scores) / len(sentence_scores)


def _score_rows(rows: Sequence[EvaluationRow], scorer: Any, *, progress_every: int = 0) -> list[float]:
    scores: list[float] = []
    for index, row in enumerate(rows, start=1):
        value = float(scorer.score_document(row.source or "", row.prediction))
        if not math.isfinite(value):
            raise ValueError(f"SCALE returned a non-finite score for row {row.identifier!r}")
        scores.append(min(1.0, max(0.0, value)))
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(rows)):
            print(f"[scale] scored {index}/{len(rows)} examples", flush=True)
    return scores


def evaluate(
    predictions_file: Path,
    model_path: Path,
    output_file: Path | None = None,
    *,
    size: str = "large",
    tokenizer_path: Path | None = None,
    prediction_field: str = "prediction",
    reference_field: str = "reference",
    source_field: str = "source",
    source_file: Path | None = None,
    source_file_field: str = "text",
    source_id_field: str = "id",
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    window_size: float = DEFAULT_WINDOW_SIZE,
    details: bool = False,
    progress_every: int = 0,
    scorer: Any | None = None,
) -> dict[str, Any]:
    predictions_file = predictions_file.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    source_joined = False
    try:
        rows = load_jsonl(
            predictions_file,
            prediction_field=prediction_field,
            reference_field=reference_field,
            source_field=source_field,
        )
    except ValueError as exc:
        if not str(exc).startswith(f"Missing {source_field!r}"):
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
        )
        rows = _attach_sources(
            rows,
            source_file,
            source_file_field=source_file_field,
            source_id_field=source_id_field,
        )
        source_joined = True

    evaluator = scorer or LocalSCALE(
        model_path,
        size=size,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        device=device,
        chunk_size=chunk_size,
        window_size=window_size,
    )
    scores = _score_rows(rows, evaluator, progress_every=progress_every)
    mean_score = sum(scores) / len(scores)
    result: dict[str, Any] = {
        "schema_version": "seam.scale_source_support.v1",
        "metric": "SCALE",
        "paper": SCALE_PAPER,
        "definition": (
            "Mean over prediction sentences of the maximum SCALE Yes probability "
            "over overlapping source chunks; higher means stronger source support"
        ),
        "score_direction": "scale_consistency_higher_is_better",
        "scale_consistency": mean_score,
        "unsupported_content_proxy": 1.0 - mean_score,
        "score_scale": "0-1",
        "num_examples": len(rows),
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path.expanduser().resolve()) if tokenizer_path is not None else str(model_path),
        "scale_size": size,
        "device": device or _default_device(),
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "window_size": window_size,
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
                "scale_consistency": scores[row.row_index],
                "unsupported_content_proxy": 1.0 - scores[row.row_index],
            }
            for row in rows
        ]

    output_path = metric_output_path(predictions_file, output_file, ".scale.json")
    result["output_file"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source support with SCALE (Lattimer et al., EMNLP 2023). "
            "--model-path must be a local Flan-T5 directory; network downloads are disabled."
        )
    )
    parser.add_argument(
        "predictions", type=Path, help="Prediction JSONL with prediction/reference and optionally source"
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Local Flan-T5 checkpoint directory")
    parser.add_argument("--tokenizer-path", type=Path, help="Optional local tokenizer directory")
    parser.add_argument(
        "--size", choices=sorted(SCALE_SIZES), default="large", help="Flan-T5 size for metadata/reproducibility"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--source-file", type=Path, help="Original test JSONL used to join source by prediction id")
    parser.add_argument("--source-file-field", default="text")
    parser.add_argument("--source-id-field", default="id")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="SCALE source window size in tokens")
    parser.add_argument(
        "--window-size", type=float, default=DEFAULT_WINDOW_SIZE, help="Fractional overlap between source windows"
    )
    parser.add_argument("--details", action="store_true", help="Include per-example scores in the JSON output")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N examples; 0 disables")
    args = parser.parse_args()
    output = evaluate(
        args.predictions,
        args.model_path,
        args.output,
        size=args.size,
        tokenizer_path=args.tokenizer_path,
        prediction_field=args.prediction_field,
        reference_field=args.reference_field,
        source_field=args.source_field,
        source_file=args.source_file,
        source_file_field=args.source_file_field,
        source_id_field=args.source_id_field,
        batch_size=args.batch_size,
        device=args.device,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        details=args.details,
        progress_every=args.progress_every,
    )
    print(
        "SCALE="
        f"{output['scale_consistency']:.4f} "
        "UnsupportedContentProxy="
        f"{output['unsupported_content_proxy']:.4f} (lower is better)"
    )
    print(f"Saved metrics: {Path(output['output_file']).resolve()}")


if __name__ == "__main__":
    main()
