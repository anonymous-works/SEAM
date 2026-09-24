from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("source/target lists must contain strings")
        return "\n".join(item.strip() for item in value if item.strip())
    if not isinstance(value, str):
        raise ValueError("source and target fields must be strings or lists of strings")
    return value


def _field(row: Mapping[str, Any], name: str, fallbacks: Iterable[str]) -> Any:
    value: Any = row
    try:
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(name)
            value = value[part]
        return value
    except KeyError:
        for fallback in fallbacks:
            if fallback in row and row[fallback] not in (None, "", []):
                return row[fallback]
    raise KeyError(name)


def read_jsonl(path: str | Path, *, max_examples: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if max_examples > 0 and len(records) >= max_examples:
                break
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")

            row.setdefault("_decoder_baseline_row_id", f"row-{line_number:08d}")
            records.append(row)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def record_texts(row: Mapping[str, Any], data: Mapping[str, Any]) -> tuple[str, str, str]:
    source = _as_text(_field(row, str(data["source_field"]), ("source", "text", "article")))
    target = _as_text(_field(row, str(data["target_field"]), ("target", "summary", "abstract")))
    id_field = str(data.get("id_field", "id")).strip()
    identifier = ""
    if id_field:
        try:
            value = _field(row, id_field, ())
        except KeyError:
            value = None
        if value not in (None, ""):
            identifier = str(value)
    if not identifier:
        for fallback in ("id", "_id", "uid", "example_id", "_decoder_baseline_row_id"):
            value = row.get(fallback)
            if value not in (None, ""):
                identifier = str(value)
                break
    if bool(data.get("clean_text", True)):
        source = unicodedata.normalize("NFC", source).strip()
        target = unicodedata.normalize("NFC", target).strip()
    if not source or not target:
        raise ValueError("source and target must be non-empty")
    return identifier, source, target


def _ids(tokenizer: Any, text: str, *, add_special_tokens: bool, max_length: int | None = None) -> list[int]:
    kwargs: dict[str, Any] = {"add_special_tokens": add_special_tokens}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": int(max_length)})
    values = tokenizer(text, **kwargs)["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _prompt_parts(tokenizer: Any, source: str, data: Mapping[str, Any], source_length: int) -> list[int]:
    prefix = _ids(tokenizer, str(data.get("source_prefix", "")), add_special_tokens=True)
    suffix = _ids(tokenizer, str(data.get("prompt_suffix", "")), add_special_tokens=False)
    if not prefix:
        prefix = _ids(tokenizer, "", add_special_tokens=True)
    source_ids = _ids(tokenizer, source, add_special_tokens=False, max_length=source_length)
    return [*prefix, *source_ids, *suffix]


def encode_prompt(
    tokenizer: Any,
    source: str,
    data: Mapping[str, Any],
    *,
    max_total_length: int | None = None,
) -> list[int]:

    source_limit = int(data["max_source_length"])
    if max_total_length is not None:
        prefix = _ids(tokenizer, str(data.get("source_prefix", "")), add_special_tokens=True)
        suffix = _ids(tokenizer, str(data.get("prompt_suffix", "")), add_special_tokens=False)
        overhead = len(prefix) + len(suffix) or 1
        source_limit = min(source_limit, max(1, int(max_total_length) - overhead))
    return _prompt_parts(tokenizer, source, data, source_limit)


def _target_ids(tokenizer: Any, target: str, data: Mapping[str, Any]) -> list[int]:
    limit = int(data["max_target_length"])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    reserve = 1 if eos_id is not None else 0
    values = _ids(tokenizer, target, add_special_tokens=False, max_length=max(1, limit - reserve))
    if eos_id is not None and (not values or values[-1] != int(eos_id)):
        values.append(int(eos_id))
    if not values:
        raise ValueError("Target tokenization produced an empty sequence")
    return values[:limit]


class CausalSummarizationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        data: Mapping[str, Any],
        *,
        max_examples: int = 0,
        model_context_length: int | None = None,
    ) -> None:
        self.rows = read_jsonl(path, max_examples=max_examples)
        self.tokenizer = tokenizer
        self.data = data
        configured = int(data["max_sequence_length"])
        self.max_sequence_length = min(configured, int(model_context_length)) if model_context_length else configured
        if self.max_sequence_length < int(data["max_target_length"]):
            raise ValueError("Model context is too short for the configured target budget")

        self.length_estimates = []
        for row in self.rows:
            _, source, target = record_texts(row, self.data)
            self.length_estimates.append(max(1, len(source) + len(target)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        _, source, target = record_texts(self.rows[index], self.data)
        target_ids = _target_ids(self.tokenizer, target, self.data)
        prefix = _ids(self.tokenizer, str(self.data.get("source_prefix", "")), add_special_tokens=True)
        suffix = _ids(self.tokenizer, str(self.data.get("prompt_suffix", "")), add_special_tokens=False)
        if not prefix:
            prefix = _ids(self.tokenizer, "", add_special_tokens=True)
        available_source = self.max_sequence_length - len(target_ids) - len(prefix) - len(suffix)
        if available_source <= 0:
            raise ValueError("Prompt instruction and target leave no room for source tokens")
        source_limit = min(int(self.data["max_source_length"]), available_source)
        prompt_ids = [
            *prefix,
            *_ids(self.tokenizer, source, add_special_tokens=False, max_length=source_limit),
            *suffix,
        ]
        input_ids = [*prompt_ids, *target_ids]
        if len(input_ids) > self.max_sequence_length:
            raise RuntimeError("Causal example exceeded the context budget after truncation")
        labels = [-100] * len(prompt_ids) + target_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CausalCollator:
    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        width = max(int(feature["input_ids"].numel()) for feature in features)
        if self.pad_to_multiple_of > 1:
            multiple = self.pad_to_multiple_of
            width = ((width + multiple - 1) // multiple) * multiple
        input_ids = []
        attention = []
        labels = []
        for feature in features:
            length = int(feature["input_ids"].numel())
            pad = width - length
            input_ids.append(torch.cat([feature["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            attention.append(torch.cat([feature["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([feature["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention),
            "labels": torch.stack(labels),
        }


def left_pad_prompts(prompts: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Generation prompts must be non-empty")
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full((len(prompts), width), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        values = torch.tensor(prompt, dtype=torch.long)
        input_ids[row, width - len(prompt) :] = values
        attention[row, width - len(prompt) :] = 1
    return input_ids, attention
