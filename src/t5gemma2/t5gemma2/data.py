from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SummarizationRecord:
    identifier: str
    source: str
    target: str


_DETOKENIZE_SLOW_RE = re.compile(
    r"(?:``|''|[^\S \n]|\s{2,}|^\s|\s$|\n\s|\s\n|"
    r"\s+[,.;:!?%]|\s+n['’]t\b|\s+['’](?:s|re|ve|ll|d|m)\b|"
    r"[\(\[\{]\s+|\s+[\)\]\}])",
    flags=re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?%])")
_OPEN_BRACKET_SPACE_RE = re.compile(r"([\(\[\{])\s+")
_CLOSE_BRACKET_SPACE_RE = re.compile(r"\s+([\)\]\}])")
_CONTRACTION_NT_RE = re.compile(r"\s+n['’]t\b", flags=re.IGNORECASE)
_CONTRACTION_SUFFIX_RE = re.compile(r"\s+(['’](?:s|re|ve|ll|d|m))\b", flags=re.IGNORECASE)


def _detokenize_fast_path_is_safe(text: str) -> bool:

    if not text or _DETOKENIZE_SLOW_RE.search(text):
        return False

    return text.isascii() or unicodedata.is_normalized("NFKC", text)


def _as_text(value: Any, separator: str) -> str:
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("text fields must be strings or lists of strings")
        return separator.join(item.strip() for item in value if item.strip())
    if not isinstance(value, str):
        raise ValueError("text fields must be strings or lists of strings")
    return value


def detokenize(text: str) -> str:

    if _detokenize_fast_path_is_safe(text):
        return text

    text = unicodedata.normalize("NFKC", text).replace("``", '"').replace("''", '"')
    lines = []
    for line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", line).strip()
        line = _PUNCT_SPACE_RE.sub(r"\1", line)
        line = _OPEN_BRACKET_SPACE_RE.sub(r"\1", line)
        line = _CLOSE_BRACKET_SPACE_RE.sub(r"\1", line)
        line = _CONTRACTION_NT_RE.sub("n't", line)
        line = _CONTRACTION_SUFFIX_RE.sub(r"\1", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _field(row: Mapping[str, Any], name: str) -> Any:
    value: Any = row
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(name)
        value = value[part]
    return value


def canonicalize_record(row: Mapping[str, Any], data: Mapping[str, Any], *, index: int) -> SummarizationRecord:

    source_field = str(data.get("source_field", "text"))
    target_field = str(data.get("target_field", "summary"))
    id_field = str(data.get("id_field", "id"))
    separator = str(data.get("list_separator", "\n"))

    try:
        raw_source = _field(row, source_field)
    except KeyError:
        if source_field == "text" and "source" in row:
            raw_source = row["source"]
        else:
            raise KeyError(f"Missing field {source_field!r}")
    try:
        raw_target = _field(row, target_field)
    except KeyError:
        if target_field == "summary" and "target" in row:
            raw_target = row["target"]
        else:
            raise KeyError(f"Missing field {target_field!r}")

    source = _as_text(raw_source, separator).strip()
    target = _as_text(raw_target, separator).strip()
    if bool(data.get("detokenize", False)):
        source, target = detokenize(source), detokenize(target)
    source, target = source.strip(), target.strip()
    if not source or not target:
        raise ValueError("source and target must be non-empty")

    raw_id = row.get(id_field, row.get("article_id", ""))
    identifier = str(raw_id) if raw_id not in (None, "") else str(index + 1)
    return SummarizationRecord(identifier=identifier, source=source, target=target)


def load_summarization_jsonl(
    path: str | Path,
    data: Mapping[str, Any],
    *,
    limit: int = -1,
) -> list[SummarizationRecord]:

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[SummarizationRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if limit > 0 and len(records) >= limit:
                break
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            try:
                record = canonicalize_record(row, data, index=len(records))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid summarization record at {path}:{line_number}: {exc}") from exc
            if record.identifier in seen:
                raise ValueError(f"Duplicate id {record.identifier!r} at {path}:{line_number}")
            seen.add(record.identifier)
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def resolve_path(value: str | Path, *, base: Path) -> Path:

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    return (base / path).resolve()
