from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationRow:
    row_index: int
    identifier: Any
    prediction: str
    references: tuple[str, ...]
    source: str | None = None


def canonical(value: Any) -> Any:

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    return value


def _as_text(value: Any, field: str, row_number: int) -> str:
    if value is None:
        raise ValueError(f"{field!r} cannot be null at JSONL row {row_number}")
    return unicodedata.normalize("NFC", str(value))


def _references(value: Any, field: str, row_number: int) -> tuple[str, ...]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field!r} cannot be an empty list at JSONL row {row_number}")
        return tuple(_as_text(item, field, row_number) for item in value)
    return (_as_text(value, field, row_number),)


def load_jsonl(
    path: Path,
    *,
    prediction_field: str = "prediction",
    reference_field: str = "reference",
    source_field: str | None = None,
) -> list[EvaluationRow]:

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Predictions JSONL not found: {path}")

    rows: list[EvaluationRow] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            if prediction_field not in value or reference_field not in value:
                raise ValueError(f"Missing {prediction_field!r}/{reference_field!r} at {path}:{line_number}")

            row_index = len(rows)
            identifier = canonical(value.get("id", row_index))
            id_key = json.dumps(identifier, ensure_ascii=False, sort_keys=True)
            if id_key in seen_ids:
                raise ValueError(f"Duplicate example ID at {path}:{line_number}: {identifier!r}")
            seen_ids.add(id_key)
            source = None
            if source_field is not None:
                if source_field not in value:
                    raise ValueError(f"Missing {source_field!r} at {path}:{line_number}")
                source = _as_text(value[source_field], source_field, line_number)
            rows.append(
                EvaluationRow(
                    row_index=row_index,
                    identifier=identifier,
                    prediction=_as_text(value[prediction_field], prediction_field, line_number),
                    references=_references(value[reference_field], reference_field, line_number),
                    source=source,
                )
            )

    if not rows:
        raise ValueError(f"No prediction rows found: {path}")
    return rows


def metric_output_path(predictions: Path, output: Path | None, suffix: str) -> Path:

    return (output if output is not None else predictions.with_suffix(suffix)).expanduser().resolve()
