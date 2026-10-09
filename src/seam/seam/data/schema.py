from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _as_text(value: Any, separator: str = "\n") -> str:
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("text fields must be strings or lists of strings")
        return separator.join(item.strip() for item in value if item.strip())
    if not isinstance(value, str):
        raise ValueError("text fields must be strings or lists of strings")
    return value


@dataclass(frozen=True)
class CanonicalRecord:
    example_id: str
    source: str
    target: str

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        source_field: str = "text",
        target_field: str = "summary",
        id_field: str = "id",
        separator: str = "\n",
    ) -> "CanonicalRecord":
        def field(name: str) -> Any:
            value: Any = row
            for part in name.split("."):
                if not isinstance(value, Mapping) or part not in value:
                    raise KeyError(f"Missing field {name!r}")
                value = value[part]
            return value

        try:
            raw_source = field(source_field)
        except KeyError:
            if source_field == "text" and "source" in row:
                raw_source = row["source"]
            else:
                raise
        try:
            raw_target = field(target_field)
        except KeyError:
            if target_field == "summary" and "target" in row:
                raw_target = row["target"]
            else:
                raise
        source = _as_text(raw_source, separator).strip()
        target = _as_text(raw_target, separator).strip()
        if not source or not target:
            raise ValueError("source and target must be non-empty")
        raw_id = row.get(id_field, row.get("article_id", ""))
        example_id = str(raw_id) if raw_id not in (None, "") else ""
        return cls(example_id, source, target)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.example_id, "text": self.source, "summary": self.target}


def iter_jsonl(path: str, **kwargs: Any) -> Iterable[CanonicalRecord]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                if not isinstance(row, Mapping):
                    raise ValueError("record is not an object")
                yield CanonicalRecord.from_mapping(row, **kwargs)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid summarization record at {path}:{line_number}: {exc}") from exc
