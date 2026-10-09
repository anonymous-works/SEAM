from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator

from .normalization import detokenize

SUPPORTED_DATASETS = ("pubmed", "arxiv", "cnndm", "wikilingua", "booksum", "govreport", "custom")
_LABEL_FILES = {"train": "train.label.jsonl", "validation": "val.label.jsonl", "test": "test.label.jsonl"}
_GENERIC_FILES = {
    "train": ("train.jsonl", "train.json", "train.txt"),
    "validation": ("val.jsonl", "validation.jsonl", "val.json", "validation.json", "val.txt", "validation.txt"),
    "test": ("test.jsonl", "test.json", "test.txt"),
}
_SOURCE_KEYS = (
    "text",
    "source",
    "document",
    "document_text",
    "article_text",
    "article",
    "chapter",
    "chapter_text",
    "report",
)
_TARGET_KEYS = (
    "summary",
    "target",
    "summary_text",
    "abstract_text",
    "abstract",
    "highlights",
    "highlight",
    "report_summary",
)
_ID_KEYS = (
    "id",
    "doc_id",
    "article_id",
    "report_id",
    "chapter_id",
    "paper_id",
    "book_id",
    "guid",
)
_TEXT_KEYS = {
    "source": (
        "title",
        "section_title",
        "heading",
        "text",
        "content",
        "paragraphs",
        "body",
        "sections",
        "subsections",
        "children",
    ),
    "target": (
        "title",
        "section_title",
        "summary",
        "summary_text",
        "text",
        "content",
        "paragraphs",
        "highlights",
        "highlight",
        "sections",
        "subsections",
        "children",
    ),
}
_NON_TEXT_KEYS = {
    "id",
    "doc_id",
    "article_id",
    "report_id",
    "chapter_id",
    "book_id",
    "paper_id",
    "guid",
    "depth",
    "type",
}


def _value_present(value: Any) -> bool:
    return value not in (None, "", []) and value != {}


def _get_field(row: dict[str, Any], key: str) -> Any:

    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = _get_field(row, key)
        if _value_present(value):
            return value
    return None


def _records_from_value(value: Any, path: Path, line_number: int = 1) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if len(value) == 1:
            key, wrapped = next(iter(value.items()))
            if key in {"data", "records", "examples", "items"} and isinstance(wrapped, list):
                value = wrapped
            else:
                yield value
                return
        else:
            yield value
            return
    if isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain objects")
            yield row
        return
    raise ValueError(f"{path}:{line_number} must contain JSON objects or arrays")


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:

    if path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                value = json.load(handle)
        except json.JSONDecodeError:
            pass
        else:
            yield from _records_from_value(value, path)
            return

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
            yield from _records_from_value(value, path, line_number)


def _split_candidate_groups(dataset: str, split: str) -> tuple[tuple[str, ...], ...]:
    generic = tuple((name,) for name in _GENERIC_FILES[split])
    if dataset == "govreport":
        suffix = "valid" if split == "validation" else split

        govreport = (
            (f"gao_{suffix}.jsonl", f"crs_{suffix}.jsonl"),
            (f"gao_{suffix}.json", f"crs_{suffix}.json"),
            (f"gao_{suffix}.jsonl",),
            (f"crs_{suffix}.jsonl",),
            (f"gao_{suffix}.json",),
            (f"crs_{suffix}.json",),
        )
        return generic + govreport
    if dataset in {"pubmed", "arxiv"}:
        return ((_LABEL_FILES[split],),) + generic
    return generic


def _find_split(input_dir: Path, dataset: str, split: str) -> tuple[Path, ...]:
    groups = _split_candidate_groups(dataset, split)
    locations = (input_dir, input_dir / "data", input_dir / "document")
    for location in locations:
        for group in groups:
            paths = tuple(location / name for name in group)
            if all(path.is_file() for path in paths):
                return paths
    tried = ", ".join(name for group in groups for name in group)
    raise FileNotFoundError(f"Missing {split} data in {input_dir}; tried {tried}")


def _copy_raw(source: Path, raw_dir: Path, split: str) -> Path:
    del split
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / source.name
    if source.resolve() == destination.resolve():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination


def _join_text(value: Any, separator: str, *, role: str) -> str:

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_join_text(item, separator, role=role) for item in value]
        return separator.join(part for part in parts if part)
    if isinstance(value, dict):
        parts: list[str] = []
        used: set[str] = set()
        for key in _TEXT_KEYS[role]:
            if key not in value or key in used or not _value_present(value[key]):
                continue
            part = _join_text(value[key], separator, role=role)
            if part:
                parts.append(part)
                used.add(key)
        if parts:
            return separator.join(parts)

        for key, item in value.items():
            if key in _NON_TEXT_KEYS or not _value_present(item):
                continue
            part = _join_text(item, separator, role=role)
            if part:
                parts.append(part)
        return separator.join(parts)
    return str(value).strip()


def _example_id(row: dict[str, Any], source: Path, dataset: str, split: str, index: int) -> str:
    if dataset == "booksum":
        book_id = _get_field(row, "book_id")
        summary_id = _first(row, ("summary_id", "summary_name"))
        if _value_present(book_id) and _value_present(summary_id):
            return f"{book_id}::{summary_id}"
        if _value_present(book_id):
            chapter_path = _get_field(row, "chapter_path")
            if _value_present(chapter_path):
                return f"{book_id}::{chapter_path}"

    raw_id = _first(row, _ID_KEYS)
    return _source_qualified_id(raw_id, source, dataset, f"{split}_{index:06d}")


def _source_qualified_id(raw_id: Any, source: Path, dataset: str, fallback: str) -> str:
    identifier = str(raw_id).strip() if _value_present(raw_id) else fallback
    if dataset == "govreport":
        stem = source.stem.lower()
        for prefix in ("gao", "crs"):
            if stem.startswith(prefix + "_") and not identifier.lower().startswith(prefix + "_"):
                identifier = f"{prefix.upper()}_{identifier}"
                break
    return identifier


def _convert(
    sources: Path | Iterable[Path],
    destination: Path,
    split: str,
    dataset: str,
    *,
    source_field: str | None = None,
    target_field: str | None = None,
    id_field: str | None = None,
    list_separator: str = "\n",
    detokenize_text: bool | None = None,
    allow_duplicate_ids: bool = False,
) -> dict[str, Any]:
    source_paths = (sources,) if isinstance(sources, Path) else tuple(sources)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    ids: set[str] = set()
    kept = skipped = 0
    should_detokenize = dataset in {"pubmed", "arxiv", "cnndm"} if detokenize_text is None else detokenize_text
    try:
        with temporary.open("w", encoding="utf-8") as output:
            global_index = 0
            for source in source_paths:
                for row in _iter_json_records(source):
                    source_value = _get_field(row, source_field) if source_field else _first(row, _SOURCE_KEYS)
                    target_value = _get_field(row, target_field) if target_field else _first(row, _TARGET_KEYS)
                    if source_value is None or target_value is None:
                        skipped += 1
                        global_index += 1
                        continue
                    text = _join_text(source_value, list_separator, role="source").strip()
                    summary = _join_text(target_value, list_separator, role="target").strip()
                    if should_detokenize:
                        text, summary = detokenize(text), detokenize(summary)
                    if not text or not summary:
                        skipped += 1
                        global_index += 1
                        continue
                    if id_field:
                        raw_id = _get_field(row, id_field)
                        example_id = _source_qualified_id(raw_id, source, dataset, f"{split}_{global_index:06d}")
                    else:
                        example_id = _example_id(row, source, dataset, split, global_index)
                    if example_id in ids:
                        if not allow_duplicate_ids:
                            raise ValueError(f"Duplicate id {example_id!r} in {source}")
                        base_identifier = example_id
                        suffix = 1
                        example_id = f"{base_identifier}::{global_index:06d}"
                        while example_id in ids:
                            suffix += 1
                            example_id = f"{base_identifier}::{global_index:06d}_{suffix}"
                    ids.add(example_id)
                    prepared = {
                        "id": example_id,
                        "text": text,
                        "summary": summary,
                        "task": "summarization",
                        "dataset": dataset,
                    }
                    output.write(json.dumps(prepared, ensure_ascii=False) + "\n")
                    kept += 1
                    global_index += 1
        if not kept:
            raise ValueError(f"No valid text/summary pairs found in {', '.join(str(path) for path in source_paths)}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"kept": kept, "skipped": skipped, "ids": ids}


def _register_sources(path: Path, split: str, connection: sqlite3.Connection) -> int:
    collisions = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                text = str(json.loads(raw)["text"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid prepared row at {path}:{line_number}") from exc
            existing = connection.execute("SELECT split FROM source_registry WHERE text = ?", (text,)).fetchone()
            if existing is not None:
                if existing[0] != split:
                    collisions += 1
                continue
            connection.execute("INSERT INTO source_registry(text, split) VALUES (?, ?)", (text, split))
    connection.commit()
    return collisions


def prepare_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    raw_copy_dir: str | Path | None = None,
    allow_cross_split_content: bool = False,
    source_field: str | None = None,
    target_field: str | None = None,
    id_field: str | None = None,
    list_separator: str = "\n",
    detokenize_text: bool | None = None,
    allow_duplicate_ids: bool = False,
) -> dict[str, Any]:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {', '.join(SUPPORTED_DATASETS)}")
    input_path = Path(input_dir).expanduser().resolve()
    processed = Path(output_dir).expanduser().resolve()
    raw_dir = Path(raw_copy_dir or processed.parent / "raw" / dataset).expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    processed.mkdir(parents=True, exist_ok=True)
    registry = processed / ".cross_split_sources.sqlite3"
    registry.unlink(missing_ok=True)
    connection = sqlite3.connect(registry)
    report: dict[str, Any] = {"dataset": dataset, "input_dir": str(input_path), "splits": {}}
    seen_ids: set[str] = set()
    try:
        connection.execute("CREATE TABLE source_registry (text TEXT PRIMARY KEY, split TEXT NOT NULL)")
        for split in ("train", "validation", "test"):
            sources = _find_split(input_path, dataset, split)
            copied = tuple(_copy_raw(source, raw_dir, split) for source in sources)
            destination = processed / ("validation.jsonl" if split == "validation" else f"{split}.jsonl")
            stats = _convert(
                copied,
                destination,
                split,
                dataset,
                source_field=source_field,
                target_field=target_field,
                id_field=id_field,
                list_separator=list_separator,
                detokenize_text=detokenize_text,
                allow_duplicate_ids=allow_duplicate_ids,
            )
            duplicate_ids = sorted(stats["ids"] & seen_ids)
            duplicate_sources = _register_sources(destination, split, connection)
            if (duplicate_ids or duplicate_sources) and not allow_cross_split_content:
                details = []
                if duplicate_ids:
                    details.append(f"duplicate_ids={duplicate_ids[:5]}")
                if duplicate_sources:
                    details.append(f"duplicate_sources={duplicate_sources}")
                raise ValueError("Cross-split content leakage: " + ", ".join(details))
            seen_ids.update(stats["ids"])
            stats.pop("ids")
            source_value: str | list[str] = str(sources[0]) if len(sources) == 1 else [str(path) for path in sources]
            copied_value: str | list[str] = str(copied[0]) if len(copied) == 1 else [str(path) for path in copied]
            report["splits"][split] = {
                **stats,
                "source_path": source_value,
                "source_paths": [str(path) for path in sources],
                "raw_copy": copied_value,
                "raw_copies": [str(path) for path in copied],
                "processed_path": str(destination),
                "duplicate_ids": len(duplicate_ids),
                "duplicate_sources": duplicate_sources,
            }
        report_path = processed / "preparation_report.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        return report
    finally:
        connection.close()
        registry.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare summarization datasets for SEAM")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-copy-dir", default=None)
    parser.add_argument("--source-field", default=None, help="Optional source field or dotted path")
    parser.add_argument("--target-field", default=None, help="Optional target field or dotted path")
    parser.add_argument("--id-field", default=None, help="Optional ID field or dotted path")
    parser.add_argument("--list-separator", default="\n")
    parser.add_argument(
        "--detokenize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize punctuation spacing (dataset defaults: PubMed/ArXiv/CNNDM enabled)",
    )
    parser.add_argument("--allow-duplicate-ids", action="store_true")
    parser.add_argument("--allow-cross-split-content", action="store_true")
    args = parser.parse_args()
    report = prepare_dataset(
        args.input_dir,
        args.output_dir,
        dataset=args.dataset,
        raw_copy_dir=args.raw_copy_dir,
        allow_cross_split_content=args.allow_cross_split_content,
        source_field=args.source_field,
        target_field=args.target_field,
        id_field=args.id_field,
        list_separator=args.list_separator,
        detokenize_text=args.detokenize,
        allow_duplicate_ids=args.allow_duplicate_ids,
    )
    for split, stats in report["splits"].items():
        print(f"{split}: {stats['kept']} examples (skipped {stats['skipped']}) -> {stats['processed_path']}")


if __name__ == "__main__":
    main()
