from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def merge(shards: list[Path], output: Path) -> int:
    if not shards:
        raise ValueError("At least one evaluation shard is required")
    rows: dict[int, dict] = {}
    expected_total: int | None = None
    for rank, shard in enumerate(shards):
        metrics_path = shard.parent / "metrics.json"
        if not shard.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Missing shard output or metrics: {shard}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        total = metrics.get("test_examples_total")
        if type(total) is not int or total < 1:
            raise ValueError(f"Invalid test_examples_total in {metrics_path}")
        if expected_total is None:
            expected_total = total
        if total != expected_total or metrics.get("shard_rank") != rank or metrics.get("num_shards") != len(shards):
            raise ValueError(f"Shard metadata mismatch: {metrics_path}")
        with shard.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {shard}:{line_number}") from exc
                index = row.get("index") if isinstance(row, dict) else None
                if type(index) is not int or index < 0 or index >= total or index % len(shards) != rank:
                    raise ValueError(f"Invalid shard index at {shard}:{line_number}")
                if index in rows:
                    raise ValueError(f"Duplicate evaluation index: {index}")
                if any(not isinstance(row.get(key), str) for key in ("id", "source", "reference", "prediction")):
                    raise ValueError(f"Invalid prediction row at {shard}:{line_number}")
                rows[index] = row

    if len(rows) != expected_total or set(rows) != set(range(expected_total)):
        raise ValueError(f"Incomplete evaluation: {len(rows)} of {expected_total} examples")
    ids = [rows[index]["id"] for index in range(expected_total)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs across evaluation shards")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for index in range(expected_total):
            row = {key: value for key, value in rows[index].items() if key != "index"}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    return expected_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge T5Gemma test shards in original dataset order")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    count = merge(args.shards, args.output)
    print(f"Merged {count} test predictions -> {args.output}")


if __name__ == "__main__":
    main()
