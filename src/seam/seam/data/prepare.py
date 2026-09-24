from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from .normalization import detokenize
from .schema import iter_jsonl


def prepare_split(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_field: str = "text",
    target_field: str = "summary",
    id_field: str = "id",
    max_examples: int = 0,
    detokenize_text: bool = False,
) -> int:
    destination = Path(output_path)
    if destination.resolve() == Path(input_path).resolve():
        raise ValueError("prepare input and output must be different paths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for count, record in enumerate(
                iter_jsonl(
                    str(input_path),
                    source_field=source_field,
                    target_field=target_field,
                    id_field=id_field,
                ),
                start=1,
            ):
                record = replace(record, example_id=record.example_id or str(count))
                if detokenize_text:
                    record = replace(
                        record,
                        source=detokenize(record.source),
                        target=detokenize(record.target),
                    )
                row = record.as_dict()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if count % 1000 == 0:
                    logging.getLogger(__name__).info("prepare: %s records -> %s", count, destination)
                if max_examples > 0 and count >= max_examples:
                    break
        if not count:
            raise ValueError(f"No records found in {input_path}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--source-field", default="text")
    parser.add_argument("--target-field", default="summary")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--detokenize", action="store_true")
    args = parser.parse_args()
    count = prepare_split(**vars(args))
    print(f"prepared {count} records -> {args.output_path}")


if __name__ == "__main__":
    main()
