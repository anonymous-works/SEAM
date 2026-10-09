from __future__ import annotations

import json
import os
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .normalization import detokenize
from .schema import CanonicalRecord


class JsonlSummarizationDataset(Dataset[CanonicalRecord]):
    def __init__(self, path: str | Path, data_config: dict[str, Any], *, max_examples: int = 0):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        self.data_config = dict(data_config)
        self.offsets = array("Q")
        self.length_estimates = array("Q")
        self._handle = None
        self._pid = None
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
                    self.length_estimates.append(len(line))
                    if max_examples > 0 and len(self.offsets) >= max_examples:
                        break
        if not self.offsets:
            raise ValueError(f"Dataset is empty: {self.path}")
        self[0]

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> CanonicalRecord:
        if self._pid != os.getpid() or self._handle is None:
            if self._handle is not None:
                self._handle.close()
            self._handle = self.path.open("rb")
            self._pid = os.getpid()
        self._handle.seek(self.offsets[index])
        row = json.loads(self._handle.readline())
        config = self.data_config
        record = CanonicalRecord.from_mapping(
            row,
            source_field=str(config.get("source_field", "text")),
            target_field=str(config.get("target_field", "summary")),
            id_field=str(config.get("id_field", "id")),
            separator=str(config.get("list_separator", "\n")),
        )
        if config.get("detokenize", False):
            record = replace(record, source=detokenize(record.source), target=detokenize(record.target))
        return record if record.example_id else replace(record, example_id=str(index + 1))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
