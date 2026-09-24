from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any, Sequence

import torch

from .copy_alignment import align_copy_tokens, pad_copy_alignments
from .schema import CanonicalRecord


def _token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("Tokenizer output must contain input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)):
        raise ValueError("Tokenizer output must be one sequence of integer token IDs")
    try:
        return [index(value) for value in encoded]
    except TypeError as exc:
        raise ValueError(
            "Tokenizer output must be one sequence of integer token IDs, not text or multiple sequences"
        ) from exc


def _ids(tokenizer: Any, text: str) -> list[int]:
    return _token_ids(tokenizer(text, add_special_tokens=False))


class SummarizationCollator:
    def __init__(
        self,
        encoder_tokenizer: Any,
        decoder_tokenizer: Any,
        data_config: dict[str, Any],
        *,
        source_copy: bool = False,
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.data = data_config
        self.include_targets = True
        self.source_copy = source_copy
        decoder_prompt = data_config.get("decoder_prompt", "")
        self.decoder_prompt = "" if decoder_prompt is None else str(decoder_prompt)
        self.decoder_chat_template = bool(data_config.get("decoder_chat_template", False))
        decoder_prefix = data_config.get("decoder_prefix", "")
        self.decoder_prefix = "" if decoder_prefix is None else str(decoder_prefix)
        self._prompt_cache: dict[str, list[int]] = {}

        self._prompt_ids = self._prompt_ids_for() if self.decoder_prompt.strip() else []
        self.max_source_length = int(data_config.get("max_source_length", 4096))
        self.max_target_length = int(data_config.get("max_target_length", 512))
        self.encoder_prefix = str(data_config.get("encoder_prefix", ""))
        self.pad_encoder = int(getattr(encoder_tokenizer, "pad_token_id", 0) or 0)
        self.pad_decoder = int(getattr(decoder_tokenizer, "pad_token_id", 0) or 0)

    def _prompt_ids_for(self) -> list[int]:

        cache_key = f"{self.decoder_prompt}\x00{self.decoder_prefix}\x00{self.decoder_chat_template}"
        cached = self._prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        instruction = self.decoder_prompt.strip()
        if self.decoder_chat_template:
            if not instruction:
                raise ValueError("decoder_chat_template requires a non-empty decoder_prompt")
            prompt = _token_ids(
                self.decoder_tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        else:
            prompt = _ids(self.decoder_tokenizer, instruction) if instruction else []
        prompt += _ids(self.decoder_tokenizer, self.decoder_prefix)
        if not prompt:
            start_token = getattr(self.decoder_tokenizer, "bos_token_id", None)
            if start_token is None:
                start_token = getattr(self.decoder_tokenizer, "eos_token_id", None)
            if start_token is None:
                raise ValueError("An empty decoder prompt requires a BOS or EOS start token")
            prompt = [int(start_token)]
        self._prompt_cache[cache_key] = prompt
        return prompt

    def _encode_source(self, record: CanonicalRecord, return_offsets: bool = False):
        try:
            encoded = self.encoder_tokenizer(
                self.encoder_prefix + record.source,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_source_length,
            )
            offsets = encoded["offset_mapping"]
        except (TypeError, KeyError, NotImplementedError) as exc:
            raise ValueError("SEAM requires a fast encoder tokenizer with offset mapping") from exc
        source = list(encoded["input_ids"])
        content = []
        for start, end in offsets:
            article = end > max(start, len(self.encoder_prefix))
            content.append(article)
        if not any(content):
            raise ValueError(f"No source content remains after truncation for {record.example_id}")
        return (source, content, offsets) if return_offsets else (source, content)

    @staticmethod
    def _pad(rows: Sequence[Sequence[int]], value: int) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(map(len, rows), default=1)
        ids = torch.full((len(rows), width), value, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for row, values in enumerate(rows):
            size = min(len(values), width)
            ids[row, :size] = torch.tensor(values[:size], dtype=torch.long)
            mask[row, :size] = True
        return ids, mask

    def __call__(self, records: Sequence[CanonicalRecord]) -> dict[str, Any]:
        encoder_rows: list[list[int]] = []
        content_rows: list[list[bool]] = []
        prompt_rows: list[list[int]] = []
        decoder_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        copy_rows = []
        for record in records:
            source, content, offsets = self._encode_source(record, return_offsets=True)
            if self.source_copy:
                copy_rows.append(
                    align_copy_tokens(record.source, len(self.encoder_prefix), offsets, self.decoder_tokenizer)
                )
            encoder_rows.append(source)
            content_rows.append(content)

            prompt = self._prompt_ids_for()
            target = (
                _ids(self.decoder_tokenizer, record.target)[: max(1, self.max_target_length - 1)]
                if self.include_targets
                else []
            )
            eos_target = getattr(self.decoder_tokenizer, "eos_token_id", None)
            if eos_target is not None and self.include_targets:
                target = target + [int(eos_target)]
            prompt_rows.append(prompt)
            decoder_rows.append(prompt + target if target else prompt)
            labels = [-100] * len(prompt) + target
            label_rows.append(labels[: len(prompt) + self.max_target_length])

        input_ids, attention_mask = self._pad(encoder_rows, self.pad_encoder)
        source_content_mask, _ = self._pad([[int(x) for x in row] for row in content_rows], 0)
        prompt_ids, prompt_mask = self._pad(prompt_rows, self.pad_decoder)
        decoder_input_ids, decoder_attention_mask = self._pad(decoder_rows, self.pad_decoder)
        labels, _ = self._pad(label_rows, -100)
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "source_content_mask": source_content_mask.bool(),
            "decoder_prompt_ids": prompt_ids,
            "decoder_prompt_mask": prompt_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "ids": [record.example_id for record in records],
            "sources": [record.source for record in records],
            "references": [record.target for record in records],
        }
        if self.source_copy:
            result.update(pad_copy_alignments(copy_rows))
        return result
