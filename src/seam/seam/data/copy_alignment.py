from __future__ import annotations

from typing import Any

import torch

COPY_INPUT_KEYS = (
    "copy_token_ids",
    "copy_token_mask",
    "copy_encoder_indices",
    "copy_token_indices",
    "copy_alignment_weights",
)


def align_copy_tokens(source: str, prefix_length: int, encoder_offsets, tokenizer: Any) -> dict[str, list]:
    spans = [
        (i, start - prefix_length, end - prefix_length)
        for i, (start, end) in enumerate(encoder_offsets)
        if end > start >= prefix_length
    ]
    visible_end = max((end for _, _, end in spans), default=0)
    encoded = tokenizer(source[:visible_end], add_special_tokens=False, return_offsets_mapping=True)
    if "offset_mapping" not in encoded:
        raise ValueError("Source copy requires a fast decoder tokenizer with offset mapping")
    special = set(getattr(tokenizer, "all_special_ids", ()))
    special.update(
        getattr(tokenizer, name, None) for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id")
    )
    ids, enc_indices, token_indices, weights = [], [], [], []
    cursor = 0
    for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        if token in special or end <= start or (visible_end < len(source) and end >= visible_end):
            continue
        while cursor < len(spans) and spans[cursor][2] <= start:
            cursor += 1
        overlap = []
        covered = set()
        j = cursor
        while j < len(spans) and spans[j][1] < end:
            index, left, right = spans[j]
            left, right = max(left, start), min(right, end)
            if right > left:
                overlap.append((index, right - left))
                covered.update(range(left, right))
            j += 1
        if not overlap or any(not source[k].isspace() and k not in covered for k in range(start, end)):
            continue
        destination = len(ids)
        ids.append(int(token))
        total = sum(size for _, size in overlap)
        for index, size in overlap:
            enc_indices.append(index)
            token_indices.append(destination)
            weights.append(size / total)
    return dict(
        copy_token_ids=ids,
        copy_encoder_indices=enc_indices,
        copy_token_indices=token_indices,
        copy_alignment_weights=weights,
    )


def pad_copy_alignments(rows: list[dict[str, list]]) -> dict[str, torch.Tensor]:
    result = {}
    for key in ("copy_token_ids", "copy_encoder_indices", "copy_token_indices", "copy_alignment_weights"):
        width = max(1, max((len(row[key]) for row in rows), default=0))
        dtype = torch.float32 if key == "copy_alignment_weights" else torch.long
        tensor = torch.zeros(len(rows), width, dtype=dtype)
        for i, row in enumerate(rows):
            tensor[i, : len(row[key])] = torch.tensor(row[key], dtype=dtype)
        result[key] = tensor
    result["copy_token_mask"] = (
        torch.arange(result["copy_token_ids"].shape[1])[None, :]
        < torch.tensor([len(row["copy_token_ids"]) for row in rows])[:, None]
    )
    return result
