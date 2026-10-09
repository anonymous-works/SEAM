from __future__ import annotations

import pytest
import torch

from led.config import load_config, validate_config
from led.evaluate import _first_token_global_mask


@pytest.mark.parametrize(
    ("dataset", "source_limit", "target_limit"),
    [
        ("pubmed", 4096, 512),
        ("arxiv", 8192, 512),
        ("booksum", 12288, 1024),
        ("govreport", 12288, 1024),
    ],
)
def test_dataset_limits_and_effective_batch(dataset: str, source_limit: int, target_limit: int) -> None:
    config = load_config(dataset, num_train_epochs=1)
    assert config["data"]["max_source_length"] == source_limit
    assert config["data"]["max_target_length"] == target_limit
    assert config["generation"]["max_new_tokens"] == target_limit
    train = config["training"]
    assert train["per_device_train_batch_size"] * train["gradient_accumulation_steps"] == 96


def test_global_attention_selects_first_visible_token() -> None:
    attention = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])
    mask = _first_token_global_mask(attention)
    assert mask.tolist() == [[1, 0, 0, 0], [0, 1, 0, 0]]


def test_epoch_count_is_required_for_training_config() -> None:
    config = load_config("pubmed", num_train_epochs=None)
    with pytest.raises(ValueError, match="actual training horizon"):
        validate_config(config, training=True)
