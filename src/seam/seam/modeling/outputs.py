from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .source_copy import CopyState


@dataclass
class EncoderState:
    final: torch.Tensor
    taps: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor
    content_mask: torch.Tensor


@dataclass
class BridgeState:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    content_mask: torch.Tensor
    source_bias: torch.Tensor
    controller: torch.Tensor
    value_memory: Optional[torch.Tensor] = None
    copy_state: Optional[CopyState] = None


@dataclass
class SEAMOutput:
    logits: Optional[torch.Tensor]
    loss_ce: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]
    bridge: BridgeState
