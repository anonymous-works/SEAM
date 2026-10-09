from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .bridge import SEAMBridge
from .decoder import QwenCrossDecoder
from .encoder import build_encoder, resolve_dtype
from .outputs import BridgeState, SEAMOutput


class SEAMModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder = build_encoder(config)
        decoder_cfg = config["model"]
        self.decoder = QwenCrossDecoder(
            str(decoder_cfg["decoder_name"]),
            config["decoder"],
            resolve_dtype(str(decoder_cfg.get("dtype", "float32"))),
            bool(decoder_cfg.get("gradient_checkpointing", True)),
            bool(decoder_cfg.get("trust_remote_code", True)),
            str(decoder_cfg.get("attention_implementation", "sdpa")),
        )
        self.bridge = SEAMBridge(self.encoder.hidden_size, int(self.decoder.config.hidden_size), config["architecture"])

    def encode_source(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_content_mask: torch.Tensor,
        decoder_prompt_ids: torch.Tensor,
        decoder_prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
        **copy_inputs: torch.Tensor,
    ) -> BridgeState:
        state = self.encoder(input_ids, attention_mask, source_content_mask)
        prompt_embeddings = self.decoder.embed_tokens(decoder_prompt_ids)
        bridge = self.bridge(state, prompt_embeddings, decoder_prompt_mask, output_budget)
        decoder_dtype = self.decoder.embed_tokens.weight.dtype
        if bridge.memory.dtype != decoder_dtype:
            bridge = BridgeState(
                bridge.memory.to(decoder_dtype),
                bridge.memory_mask,
                bridge.content_mask,
                bridge.source_bias,
                bridge.controller,
                None if bridge.value_memory is None else bridge.value_memory.to(decoder_dtype),
            )
        if self.decoder.source_copy is not None:
            if not copy_inputs:
                raise ValueError("Source copy is enabled but source-token alignment is missing")
            bridge.copy_state = self.decoder.source_copy.prepare(
                bridge.value_memory if bridge.value_memory is not None else bridge.memory,
                bridge.source_bias,
                bridge.content_mask,
                self.decoder.embed_tokens,
                **copy_inputs,
            )
        elif copy_inputs:
            raise ValueError("Copy alignment supplied to a decoder without source copy")
        return bridge

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_content_mask: torch.Tensor,
        decoder_prompt_ids: torch.Tensor,
        decoder_prompt_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_budget: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        **copy_inputs: torch.Tensor,
    ) -> SEAMOutput:
        if output_budget is None:
            output_budget = torch.full(
                (input_ids.shape[0],),
                int(self.config["generation"].get("max_new_tokens", 256)),
                device=input_ids.device,
                dtype=torch.float32,
            )
        bridge = self.encode_source(
            input_ids,
            attention_mask,
            source_content_mask,
            decoder_prompt_ids,
            decoder_prompt_mask,
            output_budget,
            **copy_inputs,
        )
        logits, _, loss_ce = self.decoder(
            decoder_input_ids,
            bridge.memory,
            bridge.memory_mask,
            bridge.source_bias,
            decoder_attention_mask,
            labels=labels,
            return_logits=return_logits,
            value_memory=bridge.value_memory,
            copy_state=bridge.copy_state,
        )
        return SEAMOutput(logits, loss_ce, loss_ce, bridge)
