from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn

from .outputs import EncoderState


def resolve_dtype(name: str) -> torch.dtype:
    values = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in values:
        raise ValueError(f"Unsupported dtype: {name}")
    return values[name]


class PretrainedEncoderAdapter(nn.Module):
    def __init__(
        self,
        name: str,
        depth_taps: int,
        dtype: torch.dtype,
        trust_remote_code: bool = True,
        attention_implementation: str = "sdpa",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.model_name = str(name)
        self.depth_taps = int(depth_taps)
        if self.model_name == "__tiny__":
            from transformers import Qwen3Config, Qwen3Model

            raw_config = Qwen3Config(
                vocab_size=128,
                hidden_size=24,
                intermediate_size=48,
                num_hidden_layers=max(2, self.depth_taps),
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=6,
                max_position_embeddings=256,
                attention_dropout=0.0,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            self.model = Qwen3Model(raw_config).to(dtype=dtype)
        else:
            raw_config = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
            self.model = AutoModel.from_pretrained(
                name,
                config=raw_config,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attention_implementation,
            )
        self.hidden_size = int(getattr(raw_config, "hidden_size", getattr(raw_config, "d_model", 0)))
        if self.hidden_size <= 0:
            raise ValueError("Could not infer encoder hidden size")
        self.model.config.output_hidden_states = False
        self.model.config.use_cache = False
        self._supports_use_cache = "use_cache" in inspect.signature(self.model.forward).parameters
        self._tap_layers: tuple[nn.Module, ...] = ()
        if self.depth_taps > 1:
            layers = None
            for path in ("layers", "encoder.layer", "encoder.layers", "model.layers"):
                current = self.model
                for part in path.split("."):
                    current = getattr(current, part, None)
                if isinstance(current, (nn.ModuleList, list)):
                    layers = current
                    break
            if layers is None or len(layers) < self.depth_taps:
                raise ValueError(
                    "Encoder cannot expose the requested depth taps; use depth_taps=1 or an explicit adapter"
                )
            self._tap_layers = tuple(layers[-self.depth_taps : -1])
        if gradient_checkpointing:
            if not getattr(self.model, "supports_gradient_checkpointing", False):
                raise ValueError("Encoder does not support gradient checkpointing; disable it explicitly")
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, content_mask: torch.Tensor
    ) -> EncoderState:
        if input_ids.shape != attention_mask.shape or input_ids.shape != content_mask.shape:
            raise ValueError("encoder masks must match input_ids")
        captured = {}
        handles = []
        for index, layer in enumerate(self._tap_layers):

            def capture(module, inputs, output, key=index):
                captured[key] = output[0] if isinstance(output, tuple) else output

            handles.append(layer.register_forward_hook(capture))
        try:
            options = {"use_cache": False} if self._supports_use_cache else {}
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                **options,
            )
        finally:
            for handle in handles:
                handle.remove()
        final = output.last_hidden_state
        if self.depth_taps:
            if len(captured) != self.depth_taps - 1:
                raise ValueError("Encoder depth hooks did not execute exactly once per requested layer")
            taps = tuple(captured[index] for index in range(self.depth_taps - 1)) + (final,)
        else:
            taps = ()
        return EncoderState(final, taps, attention_mask.bool(), content_mask.bool())


def build_encoder(config: dict[str, Any]) -> PretrainedEncoderAdapter:
    model = config["model"]
    architecture = config["architecture"]
    return PretrainedEncoderAdapter(
        str(model["encoder_name"]),
        int(architecture.get("depth_taps", 0)),
        resolve_dtype(str(model.get("dtype", "float32"))),
        bool(model.get("trust_remote_code", True)),
        str(model.get("attention_implementation", "sdpa")),
        bool(model.get("gradient_checkpointing", True)),
    )
