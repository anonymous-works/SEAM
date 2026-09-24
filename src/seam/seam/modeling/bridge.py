from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import FocusController
from .outputs import BridgeState, EncoderState


def _bounded_gate(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    return float(maximum) * torch.sigmoid(raw.float())


class SEAMBridge(nn.Module):
    def __init__(self, encoder_hidden: int, decoder_hidden: int, config: dict):
        super().__init__()
        self.bridge_mode = str(config.get("bridge_mode", "seam"))
        if self.bridge_mode not in {"seam", "direct_projection"}:
            raise ValueError("architecture.bridge_mode must be seam or direct_projection")
        self.encoder_hidden = int(encoder_hidden)
        self.decoder_hidden = int(decoder_hidden)
        self.controller_dim = int(config.get("controller_dim", 256))
        self.depth_taps = int(config.get("depth_taps", 4))
        self.depth_rank = int(config.get("depth_rank", 128))
        self.feature_rank = int(config.get("feature_rank", 256))
        self.focus_hidden = int(config.get("focus_hidden", 256))
        self.focus_windows = tuple(int(width) for width in config.get("focus_windows", (32, 128, 512)))
        self.focus_overlap = float(config.get("focus_overlap", 0.5))
        self.focus_strength_max = float(config.get("focus_strength_max", 1.0))
        self.temperature_min = float(config.get("temperature_min", 0.5))
        self.temperature_max = float(config.get("temperature_max", 2.0))

        self.value_anchored = config.get("name", "seam_shared_values") == "seam_full_memory"
        if self.depth_taps < 0 or self.depth_rank <= 0 or self.feature_rank <= 0:
            raise ValueError("SEAM ranks and depth_taps must be non-negative/positive")
        if not self.focus_windows or any(width <= 0 for width in self.focus_windows):
            raise ValueError("SEAM focus_windows must be positive")
        if not 0.0 <= self.focus_overlap < 1.0:
            raise ValueError("SEAM focus_overlap must be in [0, 1)")

        if self.bridge_mode == "direct_projection":
            if self.encoder_hidden == self.decoder_hidden:
                self.direct_projection: nn.Module = nn.Identity()
            else:
                self.direct_projection = nn.Linear(self.encoder_hidden, self.decoder_hidden, bias=False)
                nn.init.orthogonal_(self.direct_projection.weight)
            self.value_anchored = False
            return

        self.controller = FocusController(self.encoder_hidden, self.decoder_hidden, self.controller_dim)
        if self.encoder_hidden == self.decoder_hidden:
            self.base_projection: nn.Module = nn.Identity()
        else:
            self.base_projection = nn.Linear(self.encoder_hidden, self.decoder_hidden, bias=False)
            nn.init.orthogonal_(self.base_projection.weight)

        if self.depth_taps > 1:
            self.depth_router = nn.Linear(self.controller_dim, self.depth_taps, bias=False)
            nn.init.zeros_(self.depth_router.weight)
            self.depth_content_score = nn.Linear(self.encoder_hidden, 1, bias=False)
            nn.init.zeros_(self.depth_content_score.weight)
            self.depth_norms = nn.ModuleList(nn.RMSNorm(self.encoder_hidden) for _ in range(self.depth_taps))
            self.depth_down = nn.Linear(self.encoder_hidden, self.depth_rank, bias=False)
            self.depth_out = nn.Linear(self.depth_rank, self.encoder_hidden, bias=False)
            nn.init.zeros_(self.depth_out.weight)
            self.depth_gate_raw = nn.Linear(self.controller_dim, 1)
            nn.init.zeros_(self.depth_gate_raw.weight)
            nn.init.constant_(
                self.depth_gate_raw.bias,
                float(
                    self._inverse_bounded_init(config.get("depth_gate_init", 0.02), config.get("depth_gate_max", 0.15))
                ),
            )
        self.depth_gate_max = float(config.get("depth_gate_max", 0.15))

        self.feature_norm = nn.RMSNorm(self.encoder_hidden)
        self.feature_down = nn.Linear(self.encoder_hidden, self.feature_rank, bias=False)
        self.feature_up = nn.Linear(self.feature_rank, self.decoder_hidden, bias=False)
        nn.init.zeros_(self.feature_up.weight)
        feature_gate_raw = self._inverse_bounded_init(
            config.get("feature_gate_init", 0.02), config.get("feature_gate_max", 0.20)
        )
        self.feature_gate_raw = nn.Linear(self.controller_dim, self.decoder_hidden)
        nn.init.zeros_(self.feature_gate_raw.weight)
        nn.init.constant_(self.feature_gate_raw.bias, float(feature_gate_raw))
        self.feature_gate_max = float(config.get("feature_gate_max", 0.20))

        self.focus_query = nn.Linear(self.controller_dim, self.focus_hidden, bias=False)
        self.focus_norm = nn.RMSNorm(self.decoder_hidden)
        self.focus_key = nn.Linear(self.decoder_hidden, self.focus_hidden, bias=False)
        self.focus_output = nn.Linear(self.focus_hidden, 1, bias=False)
        nn.init.zeros_(self.focus_output.weight)
        self.focus_scale = nn.Linear(self.controller_dim, len(self.focus_windows), bias=True)
        nn.init.zeros_(self.focus_scale.weight)
        nn.init.zeros_(self.focus_scale.bias)
        self.focus_scale_embedding = nn.Parameter(torch.zeros(len(self.focus_windows), self.focus_hidden))
        self.focus_strength_raw = nn.Linear(self.controller_dim, 1)
        nn.init.zeros_(self.focus_strength_raw.weight)
        nn.init.constant_(
            self.focus_strength_raw.bias,
            float(self._inverse_bounded_init(config.get("focus_strength_init", 0.10), self.focus_strength_max)),
        )
        temperature = float(config.get("temperature_init", 1.0))
        self.temperature_raw = nn.Linear(self.controller_dim, 1)
        nn.init.zeros_(self.temperature_raw.weight)
        nn.init.constant_(
            self.temperature_raw.bias,
            self._inverse_unit_interval(temperature, self.temperature_min, self.temperature_max),
        )

    @staticmethod
    def _inverse_bounded_init(value: float, maximum: float) -> torch.Tensor:
        if not 0 < float(value) < float(maximum):
            raise ValueError("Bounded gates require 0 < init < maximum")
        return torch.tensor(math.log(value / (float(maximum) - value)), dtype=torch.float32)

    @staticmethod
    def _inverse_unit_interval(value: float, minimum: float, maximum: float) -> float:
        if not 0 < minimum < value < maximum:
            raise ValueError("Require 0 < temperature_min < temperature_init < temperature_max")
        ratio = (float(value) - float(minimum)) / (float(maximum) - float(minimum))
        return math.log(ratio / (1.0 - ratio))

    def _temperature(self, controller: torch.Tensor) -> torch.Tensor:
        return self.temperature_min + (self.temperature_max - self.temperature_min) * torch.sigmoid(
            self.temperature_raw(controller.float()).squeeze(-1).float()
        )

    def _depth_weights(self, normalized_taps: list[torch.Tensor], controller: torch.Tensor) -> torch.Tensor:
        scores = torch.cat([self.depth_content_score(state) for state in normalized_taps], dim=-1)
        scores = scores + self.depth_router(controller.float())[:, None, :]
        return torch.softmax(scores.float(), dim=-1)

    def _focus_prior(self, memory: torch.Tensor, content_mask: torch.Tensor, controller: torch.Tensor) -> torch.Tensor:
        batch, length, _ = memory.shape
        content = content_mask.bool()
        temperature = self._temperature(controller).view(batch, 1, 1)
        content_count = content.sum(dim=-1, keepdim=True)
        content_index = (content.long().cumsum(-1) - 1).clamp_min(0)
        compact = torch.zeros_like(memory, dtype=torch.float32).scatter_add(
            1, content_index[..., None].expand_as(memory), memory.float() * content[..., None]
        )
        cumulative = F.pad(compact.cumsum(1), (0, 0, 1, 0))
        priors = []
        valid_scales = []
        for scale_index, width in enumerate(self.focus_windows):
            stride = max(1, int(round(width * (1.0 - self.focus_overlap))))
            last_start = (content_count - width).clamp_min(0)
            regular = torch.arange(0, max(1, length - width + 1), stride, device=memory.device)
            regular = regular[None, :].expand(batch, -1)
            starts = torch.cat((regular, last_start), dim=1)
            valid = torch.cat((regular <= last_start, last_start.remainder(stride) != 0), dim=1)
            ends = torch.minimum(starts + width, content_count)
            starts = torch.minimum(starts, content_count)
            counts = ends - starts
            valid = valid & (counts > 0)
            scale_available = valid.sum(dim=-1) >= 2

            def gather_at(indices: torch.Tensor) -> torch.Tensor:
                return cumulative.gather(1, indices[..., None].expand(-1, -1, memory.shape[-1]))

            pooled = gather_at(ends) - gather_at(starts)
            pooled = pooled / counts.unsqueeze(-1).clamp_min(1.0)
            region_features = (
                self.focus_key(self.focus_norm(pooled))
                + self.focus_query(controller.float())[:, None, :]
                + self.focus_scale_embedding[scale_index]
            )
            region_score = self.focus_output(F.silu(region_features)).squeeze(-1).float()
            valid = valid & scale_available[:, None]
            mean = (region_score * valid).sum(dim=-1, keepdim=True) / valid.sum(dim=-1, keepdim=True).clamp_min(
                1
            ).float()
            delta = region_score - mean
            rms = torch.sqrt(
                (delta.square() * valid).sum(dim=-1, keepdim=True)
                / valid.sum(dim=-1, keepdim=True).clamp_min(1).float()
                + 1.0
            )
            normalized = (delta / rms).masked_fill(~valid, 0.0)

            def overlap_add(values: torch.Tensor) -> torch.Tensor:
                differences = values.new_zeros(batch, length + 1)
                differences = differences.scatter_add(1, starts, values).scatter_add(1, ends, -values)
                return differences.cumsum(-1)[:, :length]

            token_values = overlap_add(normalized)
            token_denominator = overlap_add(valid.float()).clamp_min(1.0)
            priors.append((token_values / token_denominator).gather(1, content_index))
            valid_scales.append(scale_available)
        scale_logits = self.focus_scale(controller.float()).float().to(memory.device)
        scale_mask = torch.stack(valid_scales, dim=-1)
        no_valid = ~scale_mask.any(dim=-1)
        safe_scale_mask = scale_mask | no_valid[:, None]
        scale_weights = torch.softmax(
            scale_logits.masked_fill(~safe_scale_mask, torch.finfo(scale_logits.dtype).min), dim=-1
        ).masked_fill(no_valid[:, None], 0.0)
        focus = torch.stack(priors, dim=1)
        focus = (focus * scale_weights[:, :, None]).sum(dim=1)
        focus = focus.masked_fill(~content, 0.0)
        strength = _bounded_gate(self.focus_strength_raw(controller.float()).squeeze(-1), self.focus_strength_max)
        focus_temperature = temperature[:, 0, 0].to(focus.dtype).clamp_min(1.0e-6)[:, None]
        prior = strength[:, None].to(focus.dtype) * torch.tanh(focus / focus_temperature)
        content_count = content_count.float()
        has_content = content_count > 0
        log_mean_exp = (
            torch.logsumexp(prior.float().masked_fill(~content, torch.finfo(torch.float32).min), dim=-1, keepdim=True)
            - content_count.clamp_min(1.0).log()
        )
        log_mean_exp = torch.where(has_content, log_mean_exp, torch.zeros_like(log_mean_exp))
        prior = (prior.float() - log_mean_exp).to(focus.dtype).masked_fill(~content, 0.0)
        return prior

    def forward(
        self,
        encoder_state: EncoderState,
        prompt_embeddings: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
    ) -> BridgeState:
        final = encoder_state.final
        if final.ndim != 3:
            raise ValueError("encoder_state.final must be [batch, source_tokens, hidden]")
        if encoder_state.attention_mask.shape != final.shape[:2] or encoder_state.content_mask.shape != final.shape[:2]:
            raise ValueError("encoder masks must match encoder_state.final")
        if prompt_embeddings.shape[:2] != prompt_mask.shape:
            raise ValueError("prompt mask must match prompt embeddings")
        if len(encoder_state.taps) != self.depth_taps:
            raise ValueError(f"expected {self.depth_taps} encoder taps, got {len(encoder_state.taps)}")
        content = encoder_state.content_mask.bool() & encoder_state.attention_mask.bool()

        if self.bridge_mode == "direct_projection":
            memory_mask = encoder_state.attention_mask.bool()
            memory = self.direct_projection(final.float())
            memory = memory.masked_fill(~memory_mask.unsqueeze(-1), 0)
            source_bias = torch.zeros(final.shape[:2], device=final.device, dtype=torch.float32).masked_fill(
                ~content, 0.0
            )
            controller = torch.zeros(final.shape[0], self.controller_dim, device=final.device, dtype=torch.float32)
            return BridgeState(memory, memory_mask, content, source_bias, controller, None)

        controller = self.controller(final, content, prompt_embeddings, prompt_mask.bool(), output_budget)
        refined = final.float()

        if self.depth_taps > 1:
            normalized_taps = [norm(state.float()) for norm, state in zip(self.depth_norms, encoder_state.taps)]
            depth_weights = self._depth_weights(normalized_taps, controller)
            mixed = sum(depth_weights[..., index, None] * state for index, state in enumerate(normalized_taps))
            depth_delta = self.depth_out(F.silu(self.depth_down(mixed - normalized_taps[-1])))
            depth_gate = _bounded_gate(self.depth_gate_raw(controller.float()).squeeze(-1), self.depth_gate_max)
            refined = refined + depth_gate[:, None, None] * depth_delta
        feature_delta = self.feature_up(F.silu(self.feature_down(self.feature_norm(refined))))
        feature_gate = _bounded_gate(self.feature_gate_raw(controller.float()), self.feature_gate_max)
        memory = self.base_projection(refined) + feature_gate[:, None, :] * feature_delta
        source_bias = self._focus_prior(memory, content, controller)
        memory = memory.masked_fill(~encoder_state.attention_mask.bool().unsqueeze(-1), 0)
        source_bias = source_bias.masked_fill(~content, 0.0)
        value_memory = None
        if self.value_anchored:
            value_memory = self.base_projection(final.float()).masked_fill(
                ~encoder_state.attention_mask.bool().unsqueeze(-1), 0
            )
        return BridgeState(memory, encoder_state.attention_mask.bool(), content, source_bias, controller, value_memory)
