from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class FocusController(nn.Module):
    def __init__(self, source_hidden: int, prompt_hidden: int, controller_dim: int):
        super().__init__()
        self.source = nn.Sequential(nn.RMSNorm(source_hidden), nn.Linear(source_hidden, controller_dim, bias=False))
        self.prompt = nn.Sequential(nn.RMSNorm(prompt_hidden), nn.Linear(prompt_hidden, controller_dim, bias=False))
        self.scalar = nn.Sequential(nn.Linear(3, controller_dim), nn.SiLU(), nn.Linear(controller_dim, controller_dim))
        nn.init.xavier_uniform_(self.source[1].weight)
        nn.init.xavier_uniform_(self.prompt[1].weight)
        nn.init.xavier_uniform_(self.scalar[-1].weight, gain=0.01)
        nn.init.zeros_(self.scalar[-1].bias)
        self.output = nn.RMSNorm(controller_dim)

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
    ) -> torch.Tensor:
        if source.ndim != 3 or prompt.ndim != 3:
            raise ValueError("controller inputs must be [batch, tokens, hidden]")
        source_pool = masked_mean(source.float(), source_mask)
        prompt_pool = masked_mean(prompt.float(), prompt_mask)
        n = source_mask.sum(dim=1, keepdim=True).float()
        k = output_budget.reshape(-1, 1).float()
        scalar = torch.cat((torch.log1p(n), torch.log1p(k.clamp_min(0)), k / n.clamp_min(1.0)), dim=-1)
        return self.output((self.source(source_pool) + self.prompt(prompt_pool) + self.scalar(scalar)).float())
