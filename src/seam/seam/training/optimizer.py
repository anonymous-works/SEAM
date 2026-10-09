from __future__ import annotations

from typing import Any

import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model


def _component(name: str) -> str:
    if ".cross." in name or name.endswith("cross_gate") or ".cross_norm." in name or ".source_copy." in name:
        return "cross_attention"
    for component in ("encoder", "decoder", "bridge"):
        if name.startswith(component + "."):
            return component
    raise ValueError(f"Unclassified SEAM parameter: {name}")


def set_stage_trainability(model: torch.nn.Module, stage: str) -> None:
    if stage not in {"interface_warmup", "full_finetune"}:
        raise ValueError(f"Unknown SEAM training stage: {stage}")
    for name, parameter in _unwrap_model(model).named_parameters():
        parameter.requires_grad = stage == "full_finetune" or _component(name) in {"bridge", "cross_attention"}


def build_optimizer(model: torch.nn.Module, config: dict[str, Any], stage: str) -> torch.optim.Optimizer:
    training = config["training"]
    prefix = "warmup" if stage == "interface_warmup" else "full"
    groups = {}
    seen = set()
    for name, parameter in _unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"Duplicate optimizer parameter: {name}")
        seen.add(id(parameter))
        component = _component(name)
        decay = parameter.ndim > 1
        key = (component, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "name": component,
                "lr": float(training[f"{prefix}_{component}_lr"]),
                "weight_decay": float(training.get("weight_decay", 0.01)) if decay else 0.0,
            }
        groups[key]["params"].append(parameter)
    if not groups:
        raise ValueError(f"No trainable parameters for stage {stage}")
    fused = bool(training.get("fused_optimizer", True)) and all(
        parameter.device.type == "cuda" for group in groups.values() for parameter in group["params"]
    )
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.95), eps=1e-8, fused=fused)
