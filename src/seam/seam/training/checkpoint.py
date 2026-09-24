from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:

    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model


def architecture_spec(config: dict[str, Any]) -> dict[str, Any]:
    arch = config["architecture"]
    decoder = config["decoder"]
    spec = {
        "graph_version": "seam_full_memory_graph",
        "architecture": arch.get("name"),
        "controller_dim": int(arch.get("controller_dim", 0)),
        "depth_taps": int(arch.get("depth_taps", 0)),
        "depth_rank": int(arch.get("depth_rank", 0)),
        "depth_gate_max": float(arch.get("depth_gate_max", 0.0)),
        "feature_rank": int(arch.get("feature_rank", 0)),
        "feature_gate_max": float(arch.get("feature_gate_max", 0.0)),
        "focus_hidden": int(arch.get("focus_hidden", 0)),
        "focus_windows": tuple(int(value) for value in arch.get("focus_windows", ())),
        "focus_overlap": float(arch.get("focus_overlap", 0.0)),
        "focus_strength_max": float(arch.get("focus_strength_max", 0.0)),
        "temperature_min": float(arch.get("temperature_min", 0.0)),
        "temperature_max": float(arch.get("temperature_max", 0.0)),
        "cross_attention_every": int(decoder.get("cross_attention_every", 1)),
        "cross_gate_max": float(decoder.get("cross_gate_max", 0.0)),
    }
    spec["bridge_mode"] = str(arch.get("bridge_mode", "seam"))
    copy_config = decoder.get("source_copy", {})
    if copy_config.get("enabled", False):
        spec["source_copy"] = {"alignment": "char_overlap", "key_dim": int(copy_config.get("key_dim", 128))}
    return spec


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    *,
    epoch: int,
    step: int,
    best_metric: float | None = None,
    stage: str | None = None,
    stage_epoch: int | None = None,
    elapsed_train_seconds: float | None = None,
    scheduler: Any = None,
) -> None:
    model = _unwrap_model(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": best_metric,
        "stage": stage,
        "stage_epoch": stage_epoch,
        "elapsed_train_seconds": None if elapsed_train_seconds is None else float(elapsed_train_seconds),
        "architecture_spec": architecture_spec(config),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _embedding_shape_mismatches(
    checkpoint_model: dict[str, Any], model: torch.nn.Module
) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:

    current_model = _unwrap_model(model).state_dict()
    mismatches = []
    for key in (
        "encoder.model.embed_tokens.weight",
        "decoder.backbone.embed_tokens.weight",
        "decoder.lm_head.weight",
    ):
        checkpoint_tensor = checkpoint_model.get(key)
        current_tensor = current_model.get(key)
        if checkpoint_tensor is None or current_tensor is None:
            continue
        checkpoint_shape = tuple(int(value) for value in checkpoint_tensor.shape)
        current_shape = tuple(int(value) for value in current_tensor.shape)
        if checkpoint_shape != current_shape:
            mismatches.append((key, checkpoint_shape, current_shape))
    return mismatches


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    config: dict[str, Any] | None = None,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    scheduler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = _unwrap_model(model)
    if config is not None and state.get("architecture_spec") != architecture_spec(config):
        raise ValueError("Checkpoint architecture_spec does not match the active SEAM configuration")
    mismatches = _embedding_shape_mismatches(state["model"], model)
    if mismatches:
        details = "; ".join(
            f"{key}: checkpoint={checkpoint_shape}, active_model={current_shape}"
            for key, checkpoint_shape, current_shape in mismatches
        )
        raise ValueError(
            "Checkpoint/backbone vocabulary mismatch ("
            f"{details}). The checkpoint and evaluation config use different encoder or decoder tokenizers; "
            "for example, Qwen3-Embedding-0.6B has 151669 rows while Qwen3-0.6B has 151936. "
            "Evaluate with the resolved_config.yaml saved beside this checkpoint, and keep its "
            "model.encoder_name/model.decoder_name pair. Do not use strict=False or resize the embeddings."
        )
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    rng_state = state.get("rng_state")
    if rng_state and restore_rng:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].to(device="cpu"))
        if rng_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.to(device="cpu") for value in rng_state["cuda"]])
    return {
        key: state.get(key)
        for key in (
            "epoch",
            "step",
            "best_metric",
            "stage",
            "stage_epoch",
            "elapsed_train_seconds",
            "architecture_spec",
        )
    }
