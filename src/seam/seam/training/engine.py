from __future__ import annotations

import json
import logging
import math
import random
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist

from ..data.copy_alignment import COPY_INPUT_KEYS
from .checkpoint import load_checkpoint, save_checkpoint
from .optimizer import build_optimizer, set_stage_trainability

LOGGER = logging.getLogger("seam.train")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_bar(fraction: float, width: int = 18) -> str:
    fraction = min(1.0, max(0.0, float(fraction)))
    filled = int(round(width * fraction))
    if filled >= width:
        return "[" + "=" * width + "]"
    if filled == 0:
        return "[" + "." * width + "]"
    return "[" + "=" * (filled - 1) + ">" + "." * (width - filled) + "]"


def _stage_label(stage: str) -> str:
    return {"interface_warmup": "warmup", "full_finetune": "full"}.get(stage, stage)


def _peak_vram_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)


class SEAMTrainer:
    def __init__(self, model: torch.nn.Module, config: dict[str, Any], device: torch.device | str):
        self.model = model.to(device=device, dtype=torch.float32)
        self.config = config
        self.device = torch.device(device)
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main_process = self.rank == 0
        if not self.is_main_process:
            LOGGER.setLevel(logging.WARNING)
        self.use_bf16 = self.device.type == "cuda" and config["model"].get("compute_dtype", "bfloat16") == "bfloat16"
        LOGGER.info(
            "precision | parameters=FP32 | optimizer=FP32 | compute=%s", "BF16 autocast" if self.use_bf16 else "FP32"
        )
        self.global_step = 0
        self.best_metric: float | None = None
        self.stage_optimizer_step = 0
        self.epoch = 0
        self.scheduler = None
        self.metrics_path = Path(self.config["experiment"]["output_dir"]) / "training_metrics.jsonl"
        self._fit_started_at: float | None = None
        self._elapsed_before_fit = 0.0
        self._fit_start_global_step = 0

    def _write_metric(self, record: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _elapsed_train_seconds(self) -> float:
        active = 0.0 if self._fit_started_at is None else time.monotonic() - self._fit_started_at
        return self._elapsed_before_fit + max(0.0, active)

    def _run_epoch(
        self,
        loader: Iterable[dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        stage: str,
        train: bool,
        global_epoch: int | None = None,
        total_epochs: int | None = None,
        total_training_steps: int | None = None,
    ) -> dict[str, float]:
        self.model.train(train)
        accum = int(self.config["training"]["gradient_accumulation_steps"])
        ce_sum = torch.zeros((), device=self.device)
        token_total = 0
        iterator = iter(loader)
        epoch_step = 0
        epoch_steps = math.ceil(len(loader) / accum) if train else len(loader)
        global_epoch = int(global_epoch or self.epoch or 1)
        total_epochs = int(total_epochs or global_epoch)
        total_training_steps = int(total_training_steps or epoch_steps * total_epochs)
        if train and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        epoch_started_at = time.monotonic()
        while window := list(islice(iterator, accum if train else 1)):
            started = time.monotonic()
            counts = [int(raw["labels"][:, 1:].ne(-100).sum()) for raw in window]
            window_tokens = sum(counts)
            if self.distributed:
                token_count = torch.tensor(window_tokens, device=self.device, dtype=torch.long)
                dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
                window_tokens = int(token_count.item())
            if train:
                optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros_like(ce_sum)
            for raw_batch, tokens in zip(window, counts):
                batch = _move(raw_batch, self.device)
                with torch.set_grad_enabled(train):
                    with torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext():
                        output = self.model(
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["source_content_mask"],
                            batch["decoder_prompt_ids"],
                            batch["decoder_prompt_mask"],
                            batch["decoder_input_ids"],
                            batch.get("decoder_attention_mask"),
                            batch.get("labels"),
                            return_logits=False,
                            **{key: batch[key] for key in COPY_INPUT_KEYS if key in batch},
                        )
                        if output.loss_ce is None:
                            raise RuntimeError("SEAM training requires decoder labels")

                        log_scale = tokens / max(1, window_tokens)
                        scale = log_scale
                        if self.distributed:
                            scale *= self.world_size
                        loss = output.loss_ce * scale
                    if train:
                        loss.backward()
                step_loss += output.loss_ce.detach() * log_scale
                ce_sum += output.loss_ce.detach() * tokens
                token_total += tokens
                del output, batch, loss
            if train:
                grad = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.config["training"]["max_grad_norm"]), error_if_nonfinite=True
                )
                learning_rates = ",".join(
                    dict.fromkeys(
                        f"{group.get('name', i)}:{group['lr']:.2e}" for i, group in enumerate(optimizer.param_groups)
                    )
                )
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.stage_optimizer_step += 1
                epoch_step += 1
                if self.distributed:
                    dist.all_reduce(step_loss, op=dist.ReduceOp.SUM)
                if self.global_step % int(self.config["training"]["log_every_steps"]) == 0:
                    window_elapsed = time.monotonic() - started
                    epoch_elapsed = time.monotonic() - epoch_started_at
                    examples = sum(int(raw["input_ids"].shape[0]) for raw in window)
                    epoch_progress = epoch_step / max(1, epoch_steps)
                    total_progress = (global_epoch - 1 + epoch_progress) / max(1, total_epochs)
                    schedule_step = max(0, global_epoch - 1) * epoch_steps + epoch_step
                    run_step = self.global_step - self._fit_start_global_step
                    epoch_eta = epoch_elapsed * (1.0 - epoch_progress) / max(epoch_progress, 1.0e-9)
                    remaining_steps = max(0, total_epochs - global_epoch) * epoch_steps + max(
                        0, epoch_steps - epoch_step
                    )
                    total_eta = epoch_elapsed / max(1, epoch_step) * remaining_steps
                    total_elapsed = self._elapsed_train_seconds()
                    peak_vram_gib = _peak_vram_gib(self.device)
                    record = {
                        "type": "step",
                        "stage": stage,
                        "epoch": global_epoch,
                        "step": self.global_step,
                        "run_step": run_step,
                        "schedule_step": schedule_step,
                        "epoch_step": epoch_step,
                        "epoch_steps": epoch_steps,
                        "epoch_progress": round(epoch_progress, 6),
                        "epoch_percent": round(100.0 * epoch_progress, 3),
                        "total_progress": round(total_progress, 6),
                        "total_percent": round(100.0 * total_progress, 3),
                        "ce": round(float(step_loss), 6),
                        "grad_norm": round(float(grad), 6),
                        "learning_rate": {
                            group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)
                        },
                        "seconds": round(window_elapsed, 4),
                        "window_seconds": round(window_elapsed, 4),
                        "epoch_elapsed_seconds": round(epoch_elapsed, 4),
                        "total_elapsed_seconds": round(total_elapsed, 4),
                        "epoch_eta_seconds": round(epoch_eta, 4),
                        "total_eta_seconds": round(total_eta, 4),
                        "peak_vram_gib": peak_vram_gib,
                        "examples": examples,
                        "tokens": window_tokens,
                        "examples_per_second": round(examples / max(window_elapsed, 1e-9), 4),
                        "tokens_per_second": round(window_tokens / max(window_elapsed, 1e-9), 4),
                    }
                    self._write_metric(record)
                    LOGGER.info(
                        "[train] stage=%s | epoch=%d/%d | epoch_progress=%s %5.1f%% | step=%d/%d | total_step=%d/%d | run_step=%d | global_step=%d | CE=%.5f | grad=%.4f | lr=%s | elapsed=%s | epoch_eta=%s | total_eta=%s | vram=%s | ex/s=%.2f | tok/s=%.0f",
                        _stage_label(stage),
                        global_epoch,
                        total_epochs,
                        _progress_bar(epoch_progress),
                        100.0 * epoch_progress,
                        epoch_step,
                        epoch_steps,
                        schedule_step,
                        total_training_steps,
                        run_step,
                        self.global_step,
                        float(step_loss),
                        float(grad),
                        learning_rates,
                        _format_duration(total_elapsed),
                        _format_duration(epoch_eta),
                        _format_duration(total_eta),
                        f"{peak_vram_gib:.2f}GiB" if peak_vram_gib is not None else "NA",
                        examples / max(window_elapsed, 1e-9),
                        window_tokens / max(window_elapsed, 1e-9),
                    )
        if self.distributed:
            totals = torch.tensor([float(ce_sum.detach()), float(token_total)], device=self.device, dtype=torch.float64)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            ce = float(totals[0].item()) / max(1.0, float(totals[1].item()))
        else:
            ce = float(ce_sum) / max(1, token_total)
        return {"loss": ce, "ce": ce}

    def fit(self, train_loader, validation_loader=None, resume_checkpoint: str | None = None) -> None:
        resume_info = None
        if resume_checkpoint:
            resume_info = load_checkpoint(resume_checkpoint, self.model, config=self.config)
            self.global_step = int(resume_info.get("step") or 0)
            self.best_metric = resume_info.get("best_metric")
            self._elapsed_before_fit = float(resume_info.get("elapsed_train_seconds") or 0.0)
        else:
            self._elapsed_before_fit = 0.0
        self._fit_start_global_step = self.global_step
        self._fit_started_at = time.monotonic()
        training = self.config["training"]
        resume_scheduler = bool(training.get("resume_scheduler", True))
        stages = (
            ("interface_warmup", int(training["interface_warmup_epochs"])),
            ("full_finetune", int(training["full_finetune_epochs"])),
        )
        steps_per_epoch = math.ceil(len(train_loader) / int(training["gradient_accumulation_steps"]))
        total_epochs = sum(epochs for _, epochs in stages)
        total_training_steps = max(1, steps_per_epoch * total_epochs)
        stage_order = {name: index for index, (name, _) in enumerate(stages)}
        if resume_info and resume_info.get("stage") not in stage_order:
            raise ValueError("Checkpoint has no recognized training stage")
        carried_state = {}
        for stage, epochs in stages:
            if epochs <= 0 or (resume_info and stage_order[stage] < stage_order[resume_info["stage"]]):
                continue
            set_stage_trainability(self.model, stage)
            optimizer = build_optimizer(self.model, self.config, stage)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter in carried_state:
                        optimizer.state[parameter] = carried_state[parameter]
            carried_state = {}
            total_steps = max(1, steps_per_epoch * epochs)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step, total=total_steps: max(0.0, 1.0 - step / total)
            )
            start_epoch = 1
            self.stage_optimizer_step = 0
            if resume_info and resume_info["stage"] == stage:
                load_checkpoint(
                    resume_checkpoint,
                    self.model,
                    optimizer,
                    self.config,
                    scheduler=self.scheduler if resume_scheduler else None,
                )
                start_epoch = int(resume_info.get("stage_epoch") or 0) + 1
                self.stage_optimizer_step = max(0, start_epoch - 1) * steps_per_epoch
                if not resume_scheduler:
                    self.scheduler.last_epoch = self.stage_optimizer_step
                    self.scheduler._step_count = self.stage_optimizer_step + 1
                    learning_rates = [
                        base_lr * lr_lambda(self.stage_optimizer_step)
                        for base_lr, lr_lambda in zip(self.scheduler.base_lrs, self.scheduler.lr_lambdas)
                    ]
                    for group, learning_rate in zip(optimizer.param_groups, learning_rates):
                        group["lr"] = learning_rate
                    self.scheduler._last_lr = list(learning_rates)
            for epoch in range(start_epoch, epochs + 1):
                self.epoch = epoch + (stages[0][1] if stage == "full_finetune" else 0)
                global_epoch = self.epoch
                if hasattr(train_loader.batch_sampler, "set_epoch"):
                    train_loader.batch_sampler.set_epoch(global_epoch)
                metrics = self._run_epoch(
                    train_loader,
                    optimizer,
                    stage,
                    True,
                    global_epoch,
                    total_epochs,
                    total_training_steps,
                )
                LOGGER.info(
                    "[train] epoch %d/%d complete | stage=%s | CE=%.5f | elapsed=%s",
                    global_epoch,
                    total_epochs,
                    _stage_label(stage),
                    metrics["ce"],
                    _format_duration(self._elapsed_train_seconds()),
                )
                self._write_metric(
                    {
                        "type": "epoch",
                        "split": "train",
                        "stage": stage,
                        "epoch": global_epoch,
                        "total_epochs": total_epochs,
                        "epoch_percent": 100.0,
                        "total_percent": round(100.0 * global_epoch / max(1, total_epochs), 3),
                        "ce": metrics["ce"],
                        "total_elapsed_seconds": round(self._elapsed_train_seconds(), 4),
                    }
                )
                validation = None
                if validation_loader is not None:
                    validation = self._run_epoch(
                        validation_loader,
                        optimizer,
                        stage,
                        False,
                        global_epoch,
                        total_epochs,
                        total_training_steps,
                    )
                    LOGGER.info(
                        "[validation] epoch %d/%d | CE=%.5f | elapsed=%s",
                        global_epoch,
                        total_epochs,
                        validation["ce"],
                        _format_duration(self._elapsed_train_seconds()),
                    )
                    self._write_metric(
                        {
                            "type": "epoch",
                            "split": "validation",
                            "stage": stage,
                            "epoch": global_epoch,
                            "total_epochs": total_epochs,
                            "total_elapsed_seconds": round(self._elapsed_train_seconds(), 4),
                            "ce": validation["ce"],
                        }
                    )
                output_dir = self.config["experiment"]["output_dir"]
                if (
                    self.is_main_process
                    and validation is not None
                    and bool(training.get("save_best", False))
                    and (self.best_metric is None or validation["loss"] < self.best_metric)
                ):
                    self.best_metric = float(validation["loss"])
                    save_checkpoint(
                        f"{output_dir}/best.pt",
                        self.model,
                        optimizer,
                        self.config,
                        epoch=global_epoch,
                        step=self.global_step,
                        best_metric=self.best_metric,
                        stage=stage,
                        stage_epoch=epoch,
                        elapsed_train_seconds=self._elapsed_train_seconds(),
                        scheduler=self.scheduler,
                    )
                if self.is_main_process and bool(training.get("save_each_epoch", True)):
                    save_checkpoint(
                        f"{output_dir}/epoch_{global_epoch:03d}.pt",
                        self.model,
                        optimizer,
                        self.config,
                        epoch=global_epoch,
                        step=self.global_step,
                        best_metric=self.best_metric,
                        stage=stage,
                        stage_epoch=epoch,
                        elapsed_train_seconds=self._elapsed_train_seconds(),
                        scheduler=self.scheduler,
                    )
                if self.is_main_process:
                    save_checkpoint(
                        f"{output_dir}/last.pt",
                        self.model,
                        optimizer,
                        self.config,
                        epoch=global_epoch,
                        step=self.global_step,
                        best_metric=self.best_metric,
                        stage=stage,
                        stage_epoch=epoch,
                        elapsed_train_seconds=self._elapsed_train_seconds(),
                        scheduler=self.scheduler,
                    )
                if self.distributed:
                    dist.barrier()
            carried_state = dict(optimizer.state)
        LOGGER.info(
            "[train] complete | epochs=%d/%d | optimizer_steps=%d/%d | global_step=%d | total_elapsed=%s",
            total_epochs,
            total_epochs,
            self.global_step - self._fit_start_global_step,
            total_training_steps,
            self.global_step,
            _format_duration(self._elapsed_train_seconds()),
        )
