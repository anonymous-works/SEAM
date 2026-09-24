from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_MODEL_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_GPU_SPEC = re.compile(r"[0-9]+(?:,[0-9]+)*")


def _merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _selected(value: str | None, env_name: str, default: list[str], available: dict[str, Any]) -> list[str]:
    raw = value or os.environ.get(env_name, "")
    names = [part.strip() for part in raw.split(",") if part.strip()] if raw else list(default)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown {env_name.lower()} entry(s): {unknown}; available={sorted(available)}")
    if not names:
        raise ValueError(f"No entries selected for {env_name.lower()}")
    return names


def _parse_gpu_ids(value: str) -> list[str]:

    spec = str(value).strip()
    if not spec or _GPU_SPEC.fullmatch(spec) is None:
        raise ValueError("GPU_ID must be a comma-separated list of GPU indices, for example 0,1")
    ids = spec.split(",")
    if len(set(ids)) != len(ids):
        raise ValueError(f"GPU_ID contains duplicate devices: {spec}")
    return ids


def _distributed_train_command(
    base_command: list[str],
    *,
    world_size: int,
    python_executable: str,
    torchrun_path: str | None = None,
) -> list[str]:

    if world_size <= 1:
        return base_command
    if torchrun_path is None:
        candidate = Path(python_executable).with_name("torchrun")
        torchrun_path = str(candidate) if candidate.is_file() else None
    launcher = [torchrun_path] if torchrun_path else [python_executable, "-m", "torch.distributed.run"]
    return [
        *launcher,
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        *base_command[1:],
    ]


def _resolve_model(model_name: str, spec: dict[str, Any]) -> str:
    env_name = str(spec.get("path_env", "")).strip()
    if env_name:
        configured = os.environ.get(env_name, "").strip()
        if configured:
            return str(Path(configured).expanduser().resolve())
    model_root = os.environ.get("MODEL_ROOT", "").strip()
    local_dir = str(spec.get("local_dir", "")).strip()
    if model_root and local_dir:
        return str((Path(model_root).expanduser() / local_dir).resolve())
    if local_dir:
        return str(Path(local_dir).expanduser().resolve())
    raise ValueError(
        f"No local path configured for {model_name}; set {env_name or 'MODEL_ROOT'} or provide a model.local_dir"
    )


def _resolve_generation(
    defaults: dict[str, Any], model_spec: dict[str, Any], dataset_spec: dict[str, Any], model_name: str
) -> dict[str, Any]:

    generation = _merge(
        defaults.get("generation", {}),
        model_spec.get("generation", {}),
        dataset_spec.get("generation", {}),
    )
    batch_matrix = generation.pop("batch_size_by_model", None)
    if batch_matrix is None:
        return generation
    if not isinstance(batch_matrix, dict):
        raise ValueError(f"datasets.*.generation.batch_size_by_model must be a mapping, got {batch_matrix!r}")
    selected = batch_matrix.get(model_name, batch_matrix.get("default"))
    if selected is None:
        raise ValueError(
            f"No generation batch size for model {model_name!r}; add it to the dataset matrix or provide default"
        )
    generation["batch_size"] = selected
    return generation


def build_run_config(
    suite: dict[str, Any],
    suite_path: Path,
    model_name: str,
    dataset_name: str,
    *,
    max_train_examples: int = 0,
    max_validation_examples: int = 0,
    max_test_examples: int = 0,
) -> tuple[dict[str, Any], Path]:
    models = suite["models"]
    datasets = suite["datasets"]
    model_spec = models[model_name]
    dataset_spec = datasets[dataset_name]
    defaults = suite.get("defaults", {})
    output_root = _resolve(suite.get("output_root", "../../../runs/decoder_only"), suite_path.parent)
    data_root = _resolve(suite.get("data_root", "../../seam/datasets"), suite_path.parent)
    data_dir = _resolve(dataset_spec.get("data_dir", dataset_name), data_root)
    run_name = f"{model_name}__{dataset_name}"
    model_config = _merge(defaults.get("model", {}), model_spec.get("model", {}))
    model_config.update(
        {
            "model_id": str(model_spec.get("model_id", model_config.get("model_id", model_name))),
            "name_or_path": _resolve_model(model_name, model_spec),
            "family": str(model_spec.get("family", model_config.get("family", "causal_lm"))),
            "local_files_only": True,
        }
    )
    if "diffusion_paradigm" in model_spec:
        model_config["diffusion_paradigm"] = model_spec["diffusion_paradigm"]
    data_config = _merge(defaults.get("data", {}), dataset_spec)
    for key in ("data_dir", "training", "generation"):
        data_config.pop(key, None)
    data_config["dataset"] = dataset_name
    data_config.update(
        {
            "train_file": str(data_dir / str(dataset_spec.get("train_file", "train.jsonl"))),
            "validation_file": str(data_dir / str(dataset_spec.get("validation_file", "validation.jsonl"))),
            "test_file": str(data_dir / str(dataset_spec.get("test_file", "test.jsonl"))),
        }
    )
    training = _merge(defaults.get("training", {}), model_spec.get("training", {}), dataset_spec.get("training", {}))
    limits = {
        "max_train_examples": int(max_train_examples),
        "max_validation_examples": int(max_validation_examples),
        "max_test_examples": int(max_test_examples),
    }
    config: dict[str, Any] = {
        "run": {"name": run_name, "output_dir": str(output_root / run_name)},
        "model": model_config,
        "data": data_config,
        "training": training,
        "generation": _resolve_generation(defaults, model_spec, dataset_spec, model_name),
        "limits": limits,
    }
    safe_name = _MODEL_NAME.sub("_", run_name)
    config_path = output_root / ".configs" / f"{safe_name}.yaml"
    return config, config_path


def _write_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _data_preflight(config: dict[str, Any]) -> None:
    missing = [
        config["data"][key]
        for key in ("train_file", "validation_file", "test_file")
        if not Path(config["data"][key]).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing prepared dataset files: " + ", ".join(missing))


def _model_preflight(config: dict[str, Any]) -> None:
    model_path = Path(config["model"]["name_or_path"]).expanduser()
    if not model_path.is_dir():
        model_id = config["model"].get("model_id", "unknown")
        raise FileNotFoundError(
            "Local model directory does not exist: "
            f"{model_path}. Set the corresponding *_PATH variable (model_id={model_id}); "
            "Hugging Face loading/downloads are disabled."
        )


def run_suite(args: argparse.Namespace) -> int:
    suite_path = Path(args.config).expanduser().resolve()
    with suite_path.open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle) or {}
    if (
        not isinstance(suite, dict)
        or not isinstance(suite.get("models"), dict)
        or not isinstance(suite.get("datasets"), dict)
    ):
        raise ValueError("Suite config requires mapping sections: models and datasets")
    model_names = _selected(
        args.models, "DECODER_MODELS", suite.get("model_order", list(suite["models"])), suite["models"]
    )
    dataset_names = _selected(
        args.datasets, "DECODER_DATASETS", suite.get("dataset_order", list(suite["datasets"])), suite["datasets"]
    )
    gpu = os.environ.get("GPU_ID", str(suite.get("gpu", "0"))).strip()
    gpu_ids = _parse_gpu_ids(gpu)
    world_size = len(gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    output_root = _resolve(suite.get("output_root", "../../../runs/decoder_only"), suite_path.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "suite_status.jsonl"
    entries = [(model_name, dataset_name) for model_name in model_names for dataset_name in dataset_names]
    print(
        f"Sequential suite: GPU_ID={','.join(gpu_ids)}; DDP training world_size={world_size}; "
        f"evaluation_backend={args.eval_backend}; runs={len(entries)}"
    )
    for index, (model_name, dataset_name) in enumerate(entries, start=1):
        config, config_path = build_run_config(
            suite,
            suite_path,
            model_name,
            dataset_name,
            max_train_examples=args.max_train_examples,
            max_validation_examples=args.max_validation_examples,
            max_test_examples=args.max_test_examples,
        )
        _write_config(config, config_path)
        if not args.dry_run:
            _model_preflight(config)
            _data_preflight(config)
        run_dir = Path(config["run"]["output_dir"])
        print(f"[{index}/{len(entries)}] {model_name} on {dataset_name} -> {run_dir}")
        started = time.time()
        status = "planned"
        error = ""
        train_command = _distributed_train_command(
            [sys.executable, "-m", "decoder_only.train", "--config", str(config_path)],
            world_size=world_size,
            python_executable=sys.executable,
        )
        if args.overwrite_output_dir:
            train_command.append("--overwrite-output-dir")
        if args.dry_run:
            print(f"  train: {shlex.join(train_command)}")
        if not args.dry_run:
            command_env = os.environ.copy()
            try:
                subprocess.run(train_command, cwd=str(suite_path.parents[3]), env=command_env, check=True)
                if not args.skip_eval:
                    prediction_path = run_dir / f"{args.split}_predictions.jsonl"
                    evaluate_command = [
                        sys.executable,
                        "-m",
                        "decoder_only.evaluate",
                        "--config",
                        str(config_path),
                        "--checkpoint",
                        str(run_dir / "final_model"),
                        "--output",
                        str(prediction_path),
                        "--split",
                        args.split,
                        "--backend",
                        args.eval_backend,
                        "--vllm-base-url",
                        args.vllm_base_url,
                        "--vllm-startup-timeout",
                        str(args.vllm_startup_timeout),
                        "--progress-every",
                        str(args.progress_every),
                    ]
                    if args.vllm_model:
                        evaluate_command.extend(["--vllm-model", args.vllm_model])
                    if args.vllm_batch_size > 0:
                        evaluate_command.extend(["--vllm-batch-size", str(args.vllm_batch_size)])
                    evaluate_command.append(
                        "--start-vllm-service" if args.start_vllm_service else "--no-start-vllm-service"
                    )
                    if args.max_eval_examples > 0:
                        evaluate_command.extend(["--max-examples", str(args.max_eval_examples)])

                    eval_env = command_env.copy()
                    eval_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
                    eval_env["GPU_ID"] = gpu_ids[0]
                    subprocess.run(evaluate_command, cwd=str(suite_path.parents[3]), env=eval_env, check=True)
                status = "complete"
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error = f"exit_status={exc.returncode}"
                if not args.continue_on_error:
                    with status_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps({"model": model_name, "dataset": dataset_name, "status": status, "error": error})
                            + "\n"
                        )
                    raise
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "gpu_ids": gpu_ids,
                        "world_size": world_size,
                        "run_dir": str(run_dir),
                        "config": str(config_path),
                        "status": status,
                        "error": error,
                        "elapsed_seconds": round(time.time() - started, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Suite finished; status log: {status_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential decoder-baseline matrix runner with optional DDP training")
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "decoder_only_benchmark.yaml")
    )
    parser.add_argument(
        "--models", default=None, help="Comma-separated model keys; defaults to DECODER_MODELS or suite order"
    )
    parser.add_argument(
        "--datasets", default=None, help="Comma-separated dataset keys; defaults to DECODER_DATASETS or suite order"
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument(
        "--eval-backend",
        "--backend",
        dest="eval_backend",
        choices=("auto", "vllm", "local"),
        default=os.environ.get("DECODER_EVAL_BACKEND", "auto"),
        help="auto uses vLLM for standard decoder-only LMs and local native AR generation for Nemotron; use local to force in-process Transformers",
    )
    parser.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--vllm-batch-size", type=int, default=int(os.environ.get("VLLM_BATCH_SIZE", "0")))
    parser.add_argument("--vllm-startup-timeout", type=float, default=900.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--start-vllm-service",
        dest="start_vllm_service",
        action="store_true",
        help="Start vllm serve from each evaluation checkpoint",
    )
    parser.add_argument(
        "--no-start-vllm-service",
        dest="start_vllm_service",
        action="store_false",
        help="Use an already running VLLM_BASE_URL service",
    )
    parser.set_defaults(start_vllm_service=not bool(os.environ.get("VLLM_BASE_URL")))
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_suite(args))


if __name__ == "__main__":
    main()
