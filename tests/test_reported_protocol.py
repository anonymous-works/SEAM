from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from decoder_only.benchmark import build_run_config
from seam.config import load_config
from seam.training import engine

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("pubmed", "arxiv", "booksum", "govreport")


@pytest.mark.parametrize("dataset", DATASETS)
def test_seam_and_t5_recipes_match_appendix(dataset):
    config = load_config(ROOT / f"src/seam/configs/seam_{dataset}.yaml")
    train = config["training"]
    assert train["batch_size"] * train["gradient_accumulation_steps"] == 96
    assert train["interface_warmup_epochs"] == 0
    assert train["save_best"] is True
    assert train["full_encoder_lr"] == train["full_decoder_lr"] == 3e-5
    assert train["full_bridge_lr"] == train["full_cross_attention_lr"] == 5e-5
    t5 = yaml.safe_load((ROOT / f"src/t5gemma2/configs/t5gemma2_{dataset}.yaml").read_text())
    train = t5["training"]
    assert train["per_device_train_batch_size"] * train["gradient_accumulation_steps"] == 96
    assert train["learning_rate"] == 3e-5
    assert train["eval_strategy"] == "epoch"


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model", ("qwen3_0_6b", "qwen3_1_7b", "qwen3_4b", "llama3_3b"))
def test_decoder_matrix_matches_appendix(dataset, model, monkeypatch):
    path = ROOT / "src/decoder_only/configs/decoder_only_benchmark.yaml"
    suite = yaml.safe_load(path.read_text())
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    monkeypatch.delenv(suite["models"][model]["path_env"], raising=False)
    config, _ = build_run_config(suite, path, model, dataset)
    train = config["training"]
    assert train["per_device_train_batch_size"] * train["gradient_accumulation_steps"] == 96
    assert train["learning_rate"] == 3e-5
    assert (
        config["data"]["max_source_length"]
        == {
            "pubmed": 4096,
            "arxiv": 8192,
            "booksum": 12288,
            "govreport": 12288,
        }[dataset]
    )
    if model == "llama3_3b":
        assert config["model"]["model_id"] == "meta-llama/Llama-3.2-3B"


def test_seam_saves_validation_best_not_last(tmp_path, monkeypatch):
    config = load_config(ROOT / "src/seam/configs/seam_pubmed.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)
    config["training"]["full_finetune_epochs"] = 3
    model = torch.nn.Linear(1, 1, bias=False)
    trainer = engine.SEAMTrainer(model, config, "cpu")
    monkeypatch.setattr(engine, "set_stage_trainability", lambda *args: None)
    monkeypatch.setattr(engine, "build_optimizer", lambda *args: torch.optim.SGD(model.parameters(), lr=0.01))

    def run_epoch(loader, optimizer, stage, train, global_epoch, *args):
        if train:
            with torch.no_grad():
                model.weight.fill_(global_epoch)
            trainer.global_step += 1
        loss = {1: 1.0, 2: 0.25, 3: 0.5}[global_epoch]
        return {"loss": loss, "ce": loss}

    monkeypatch.setattr(trainer, "_run_epoch", run_epoch)

    class Loader(list):
        batch_sampler = None

    trainer.fit(Loader([None]), Loader([None]))
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    assert best["epoch"] == 2
    assert best["best_metric"] == 0.25
    assert best["model"]["weight"].item() == 2
    assert last["epoch"] == 3
    assert last["model"]["weight"].item() == 3
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert manifest["epoch"] == 2
    assert manifest["validation_loss"] == 0.25
    with pytest.raises(ValueError, match="non-empty validation"):
        trainer.fit(Loader([None]), None)


@pytest.mark.parametrize("dataset", DATASETS)
def test_seam_prompt_serialization(dataset):
    from seam.data.collate import SummarizationCollator

    config = load_config(ROOT / f"src/seam/configs/seam_{dataset}.yaml")
    suffix = "Abstract:\n" if dataset in {"pubmed", "arxiv"} else "Summary:\n"
    assert config["data"]["decoder_prefix"] == suffix

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": config["data"]["decoder_prompt"].strip()}]
            assert kwargs == {
                "tokenize": True,
                "return_dict": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return [1, 2]

        def __call__(self, text, **kwargs):
            assert text == suffix
            assert kwargs == {"add_special_tokens": False}
            return {"input_ids": [3]}

    tokenizer = Tokenizer()
    collator = SummarizationCollator(tokenizer, tokenizer, config["data"])
    assert collator._prompt_ids_for() == [1, 2, 3]


def test_t5_evaluation_defaults_to_best_export(tmp_path):
    from t5gemma2.evaluate import resolve_checkpoint_source

    config = {"project": {"output_dir": str(tmp_path)}}
    (tmp_path / "final_model").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_source(config, None)
    best = tmp_path / "best_model"
    best.mkdir()
    assert resolve_checkpoint_source(config, None)[0] == str(best)


def _tiny_assets(tmp_path, seq2seq):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        T5Config,
        T5ForConditionalGeneration,
    )

    vocab = {token: i for i, token in enumerate(("[PAD]", "[EOS]", "[UNK]", "article", "summary", "fact"))}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    if seq2seq:
        model = T5ForConditionalGeneration(
            T5Config(
                vocab_size=len(vocab),
                d_model=8,
                d_ff=16,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=1,
                d_kv=8,
                decoder_start_token_id=0,
                pad_token_id=0,
                eos_token_id=1,
            )
        )
    else:
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocab),
                n_embd=8,
                n_layer=1,
                n_head=1,
                n_positions=64,
                pad_token_id=0,
                eos_token_id=1,
                bos_token_id=1,
            )
        )
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"id": "example", "text": "article fact", "summary": "summary fact"}) + "\n",
        )
    return model_dir


@pytest.mark.parametrize("seq2seq", (False, True), ids=("decoder_only", "t5gemma2"))
def test_hf_training_exports_best_epoch(tmp_path, monkeypatch, seq2seq):
    import transformers
    from decoder_only.train import train
    from t5gemma2 import train as t5_train

    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # Keep this offline smoke test on CPU even when macOS exposes MPS.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model_dir = _tiny_assets(tmp_path, seq2seq)
    train_cfg = {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-5,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "bf16": False,
        "tf32": False,
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
        "dataloader_num_workers": 0,
        "length_bucketing": False,
        "logging_steps": 1,
        "seed": 42,
    }
    data = {
        "train_file": str(tmp_path / "train.jsonl"),
        "validation_file": str(tmp_path / "validation.jsonl"),
        "test_file": str(tmp_path / "test.jsonl"),
        "source_field": "text",
        "target_field": "summary",
        "max_source_length": 16,
        "max_target_length": 8,
        "max_sequence_length": 32,
    }
    output_dir = tmp_path / "run"

    def override_loss(base):
        class ControlledLossTrainer(base):
            def evaluate(self, *args, **kwargs):
                metrics = super().evaluate(*args, **kwargs)
                # Deliberately make the first epoch better than the final epoch.
                metrics["eval_loss"] = 0.1 if self.state.epoch < 1.5 else 0.8
                self.log(metrics)
                return metrics

        return ControlledLossTrainer

    if seq2seq:
        monkeypatch.setattr(t5_train, "Seq2SeqTrainer", override_loss(transformers.Seq2SeqTrainer))
        config = {
            "project": {"output_dir": str(output_dir)},
            "model": {"model_name_or_path": str(model_dir), "torch_dtype": "float32"},
            "training": train_cfg,
            "data": data,
            "generation": {},
        }
    else:
        monkeypatch.setattr(transformers, "Trainer", override_loss(transformers.Trainer))
        config = {
            "run": {"name": "tiny_test", "output_dir": str(output_dir)},
            "model": {
                "name_or_path": str(model_dir),
                "local_files_only": True,
                "torch_dtype": "float32",
                "gradient_checkpointing": False,
            },
            "training": train_cfg,
            "data": data,
            "generation": {"batch_size": 1, "max_new_tokens": 8, "min_new_tokens": 0},
        }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    if seq2seq:
        monkeypatch.setattr("sys.argv", ["train", "--config", str(path)])
        t5_train.main()
        manifest = json.loads((output_dir / "best_model/checkpoint_manifest.json").read_text())
    else:
        assert train(path) == output_dir / "best_model"
        manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["best_validation_loss"] == 0.1
    assert Path(manifest["best_checkpoint"]).name == "checkpoint-1"
    cls = transformers.T5ForConditionalGeneration if seq2seq else transformers.GPT2LMHeadModel
    selected = cls.from_pretrained(output_dir / "best_model", local_files_only=True)
    first = cls.from_pretrained(Path(manifest["best_checkpoint"]), local_files_only=True)
    last = cls.from_pretrained(Path(manifest["best_checkpoint"]).parent / "checkpoint-2", local_files_only=True)
    assert all(torch.equal(selected.state_dict()[key], tensor) for key, tensor in first.state_dict().items())
    assert any(not torch.equal(selected.state_dict()[key], tensor) for key, tensor in last.state_dict().items())
