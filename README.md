# SEAM

This repository contains the SEAM long-document summarization implementation,
the LED, T5Gemma2, and decoder-only comparison pipelines, experiment
configurations, and ROUGE/BERTScore evaluation scripts.

## Environment

Use Python 3.10 or later. Install the package and baseline dependencies with:

```bash
python -m pip install -e ".[baselines,evaluation]"
```

Training scripts use locally available model checkpoints. Set the model-path
environment variables named in the relevant YAML configuration before running
them. The decoder-only runner uses `CUDA_VISIBLE_DEVICES` to select visible
GPUs; the reported configurations use one GPU.

## Data

Large datasets and model weights are not stored in this repository. Prepare
train, validation, and test JSONL files for each dataset and place them under
`src/seam/datasets/<dataset>/` (or update the corresponding `data_root`). Each
record must provide source text and a reference summary; the default field
names are `text` and `summary`. An optional `id` field is supported. The
canonical preparation entry point is:

```bash
bash src/seam/scripts/prepare_seam.sh --help
```

## Running experiments

The scripts under `src/scripts/` launch the main comparison, component
ablations, encoder replacements, and evaluation. For example:

```bash
bash src/scripts/run_rq1.sh seam pubmed
bash src/scripts/run_rq2.sh pubmed full
bash src/scripts/run_rq3.sh pubmed pplx
```

The decoder-only model matrix and per-dataset settings are in
`src/decoder_only/configs/decoder_only_benchmark.yaml`. With one GPU, the
effective training batch is the per-device batch multiplied by gradient
accumulation steps. The dataset-specific SEAM, T5Gemma2, and decoder-only
configurations use an effective batch of 96 on one GPU. Their backbone learning
rate is `3e-5`; SEAM uses `5e-5` for its added interface. SEAM starts full
fine-tuning immediately, without an interface-only warmup stage. Evaluation model
paths and metric checkpoints are supplied through the environment variables
checked by the scripts. Set `PYROUGE_HOME_DIR` to a local ROUGE-1.5.5
installation and `BERTSCORE_MODEL_PATH` to the local encoder checkpoint used
for BERTScore before running the evaluator.

The LED-large-16k baseline is configured in
`src/led/configs/led_benchmark.yaml`. Set `LED_MODEL_PATH` to a local
`allenai/led-large-16384` checkpoint and provide the run's actual epoch count;
the script selects the checkpoint with the lowest epoch-end validation loss.
It assigns global attention to the first non-padding source token, as required
by LED. Predictions use the same JSONL format as other systems. Evaluate them
with `src/scripts/evaluate_all.sh` to run ROUGE and BERTScore. The epoch count
is required at launch rather than guessed, so supply the training horizon used
for the reported run.

Use the configuration files as the source of truth for model IDs, input and
output limits, prompts, optimizer settings, and decoding parameters. Run
outputs and downloaded or prepared assets are intentionally kept outside the
tracked source files.

## Checkpoint selection and prompts

Validation runs at the end of every epoch. The lowest validation-loss checkpoint
is used for evaluation: `best.pt` for SEAM and `best_model/` for T5Gemma2 and
decoder-only baselines. SEAM writes `checkpoint_manifest.json`, T5Gemma2 writes
`best_model/checkpoint_manifest.json`, and decoder-only writes `run_manifest.json`
with the selected checkpoint, epoch or step, and validation loss. Resolved
configurations are saved alongside the run artifacts. Epoch counts remain
configurable; the released recipes and smoke tests are not archival proof of
the settings or selected checkpoints used for earlier reported scores.

SEAM serializes its decoder instruction as a user message with the native chat
template, `add_generation_prompt=True`, and `enable_thinking=False`. It then
appends `Abstract:\n` for PubMed/arXiv or `Summary:\n` for BookSum/GovReport,
without adding more special tokens. The controller uses these prompt tokens,
not reference-summary tokens. The decoder-only comparison uses the source-prefix
instructions defined in its matrix configuration instead of this SEAM-only
prompt. Its Llama baseline is `meta-llama/Llama-3.2-3B` (base), whereas the encoder
replacement uses `meta-llama/Llama-3.2-1B-Instruct`.

## Protocol checks

Install `pytest` and run the offline regression checks with:

```bash
PYTHONPATH=src/seam:src/t5gemma2:src/decoder_only:src/led python -m pytest tests src/led/tests
```

These checks verify configuration budgets, prompt serialization, and validation-best
export, including tiny local-model CPU training tests. They do not reproduce
the reported B200 experiments or establish empirical model quality.
