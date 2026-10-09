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
accumulation steps. The configured decoder-only combinations use an effective
batch of 96, matching the paper's implementation details. Evaluation model
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
