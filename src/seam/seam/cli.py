from __future__ import annotations

import argparse
import logging

from .config import load_config
from .data.prepare import prepare_split
from .data.prepare_dataset import SUPPORTED_DATASETS


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(prog="seam")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("config")
    train.add_argument("--device", default=None)
    train.add_argument("--resume-checkpoint", default=None)
    train.add_argument("--train-file", default=None, help="Override data.train_file with one JSONL file")
    train.add_argument(
        "--train-only",
        action="store_true",
        help="Load only the training split and skip validation; evaluation is a separate command",
    )
    train.add_argument("--max-train-examples", type=int, default=0)
    train.add_argument("--max-validation-examples", type=int, default=0)
    train.add_argument("--overwrite-output-dir", action="store_true")
    train.add_argument("--output-dir", default=None)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("config")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("output")
    evaluate.add_argument("--split", default="test", choices=("train", "validation", "test"))
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.add_argument("--max-examples", type=int, default=0)
    evaluate.add_argument("--do-sample", action="store_true")
    evaluate.add_argument("--temperature", type=float, default=None)
    evaluate.add_argument("--top-k", type=int, default=None)
    evaluate.add_argument("--top-p", type=float, default=None)
    evaluate.add_argument("--shard-rank", type=int, default=0)
    evaluate.add_argument("--num-shards", type=int, default=1)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("input_path")
    prepare.add_argument("output_path")
    prepare.add_argument("--source-field", default="text")
    prepare.add_argument("--target-field", default="summary")
    prepare.add_argument("--id-field", default="id")
    prepare.add_argument(
        "--detokenize",
        action="store_true",
        help="Normalize whitespace and punctuation in source and target before writing.",
    )
    prepare_dataset = subparsers.add_parser("prepare-dataset")
    prepare_dataset.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    prepare_dataset.add_argument("--input-dir", required=True)
    prepare_dataset.add_argument("--output-dir", required=True)
    prepare_dataset.add_argument("--raw-copy-dir", default=None)
    prepare_dataset.add_argument("--source-field", default=None)
    prepare_dataset.add_argument("--target-field", default=None)
    prepare_dataset.add_argument("--id-field", default=None)
    prepare_dataset.add_argument("--list-separator", default="\n")
    prepare_dataset.add_argument("--detokenize", action=argparse.BooleanOptionalAction, default=None)
    prepare_dataset.add_argument("--allow-duplicate-ids", action="store_true")
    prepare_dataset.add_argument("--allow-cross-split-content", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        print(f"valid SEAM config: {config['_meta']['config_path']}")
        return
    if args.command == "train":
        from .runtime import train as run_train

        run_train(
            args.config,
            device=args.device,
            resume_checkpoint=args.resume_checkpoint,
            train_file=args.train_file,
            train_only=args.train_only,
            max_train_examples=args.max_train_examples,
            max_validation_examples=args.max_validation_examples,
            overwrite_output_dir=args.overwrite_output_dir,
            output_dir_override=args.output_dir,
        )
        return
    if args.command == "evaluate":
        from .runtime import evaluate as run_evaluate

        result = run_evaluate(
            args.config,
            args.checkpoint,
            args.output,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            max_examples=args.max_examples,
            do_sample=True if args.do_sample else None,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            shard_rank=args.shard_rank,
            num_shards=args.num_shards,
        )
        print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "prepare-dataset":
        from .data.prepare_dataset import prepare_dataset as run_prepare_dataset

        report = run_prepare_dataset(
            args.input_dir,
            args.output_dir,
            dataset=args.dataset,
            raw_copy_dir=args.raw_copy_dir,
            allow_cross_split_content=args.allow_cross_split_content,
            source_field=args.source_field,
            target_field=args.target_field,
            id_field=args.id_field,
            list_separator=args.list_separator,
            detokenize_text=args.detokenize,
            allow_duplicate_ids=args.allow_duplicate_ids,
        )
        for split, stats in report["splits"].items():
            print(f"{split}: {stats['kept']} examples (skipped {stats['skipped']}) -> {stats['processed_path']}")
        return
    count = prepare_split(
        args.input_path,
        args.output_path,
        source_field=args.source_field,
        target_field=args.target_field,
        id_field=args.id_field,
        detokenize_text=args.detokenize,
    )
    print(f"prepared {count} records -> {args.output_path}")


if __name__ == "__main__":
    main()
