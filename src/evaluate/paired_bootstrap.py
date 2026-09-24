from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate.evaluate_rouge import _canonical

_DETAIL_SCHEMA = "eviseq.perl_rouge155_details.v1"
_OUTPUT_SCHEMA = "eviseq.perl_rouge155_paired_bootstrap.v1"
_METRICS = ("rouge1", "rouge2", "rougeL")


def _load_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _prediction_contracts(
    predictions_file: Path,
    prediction_field: str,
    reference_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with predictions_file.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {predictions_file}:{line_number}") from exc
            _require(isinstance(value, dict), f"Non-object JSONL row at {predictions_file}:{line_number}")
            _require(prediction_field in value, f"Missing {prediction_field!r} at row {line_number}")
            _require(reference_field in value, f"Missing {reference_field!r} at row {line_number}")
            row_index = len(rows)
            identifier = _canonical(value.get("id", row_index))
            reference = _canonical(value[reference_field])
            rows.append(
                {
                    "row_index": row_index,
                    "id": identifier,
                    "reference": reference,
                }
            )
    _require(bool(rows), f"No prediction rows found: {predictions_file}")
    identifiers = [json.dumps(row["id"], ensure_ascii=False, sort_keys=True) for row in rows]
    _require(len(set(identifiers)) == len(identifiers), f"Duplicate example IDs in {predictions_file}")
    return rows


def _verified_run(headline_path: Path) -> dict[str, Any]:
    headline_path = headline_path.expanduser().resolve()
    headline = _load_json(headline_path)
    detail_path = Path(str(headline.get("per_example_scores_file", ""))).expanduser().resolve()
    _require(detail_path.is_file(), f"Missing per-example artifact referenced by {headline_path}")
    details = _load_json(detail_path)
    _require(details.get("schema_version") == _DETAIL_SCHEMA, f"Unsupported details schema: {detail_path}")
    _require("Perl ROUGE-1.5.5" in str(headline.get("backend", "")), "Headline is not official Perl ROUGE")
    _require(details.get("backend") == headline.get("backend"), "Headline/details backend mismatch")
    _require(
        details.get("headline_protocol") == headline.get("pyrouge_default_args"),
        "Headline/details ROUGE protocol mismatch",
    )
    count = int(headline.get("num_examples", -1))
    _require(count > 0 and int(details.get("num_examples", -2)) == count, "Invalid/mismatched example count")

    predictions_file = Path(str(headline.get("predictions_file", ""))).expanduser().resolve()
    _require(predictions_file.is_file(), f"Predictions file no longer exists: {predictions_file}")
    raw_detail_file = Path(str(details.get("raw_detail_file", ""))).expanduser().resolve()
    _require(raw_detail_file.is_file(), f"Missing raw Perl detail output: {raw_detail_file}")

    prediction_field = str(details.get("prediction_field", "prediction"))
    reference_field = str(details.get("reference_field", "reference"))
    contracts = _prediction_contracts(predictions_file, prediction_field, reference_field)
    _require(len(contracts) == count, "Prediction/detail row-count mismatch")

    rows = details.get("rows")
    _require(isinstance(rows, list) and len(rows) == count, "Malformed detailed ROUGE rows")
    for index, (row, contract) in enumerate(zip(rows, contracts, strict=True)):
        _require(isinstance(row, dict), f"Malformed detailed row {index}")
        _require(row.get("row_index") == index, f"Detailed row index mismatch at {index}")
        _require(row.get("eval_task_id") == index + 1, f"Detailed task ID mismatch at {index}")
        _require(row.get("id") == contract["id"], f"Detailed example ID mismatch at {index}")
        if "reference" in row:
            _require(row.get("reference") == contract["reference"], f"Detailed reference mismatch at {index}")
        for metric in _METRICS:
            values = row.get(metric)
            _require(isinstance(values, dict), f"Missing {metric} details at row {index}")
            for field in ("recall", "precision", "f1"):
                value = float(values.get(field, math.nan))
                _require(math.isfinite(value) and 0.0 <= value <= 1.0, f"Invalid {metric}.{field} at row {index}")

    for metric in _METRICS:
        mean = float(np.mean([float(row[metric]["f1"]) for row in rows]))
        stored_mean = float(details.get("per_example_f1_mean", {}).get(metric, math.nan))
        _require(
            math.isfinite(stored_mean) and abs(mean - stored_mean) <= 1.0e-12,
            f"Stored {metric} per-example mean is invalid",
        )
    return {
        "headline_path": headline_path,
        "headline": headline,
        "detail_path": detail_path,
        "details": details,
        "contracts": contracts,
        "rows": rows,
        "id_reference": [(row["id"], row["reference"]) for row in contracts],
    }


def _bootstrap_interval(
    differences: np.ndarray,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float, float, float]:
    _require(differences.ndim == 1 and differences.size > 0, "Paired differences must be non-empty")
    _require(samples >= 100, "At least 100 bootstrap samples are required")
    _require(0.0 < confidence < 1.0, "Confidence must lie in (0, 1)")
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk_size = max(1, min(512, 2_000_000 // differences.size))
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = generator.integers(0, differences.size, size=(stop - start, differences.size))
        estimates[start:stop] = differences[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    p_nonpositive = (float(np.count_nonzero(estimates <= 0.0)) + 1.0) / (samples + 1.0)
    win_probability = float(np.mean(estimates > 0.0))
    return float(low), float(high), p_nonpositive, win_probability


def compare(
    candidate_headline: Path,
    baseline_headline: Path,
    output_file: Path,
    *,
    samples: int = 10_000,
    seed: int = 1729,
    confidence: float = 0.95,
) -> dict[str, Any]:
    output_file = output_file.expanduser().resolve()
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite paired-bootstrap artifact: {output_file}")
    candidate = _verified_run(candidate_headline)
    baseline = _verified_run(baseline_headline)
    _require(
        candidate["id_reference"] == baseline["id_reference"],
        "Candidate and baseline are not aligned to the same ordered IDs/references",
    )
    for key in (
        "headline_protocol",
        "detail_protocol",
        "backend",
        "prediction_field",
        "reference_field",
    ):
        _require(
            candidate["details"].get(key) == baseline["details"].get(key),
            f"Candidate/baseline details mismatch: {key}",
        )

    scores: dict[str, Any] = {}
    for metric in _METRICS:
        candidate_values = np.asarray([row[metric]["f1"] for row in candidate["rows"]], dtype=np.float64)
        baseline_values = np.asarray([row[metric]["f1"] for row in baseline["rows"]], dtype=np.float64)
        differences = (candidate_values - baseline_values) * 100.0
        low, high, p_nonpositive, win_probability = _bootstrap_interval(
            differences,
            samples,
            seed,
            confidence,
        )
        candidate_score = float(candidate["headline"][metric])
        baseline_score = float(baseline["headline"][metric])
        headline_delta = candidate_score - baseline_score
        paired_mean_delta = float(differences.mean())
        scores[metric] = {
            "candidate": candidate_score,
            "baseline": baseline_score,
            "official_headline_delta": headline_delta,
            "paired_mean_delta": paired_mean_delta,
            "ci95_low": low,
            "ci95_high": high,
            "p_nonpositive": p_nonpositive,
            "win_probability": win_probability,
        }

    result = {
        "schema_version": _OUTPUT_SCHEMA,
        "comparison_valid": True,
        "num_examples": len(candidate["rows"]),
        "pairing": {"verified": True, "key": "ordered_id_reference"},
        "scorer": {
            "backend": candidate["details"]["backend"],
            "headline_protocol": candidate["details"]["headline_protocol"],
            "detail_protocol": candidate["details"]["detail_protocol"],
        },
        "bootstrap": {
            "method": "paired_example_percentile",
            "samples": samples,
            "seed": seed,
            "confidence": confidence,
            "delta": "candidate_minus_baseline",
            "scale": "ROUGE percentage points",
        },
        "scores": scores,
        "decision": {
            "primary_metric": "rouge2",
            "rouge2_ci95_low_gt_zero": scores["rouge2"]["ci95_low"] > 0.0,
            "all_ci95_low_gt_zero": all(scores[metric]["ci95_low"] > 0.0 for metric in _METRICS),
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_headline", type=Path)
    parser.add_argument("baseline_headline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    compare(
        args.candidate_headline,
        args.baseline_headline,
        args.output,
        samples=args.samples,
        seed=args.seed,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
