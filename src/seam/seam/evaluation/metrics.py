from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any, Sequence

from rouge import Rouge


def _clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).lower().split())


def summarization_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    pairs = [(_clean(pred), _clean(ref)) for pred, ref in zip(predictions, references)]
    count = max(1, len(pairs))
    scores = dict(rouge1=0.0, rouge2=0.0, rougeL=0.0)
    valid = [(pred, ref) for pred, ref in pairs if pred.replace(".", "").strip() and ref.replace(".", "").strip()]
    if valid:
        scorer = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
        raw = scorer.get_scores([pred for pred, _ in valid], [ref for _, ref in valid])
        for output, key in (("rouge1", "rouge-1"), ("rouge2", "rouge-2"), ("rougeL", "rouge-l")):
            scores[output] = 100.0 * sum(row[key]["f"] for row in raw) / count
    lengths = [(len(pred.split()), len(ref.split())) for pred, ref in pairs]
    repetition = []
    for pred, _ in pairs:
        words = pred.split()
        trigrams = Counter(zip(words, words[1:], words[2:]))
        repetition.append(sum(max(0, frequency - 1) for frequency in trigrams.values()) / max(1, len(words) - 2))
    return {
        **scores,
        "num_examples": len(pairs),
        "rouge_backend": "rouge==1.0.0 (Python diagnostic; not ROUGE-1.5.5)",
        "prediction_words_mean": sum(p for p, _ in lengths) / count,
        "reference_words_mean": sum(r for _, r in lengths) / count,
        "length_ratio_mean": sum(p / max(1, r) for p, r in lengths) / count,
        "empty_prediction_rate": 100 * sum(p == 0 for p, _ in lengths) / count,
        "too_short_rate": 100 * sum(p < 0.5 * r for p, r in lengths) / count,
        "too_long_rate": 100 * sum(p > 1.5 * r for p, r in lengths) / count,
        "repeated_trigram_rate_mean": sum(repetition) / count,
    }
