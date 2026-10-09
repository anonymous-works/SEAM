from __future__ import annotations

import unicodedata
from typing import Sequence


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    from rouge import Rouge

    def normalize(text: str) -> str:
        return unicodedata.normalize("NFC", str(text)).lower().strip() or "<empty>"

    scores = Rouge().get_scores(
        [normalize(value) for value in predictions], [normalize(value) for value in references], avg=True
    )
    return {
        "rouge1": round(100.0 * float(scores["rouge-1"]["f"]), 4),
        "rouge2": round(100.0 * float(scores["rouge-2"]["f"]), 4),
        "rougeL": round(100.0 * float(scores["rouge-l"]["f"]), 4),
    }
