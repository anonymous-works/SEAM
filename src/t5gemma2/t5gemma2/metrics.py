from __future__ import annotations

import unicodedata
from typing import Dict, Sequence


def heter_sum_graph_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    include_rouge_lsum: bool = False,
    digits: int = 4,
) -> Dict[str, float]:
    from rouge import Rouge

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        result = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        if include_rouge_lsum:
            result["rougeLsum"] = 0.0
        return result

    def normalize(text: str) -> str:
        return unicodedata.normalize("NFC", str(text)).lower().strip()

    hypotheses = [normalize(text) or "<empty>" for text in predictions]
    targets = [normalize(text) or "<empty-reference>" for text in references]
    scores = Rouge().get_scores(hypotheses, targets, avg=True)
    result = {
        "rouge1": round(100.0 * float(scores["rouge-1"]["f"]), digits),
        "rouge2": round(100.0 * float(scores["rouge-2"]["f"]), digits),
        "rougeL": round(100.0 * float(scores["rouge-l"]["f"]), digits),
    }
    if include_rouge_lsum:
        result["rougeLsum"] = result["rougeL"]
    return result
