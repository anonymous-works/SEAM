from __future__ import annotations

import re
import unicodedata


def detokenize(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = unicodedata.normalize("NFKC", line).replace("``", '"').replace("''", '"')
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?%])", r"\1", line)
        line = re.sub(r"([\(\[\{])\s+", r"\1", line)
        line = re.sub(r"\s+([\)\]\}])", r"\1", line)
        line = re.sub(r"\s+n['’]t\b", "n't", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+(['’](?:s|re|ve|ll|d|m))\b", r"\1", line, flags=re.IGNORECASE)
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)
