"""OCR for part numbers and markings.

Many McMaster items arrive in bags labelled with the part number, and some parts
carry stamped markings (bearing numbers, thread callouts). A readable part number
is a near-certain identification, so OCR runs first and can short-circuit search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image

# McMaster part numbers: 4-5 digits, a letter, 2-4 digits/letters, e.g. 91251A537, 9452K21, 3164T1
PART_NUMBER_RE = re.compile(r"\b(\d{4,5}[A-Z]\d{1,4}[A-Z]?)\b")


@dataclass
class OCRResult:
    texts: list[str] = field(default_factory=list)
    part_numbers: list[str] = field(default_factory=list)


def extract_part_numbers(texts: list[str]) -> list[str]:
    found: list[str] = []
    for t in texts:
        cleaned = t.upper().replace("O", "0").replace(" ", "")
        for m in PART_NUMBER_RE.finditer(cleaned):
            if m.group(1) not in found:
                found.append(m.group(1))
        for m in PART_NUMBER_RE.finditer(t.upper()):
            if m.group(1) not in found:
                found.append(m.group(1))
    return found


class OCREngine:
    """Thin wrapper around easyocr (optional). ``read`` returns [] when unavailable."""

    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = False):
        self._reader = None
        try:
            import easyocr  # type: ignore

            self._reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        except Exception:  # ImportError or model download failure
            self._reader = None

    @property
    def available(self) -> bool:
        return self._reader is not None

    def read(self, img: Image.Image) -> OCRResult:
        if self._reader is None:
            return OCRResult()
        import numpy as np

        texts = [t for _, t, conf in self._reader.readtext(np.asarray(img)) if conf > 0.3]
        return OCRResult(texts=texts, part_numbers=extract_part_numbers(texts))
