"""Feedback store: confirmed identifications become labelled real photos.

Every confirmation writes the query photo to ``<queries_dir>/<part_number>/`` (the
layout ``mcv evaluate --query-dir`` and ``mcv train --extra-images`` consume) and
appends a JSON line to ``feedback.jsonl``. "None of these" answers are kept under
``_unknown/`` so hard cases can be reviewed and labelled later.
"""

from __future__ import annotations

import re
from pathlib import Path

from mcmaster_vision.schemas import Feedback

UNKNOWN_DIR = "_unknown"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def safe_segment(value: str, what: str) -> str:
    """Only plain identifiers may become path segments (no separators, no dot-dot)."""
    v = str(value).strip()
    if not _SAFE.match(v) or v in (".", ".."):
        raise ValueError(f"invalid {what}: {value!r}")
    return v


class FeedbackStore:
    def __init__(self, queries_dir: str | Path):
        self.root = Path(queries_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "feedback.jsonl"

    def record(
        self,
        image_bytes: bytes,
        request_id: str,
        part_number: str | None,
        *,
        predicted: str | None = None,
        tier: str | None = None,
        ext: str = "jpg",
    ) -> Feedback:
        request_id = safe_segment(request_id, "request_id")
        pn = safe_segment(part_number, "part_number").upper() if part_number else None
        ext = safe_segment(ext, "extension").lower()
        folder = self.root / (pn or UNKNOWN_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{request_id}.{ext}"
        path.write_bytes(image_bytes)
        fb = Feedback(
            request_id=request_id,
            part_number=part_number.upper() if part_number else None,
            predicted=predicted,
            tier=tier,
            image_path=str(path.resolve()),
        )
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(fb.model_dump_json() + "\n")
        return fb

    def entries(self) -> list[Feedback]:
        if not self.log.exists():
            return []
        return [
            Feedback.model_validate_json(ln)
            for ln in self.log.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    def stats(self) -> dict[str, int]:
        e = self.entries()
        confirmed = [x for x in e if x.part_number]
        return {
            "total": len(e),
            "confirmed": len(confirmed),
            "unknown": len(e) - len(confirmed),
            "correct_top1": sum(1 for x in confirmed if x.predicted == x.part_number),
            "parts_with_photos": len({x.part_number for x in confirmed}),
        }

    def labelled_images(self) -> dict[str, list[str]]:
        """part_number -> real photo paths (for evaluation and extra training views)."""
        out: dict[str, list[str]] = {}
        for folder in sorted(self.root.iterdir()):
            if folder.is_dir() and folder.name != UNKNOWN_DIR:
                imgs = [
                    str(f.resolve())
                    for f in sorted(folder.iterdir())
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                ]
                if imgs:
                    out[folder.name] = imgs
        return out
