"""Feedback store: confirmed identifications become labelled real photos.

Every confirmation writes the query photo to ``<queries_dir>/<part_number>/`` (the
layout ``mcv evaluate --query-dir`` and ``mcv train --extra-images`` consume) and
appends a JSON line to ``feedback.jsonl``. "None of these" answers are kept under
``_unknown/`` so hard cases can be reviewed and labelled later.
"""

from __future__ import annotations

import re
import threading
import time
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
        self._lock = threading.Lock()

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
        fb = Feedback(
            request_id=request_id,
            part_number=part_number.upper() if part_number else None,
            predicted=predicted,
            tier=tier,
            image_path=str(path.resolve()),
        )
        with self._lock:
            # a corrected tap moves the photo: the earlier label must not survive on disk,
            # or training / --with-feedback would learn the same photo under both parts
            for stale in self.root.glob(f"*/{request_id}.*"):
                if stale != path:
                    stale.unlink(missing_ok=True)
            path.write_bytes(image_bytes)
            with open(self.log, "a", encoding="utf-8") as fh:
                fh.write(fb.model_dump_json() + "\n")
                fh.flush()
        return fb

    def entries(self) -> list[Feedback]:
        """All feedback, one entry per request (a re-confirmation replaces the earlier
        answer, so a corrected tap does not count twice)."""
        if not self.log.exists():
            return []
        by_request: dict[str, Feedback] = {}
        for ln in self.log.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                fb = Feedback.model_validate_json(ln)
                by_request[fb.request_id] = fb
        return list(by_request.values())

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

    def confirmation_counts(self) -> dict[str, int]:
        """part_number -> how many times users confirmed it (a usage prior for reranking)."""
        counts: dict[str, int] = {}
        for x in self.entries():
            if x.part_number:
                counts[x.part_number] = counts.get(x.part_number, 0) + 1
        return counts

    def mtime(self) -> float:
        try:
            return self.log.stat().st_mtime
        except OSError:
            return 0.0

    def labelled_images(self) -> dict[str, list[str]]:
        """part_number -> real photo paths (for evaluation and extra training views)."""
        out: dict[str, list[str]] = {}
        for folder in sorted(self.root.iterdir()):
            if folder.is_dir() and folder.name != UNKNOWN_DIR and not folder.name.startswith("."):
                imgs = [
                    str(f.resolve())
                    for f in sorted(folder.iterdir())
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                ]
                if imgs:  # folder names are part numbers; hand-made ones may be lower case
                    out.setdefault(folder.name.upper(), []).extend(imgs)
        return out


class RecentPhotos:
    """Query photos kept on disk for a while so ``POST /feedback`` can file them under the
    confirmed part even after a restart or on another worker process. Bounded by count
    and age; the newest photo wins on a request-id collision."""

    def __init__(self, root: str | Path, *, keep: int = 500, max_age_s: float = 7 * 86400):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._writes = 0

    def _path(self, request_id: str) -> Path:
        return self.root / f"{safe_segment(request_id, 'request_id')}.bin"

    def put(self, request_id: str, data: bytes) -> None:
        p = self._path(request_id)
        tmp = p.with_suffix(".tmp")
        with self._lock:
            tmp.write_bytes(data)
            tmp.replace(p)
            self._writes += 1
            if self._writes % 50 == 0:
                self.prune()

    def get(self, request_id: str) -> bytes | None:
        try:
            return self._path(request_id).read_bytes()
        except (ValueError, OSError):  # bad id, or pruned by another worker meanwhile
            return None

    def prune(self) -> int:
        """Drop photos older than ``max_age_s`` and all but the newest ``keep``."""
        stamped: list[tuple[float, Path]] = []
        for f in self.root.glob("*.bin"):
            try:  # another worker may prune the same directory at the same time
                stamped.append((f.stat().st_mtime, f))
            except OSError:
                continue
        stamped.sort(reverse=True)
        now = time.time()
        removed = 0
        for i, (mtime, f) in enumerate(stamped):
            if i >= self.keep or now - mtime > self.max_age_s:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*.bin"))
