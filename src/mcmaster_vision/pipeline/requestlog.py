"""Append-only request log + in-memory summary for ``GET /metrics``.

One JSON line per identification (id, tier, best, confidence, latency, photos).
Joined with the feedback store it yields the live "confirmed top-1" rate, the
number that actually matters in production.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, deque
from pathlib import Path

import numpy as np

from mcmaster_vision.schemas import IdentificationResult


class RequestLog:
    def __init__(self, path: str | Path, keep: int = 5000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recent: deque[dict] = deque(maxlen=keep)
        self._lock = threading.Lock()
        self.total = 0

    def log(self, res: IdentificationResult) -> None:
        row = {
            "request_id": res.request_id,
            "created_at": res.created_at.isoformat(),
            "tier": res.tier.value,
            "best": res.best.part_number if res.best else None,
            "confidence": res.best.confidence if res.best else None,
            "photos": res.photos,
            "latency_ms": res.timings_ms.get("total"),
            "model": res.model_version,
        }
        with self._lock:
            self._recent.append(row)
            self.total += 1
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

    def summary(self, feedback=None) -> dict:
        with self._lock:
            rows = list(self._recent)
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        out = {
            "requests_total": self.total,
            "requests_window": len(rows),
            "tiers": dict(Counter(r["tier"] for r in rows)),
            "latency_ms": {
                "p50": round(float(np.median(lat)), 1) if lat else None,
                "p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
            },
            "multi_photo_share": round(sum(1 for r in rows if r["photos"] > 1) / len(rows), 3)
            if rows
            else 0.0,
        }
        if feedback is not None:
            fb = feedback.stats()
            out["feedback"] = fb
            out["confirmed_top1_rate"] = (
                round(fb["correct_top1"] / fb["confirmed"], 3) if fb["confirmed"] else None
            )
        return out
