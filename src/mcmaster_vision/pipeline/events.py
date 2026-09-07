"""Backend event tracking and the analytics that turn it into a list of problems.

Every step of the customer journey is one JSON line in ``data/logs/events.jsonl``:
``identify`` (tier, confidence, latency, best), ``cart_add`` / ``cart_remove``
(which candidate, its rank, was it the top answer), ``checkout`` (the parts
actually bought, each tied to the identification it came from), ``feedback``
(taps, purchases), and ``error`` (status, path). ``analytics()`` joins them into a
funnel, a confusion list (predicted X, bought Y), tier precision by outcome,
latency percentiles and error counts, and ``issues()`` names what is wrong in
plain words so a person, the dashboard, or ``mcv simulate`` can act on it.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class EventLog:
    def __init__(self, path: str | Path, keep: int = 20000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recent: deque[dict] = deque(maxlen=keep)
        self._lock = threading.Lock()
        self.total = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        self._recent.append(json.loads(ln))
                        self.total += 1
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return

    def log(self, kind: str, **fields: Any) -> dict:
        row = {"kind": kind, "at": datetime.now(timezone.utc).isoformat(), **fields}
        with self._lock:
            self._recent.append(row)
            self.total += 1
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        return row

    def rows(self, kind: str | None = None) -> list[dict]:
        with self._lock:
            rows = list(self._recent)
        return [r for r in rows if kind is None or r["kind"] == kind]


def analytics(events: EventLog, feedback_stats: dict | None = None) -> dict:
    """The journey in numbers, from the event window."""
    ident = events.rows("identify")
    carts = events.rows("cart_add")
    checkouts = events.rows("checkout")
    errors = events.rows("error")
    fb = events.rows("feedback")
    by_request = {r.get("request_id"): r for r in ident if r.get("request_id")}

    bought: list[dict] = []
    for c in checkouts:
        for it in c.get("items", []):
            bought.append({**it, "order_id": c.get("order_id")})
    # what was predicted vs what was bought, per identification
    confusion: Counter = Counter()
    correct_bought = 0
    wrong_bought = 0
    conf_when_right: list[float] = []
    conf_when_wrong: list[float] = []
    tier_outcomes: dict[str, list[int]] = {}
    for it in bought:
        src = by_request.get(it.get("request_id"))
        if not src:
            continue
        pred = src.get("best")
        truth = it.get("part_number")
        tier = src.get("tier") or "?"
        ok = int(pred == truth)
        tier_outcomes.setdefault(tier, [0, 0])
        tier_outcomes[tier][0] += 1
        tier_outcomes[tier][1] += ok
        if ok:
            correct_bought += 1
            conf_when_right.append(float(src.get("confidence") or 0))
        else:
            wrong_bought += 1
            conf_when_wrong.append(float(src.get("confidence") or 0))
            confusion[(pred or "(none)", truth)] += 1
    lat = [r["latency_ms"] for r in ident if r.get("latency_ms") is not None]
    ranks = [it.get("rank") for it in carts if it.get("rank")]
    n_ident = len(ident)
    n_cart_sessions = len({r.get("request_id") for r in carts if r.get("request_id")})
    n_checkout_sessions = len({it.get("request_id") for it in bought if it.get("request_id")})
    out = {
        "window": {
            "identify": n_ident,
            "cart_add": len(carts),
            "checkout": len(checkouts),
            "items_bought": len(bought),
            "feedback": len(fb),
            "errors": len(errors),
        },
        "funnel": {
            "identify_to_cart": round(n_cart_sessions / n_ident, 3) if n_ident else None,
            "cart_to_checkout": round(n_checkout_sessions / n_cart_sessions, 3)
            if n_cart_sessions
            else None,
            "identify_to_checkout": round(n_checkout_sessions / n_ident, 3) if n_ident else None,
        },
        "bought_top1_rate": round(correct_bought / (correct_bought + wrong_bought), 3)
        if (correct_bought + wrong_bought)
        else None,
        "bought_rank_hist": dict(Counter(int(r) for r in ranks)),
        "tier_precision_bought": {
            t: {"bought": n, "top1_right": k, "precision": round(k / n, 3)}
            for t, (n, k) in tier_outcomes.items()
        },
        "confidence": {
            "when_right": round(float(np.mean(conf_when_right)), 3) if conf_when_right else None,
            "when_wrong": round(float(np.mean(conf_when_wrong)), 3) if conf_when_wrong else None,
        },
        "confusions": [
            {"predicted": p, "bought": t, "times": n} for (p, t), n in confusion.most_common(12)
        ],
        "latency_ms": {
            "p50": round(float(np.median(lat)), 1) if lat else None,
            "p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
        },
        "errors": dict(
            Counter(f"{r.get('status')} {r.get('path')}" for r in errors).most_common(8)
        ),
        "tiers": dict(Counter(r.get("tier") for r in ident)),
    }
    if feedback_stats:
        out["feedback"] = feedback_stats
    return out


def issues(a: dict) -> list[dict]:
    """Plain-language problems with a suggested action, from ``analytics()`` output.
    Each has ``severity`` (high / medium / low), ``what`` and ``do``."""
    out: list[dict] = []
    w = a.get("window", {})
    top1 = a.get("bought_top1_rate")
    if top1 is not None and (a["window"]["items_bought"] >= 10) and top1 < 0.6:
        out.append(
            {
                "severity": "high",
                "what": f"only {top1:.0%} of purchases were the top answer",
                "do": "run `mcv learn` (confirmed photos into the gallery, retrain when enough), "
                "and check the confusion pairs below",
            }
        )
    for c in a.get("confusions", [])[:5]:
        if c["times"] >= 2:
            out.append(
                {
                    "severity": "medium",
                    "what": f"predicted {c['predicted']} but customers bought {c['bought']} "
                    f"({c['times']}x)",
                    "do": "look-alikes: add a distinguishing attribute or photo for both, or let "
                    "the size question / Measure tool decide",
                }
            )
    tp = a.get("tier_precision_bought", {})
    for tier in ("exact", "likely"):
        t = tp.get(tier)
        if t and t["bought"] >= 5 and t["precision"] < (0.9 if tier == "exact" else 0.7):
            out.append(
                {
                    "severity": "high",
                    "what": f"'{tier}' answers were right only {t['precision']:.0%} of the time "
                    f"when bought",
                    "do": "recalibrate: `mcv evaluate --query-dir data/queries --fit-calibration`",
                }
            )
    conf = a.get("confidence", {})
    if conf.get("when_wrong") is not None and conf.get("when_right") is not None:
        if conf["when_wrong"] > conf["when_right"] - 0.05:
            out.append(
                {
                    "severity": "medium",
                    "what": f"confidence does not separate right ({conf['when_right']:.2f}) from "
                    f"wrong ({conf['when_wrong']:.2f}) answers",
                    "do": "refit the temperature on confirmed photos",
                }
            )
    lat = a.get("latency_ms", {})
    if lat.get("p95") and lat["p95"] > 1500:
        out.append(
            {
                "severity": "medium",
                "what": f"p95 latency {lat['p95']:.0f} ms",
                "do": "use tta=fast for previews, FAISS above 50k vectors, more workers",
            }
        )
    if w.get("errors", 0) >= 3:
        top = next(iter(a.get("errors", {}).items()), ("?", 0))
        out.append(
            {
                "severity": "high"
                if any(k.startswith("5") for k in a.get("errors", {}))
                else "low",
                "what": f"{w['errors']} errors, most often {top[0]} ({top[1]}x)",
                "do": "see data/logs/events.jsonl and the server log",
            }
        )
    f = a.get("funnel", {})
    if (
        f.get("identify_to_cart") is not None
        and w.get("identify", 0) >= 20
        and f["identify_to_cart"] < 0.3
    ):
        out.append(
            {
                "severity": "medium",
                "what": f"only {f['identify_to_cart']:.0%} of identifications led to a cart add",
                "do": "customers are not finding their part in the list: check family recall and "
                "the hardest queries in the last evaluation",
            }
        )
    return out
