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
        self.keep = keep
        self._recent: deque[dict] = deque(maxlen=keep)
        self._by_request: dict[str, dict] = {}  # identify rows by request_id (bounded)
        self._lock = threading.Lock()
        self.total = 0
        self._load()

    def _remember(self, row: dict) -> None:
        self._recent.append(row)
        if row.get("kind") == "identify" and row.get("request_id"):
            self._by_request[row["request_id"]] = row
            if len(self._by_request) > self.keep:
                self._by_request.pop(next(iter(self._by_request)))

    def _load(self) -> None:
        """Read the file at boot, keeping the last ``keep`` rows; a file more than twice
        that long is compacted so boots stay fast and the disk bounded."""
        if not self.path.exists():
            return
        rows: list[dict] = []
        try:
            with open(self.path, encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(r, dict) and r.get("kind"):
                        rows.append(r)
        except OSError:
            return
        self.total = len(rows)
        for r in rows[-self.keep :]:
            self._remember(r)
        if len(rows) > 2 * self.keep:
            self._compact()

    def _compact(self) -> None:
        """Rewrite the file with the last ``keep`` rows, under a lock so several workers
        booting together do not race, and re-reading under the lock so rows another
        worker appended meanwhile are kept."""
        import fcntl
        import os

        try:
            with open(self.path.with_suffix(".lock"), "a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with open(self.path, encoding="utf-8") as fh:
                    lines = [ln for ln in fh if ln.strip()]
                if len(lines) <= 2 * self.keep:
                    return  # another worker already did it
                tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.writelines(lines[-self.keep :])
                tmp.replace(self.path)
        except OSError:
            pass

    def log(self, kind: str, **fields: Any) -> dict:
        row = {"kind": kind, "at": datetime.now(timezone.utc).isoformat(), **fields}
        with self._lock:
            self._remember(row)
            self.total += 1
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        return row

    def rows(self, kind: str | None = None) -> list[dict]:
        with self._lock:
            rows = list(self._recent)
        return [r for r in rows if kind is None or r.get("kind") == kind]

    def identify_row(self, request_id: str | None) -> dict | None:
        """The identify event for a request id: this process's window first, then the
        shared file (another worker may have served the photo)."""
        if not request_id:
            return None
        with self._lock:
            row = self._by_request.get(request_id)
        if row is not None:
            return row
        needle = f'"request_id": "{request_id}"'
        found = None
        try:
            with open(self.path, encoding="utf-8") as fh:
                for ln in fh:
                    if needle in ln and '"kind": "identify"' in ln:
                        try:
                            found = json.loads(ln)
                        except json.JSONDecodeError:
                            continue
        except OSError:
            return None
        return found if isinstance(found, dict) else None


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
    n_none = len([r for r in fb if r.get("part_number") is None and r.get("source") == "tap"])
    n_cart_sessions = len({r.get("request_id") for r in carts if r.get("request_id")})
    n_checkout_sessions = len({it.get("request_id") for it in bought if it.get("request_id")})
    out = {
        "window": {
            "identify": n_ident,
            "cart_add": len(carts),
            "checkout": len(checkouts),
            "items_bought": len(bought),
            "feedback": len(fb),
            "none_of_these": n_none,
            "errors": len(errors),
        },
        "funnel": {
            "identify_to_cart": round(n_cart_sessions / n_ident, 3) if n_ident else None,
            "cart_to_checkout": round(min(1.0, n_checkout_sessions / n_cart_sessions), 3)
            if n_cart_sessions
            else None,
            "identify_to_checkout": round(n_checkout_sessions / n_ident, 3) if n_ident else None,
            "none_of_these": round(n_none / n_ident, 3) if n_ident else None,
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


def enrich_confusions(a: dict, store) -> dict:
    """Add what the catalog knows about each confusion pair: same family, and which
    attributes differ. Two SKUs with identical specs cannot be told apart by any model
    and the issue then points at the catalog data, not the model."""
    for c in a.get("confusions", []):
        p, b = store.get(c["predicted"]), store.get(c["bought"])
        if p is None or b is None:
            continue
        keys = set(p.attributes) | set(b.attributes)
        c["same_family"] = bool(p.family_id and p.family_id == b.family_id)
        c["differ_by"] = sorted(
            k for k in keys if str(p.attributes.get(k, "")) != str(b.attributes.get(k, ""))
        )
    return a


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
            differ = c.get("differ_by")
            if differ == []:
                do = (
                    "these two SKUs have identical specifications in the catalog: no model can "
                    "tell them apart; add the attribute that differs (length, size, finish) to "
                    "the catalog data"
                )
            elif differ:
                do = (
                    f"look-alikes that differ by {', '.join(differ[:3])}: the family answer asks "
                    "for it; put a coin next to the part and Measure when it is a size"
                )
            else:
                do = (
                    "look-alikes: add a distinguishing attribute or photo for both, or let "
                    "the size question / Measure tool decide"
                )
            out.append(
                {
                    "severity": "medium",
                    "what": f"predicted {c['predicted']} but customers bought {c['bought']} "
                    f"({c['times']}x)",
                    "do": do,
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
                    "do": "recalibrate on real outcomes (`mcv learn` does it from confirmed "
                    "photos); if the tier does not move, this backbone cannot separate the "
                    "look-alikes: retrain, or a stronger backbone",
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
                "severity": "high",
                "what": f"{w['errors']} server errors, most often {top[0]} ({top[1]}x)",
                "do": "see data/logs/events.jsonl and the server log",
            }
        )
    f = a.get("funnel", {})
    if (
        f.get("none_of_these") is not None
        and w.get("identify", 0) >= 20
        and f["none_of_these"] > 0.25
    ):
        out.append(
            {
                "severity": "high",
                "what": f"{f['none_of_these']:.0%} of customers found nothing in the list",
                "do": "the part is not in the top candidates: retrain (`mcv learn` once enough "
                "purchases arrived), raise top_n, and label the photos under data/queries/_unknown",
            }
        )
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
