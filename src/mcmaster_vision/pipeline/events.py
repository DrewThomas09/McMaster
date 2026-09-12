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
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CHATTY_KINDS = frozenset({"search"})  # many per visit; kept apart so they evict nothing


class EventLog:
    def __init__(self, path: str | Path, keep: int = 20000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._recent: deque[dict] = deque(maxlen=keep)  # the customer journey
        # searches come by the hundred per visit and would push the identify and checkout
        # rows the analytics join on out of the window: their own, smaller window
        self._chatty: deque[dict] = deque(maxlen=max(1000, keep // 4))
        self._by_request: dict[str, dict] = {}  # identify rows by request_id (bounded)
        self._lock = threading.Lock()
        self.total = 0
        self._load()

    def _remember(self, row: dict) -> None:
        if row.get("kind") in CHATTY_KINDS:
            self._chatty.append(row)
            return
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
        for r in rows:  # the windows cap themselves, each kind in its own
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
                # keep the last ``keep`` journey rows and the last chatty window, in order
                keep_core, keep_chatty = self.keep, self._chatty.maxlen or self.keep
                kept: list[str] = []
                n_core = n_chatty = 0
                for ln in reversed(lines):
                    chatty = any(f'"kind": "{k}"' in ln for k in CHATTY_KINDS)
                    if chatty:
                        if n_chatty >= keep_chatty:
                            continue
                        n_chatty += 1
                    else:
                        if n_core >= keep_core:
                            continue
                        n_core += 1
                    kept.append(ln)
                kept.reverse()
                tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.writelines(kept)
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

    @staticmethod
    def append(path: str | Path, kind: str, **fields: Any) -> dict:
        """Append one row to a log file without reading it: for a process that is not
        the API (``mcv learn``, ``mcv retrain``) and must never compact it away."""
        row = {"kind": kind, "at": datetime.now(timezone.utc).isoformat(), **fields}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return row

    def rows(self, kind: str | None = None) -> list[dict]:
        with self._lock:
            if kind in CHATTY_KINDS:
                return list(self._chatty)
            rows = list(self._recent)
            if kind is None and self._chatty:
                rows = sorted([*rows, *self._chatty], key=lambda r: r.get("at", ""))
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


def search_funnel(searches: list[dict], carts: list[dict]) -> dict:
    """Typed searches in the window: how many, how many came from a facet chip (the
    variants of a name told apart by one tap), how many found nothing, and how many
    cart adds came through the search door, so the chips can be judged on real traffic
    the way ``mcv simulate-market`` judges them on synthetic shops."""
    n = len(searches)
    narrowed = sum(1 for r in searches if r.get("narrowed"))
    empty = sum(1 for r in searches if not r.get("results"))
    # searches live in a shorter window than cart adds: count the adds since the oldest
    # search kept, so the ratio compares one span with itself
    since = min((r.get("at") or "" for r in searches), default="")
    added = sum(1 for c in carts if c.get("via") == "search" and (c.get("at") or "") >= since)
    return {
        "searches": n,
        "narrowed_by_chip": narrowed,
        "narrowed_share": round(narrowed / n, 3) if n else None,
        "no_results_share": round(empty / n, 3) if n else None,
        "added_to_cart": added,
        "search_to_cart": round(min(1.0, added / n), 3) if n else None,
    }


def recommendation_take(rows: list[dict]) -> dict:
    """Of the orders placed after a For-you strip was shown, how many took a part from
    it, and how many took a part the customer had never bought before: the number that
    says whether the strip shows customers anything, or only saves them a search."""
    shown: dict[str, set[str]] = {}
    before: dict[str, set[str]] = defaultdict(set)
    n = taken = new = 0
    for r in rows:  # oldest first
        cid = r.get("client_id")
        if not cid:
            continue
        kind = r.get("kind")
        if kind == "recommend_shown":
            shown[cid] = set(r.get("parts") or [])
        elif kind == "checkout":
            items = {it.get("part_number") for it in r.get("items", [])}
            strip = shown.pop(cid, None)
            if strip is not None:
                n += 1
                got = items & strip
                taken += bool(got)
                new += bool(got - before[cid])
            before[cid] |= items
    return {
        "orders_after_strip": n,
        "take_rate": round(taken / n, 3) if n else None,
        "new_part_take_rate": round(new / n, 3) if n else None,
    }


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
            bought.append({**it, "order_id": c.get("order_id"), "at": c.get("at")})
    # what was predicted vs what was bought, per identification
    confusion: Counter = Counter()
    correct_bought = 0
    wrong_bought = 0
    conf_when_right: list[float] = []
    conf_when_wrong: list[float] = []
    tier_outcomes: dict[str, list[int]] = {}
    cat_outcomes: dict[str, list[int]] = {}
    meas_outcomes: dict[str, list[int]] = {"measured": [0, 0], "unmeasured": [0, 0]}
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
        cat = src.get("category") or "?"
        cat_outcomes.setdefault(cat, [0, 0])
        cat_outcomes[cat][0] += 1
        cat_outcomes[cat][1] += ok
        mkey = "measured" if src.get("measured") else "unmeasured"
        meas_outcomes[mkey][0] += 1
        meas_outcomes[mkey][1] += ok
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
        "recommendations": recommendation_take(events.rows()),
        # how customers reach the parts they add: a photo, a typed search, the For-you
        # strip, an old order; the split says which door is worth widening
        "found_by": dict(Counter(r.get("via") or "other" for r in carts).most_common()),
        "search": search_funnel(events.rows("search"), carts),
        "bought_rank_hist": dict(Counter(int(r) for r in ranks)),
        "tier_precision_bought": {
            t: {"bought": n, "top1_right": k, "precision": round(k / n, 3)}
            for t, (n, k) in tier_outcomes.items()
        },
        "measured_precision_bought": {
            k: {"bought": n, "top1_right": r, "precision": round(r / n, 3)}
            for k, (n, r) in meas_outcomes.items()
            if n
        },
        "category_precision_bought": {
            c: {"bought": n, "top1_right": k, "precision": round(k / n, 3)}
            for c, (n, k) in sorted(cat_outcomes.items(), key=lambda kv: kv[1][1] / kv[1][0])
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
    out["daily"] = _daily(ident, bought, by_request, events.rows("learn"))
    if feedback_stats:
        out["feedback"] = feedback_stats
    return out


def _daily(ident: list[dict], bought: list[dict], by_request: dict, learns: list[dict]) -> list:
    """One row per UTC day: identifications, parts bought, how often the bought part was
    the top answer, and learn / retrain events, so the loop's effect is visible over time."""
    days: dict[str, dict] = {}

    def row(at: str | None) -> dict | None:
        if not at:
            return None
        return days.setdefault(
            at[:10],
            {
                "day": at[:10],
                "identify": 0,
                "bought": 0,
                "bought_top1": 0,
                "learns": 0,
                "retrains": 0,
            },
        )

    for r in ident:
        d = row(r.get("at"))
        if d:
            d["identify"] += 1
    for it in bought:
        src = by_request.get(it.get("request_id"))
        d = row((src or {}).get("at") or it.get("at"))
        if d:
            d["bought"] += 1
            if src:  # a part that came from a photo: the only kind top-1 is defined for
                d["bought_from_photo"] = d.get("bought_from_photo", 0) + 1
                if src.get("best") == it.get("part_number"):
                    d["bought_top1"] += 1
    for r in learns:
        d = row(r.get("at"))
        if d:
            d["retrains" if r.get("how") == "retrain" else "learns"] += 1
    out = []
    for d in sorted(days.values(), key=lambda x: x["day"]):
        n = d.get("bought_from_photo", 0)
        d["bought_top1_rate"] = round(d["bought_top1"] / n, 3) if n else None
        out.append(d)
    return out[-30:]


def enrich_confusions(a: dict, store) -> dict:
    """Add what the catalog knows about each confusion pair: same family, and which
    attributes differ. Two SKUs with identical specs cannot be told apart by any model
    and the issue then points at the catalog data, not the model."""
    for c in a.get("confusions", []):
        p, b = store.get(c["predicted"]), store.get(c["bought"])
        if p is None or b is None:
            continue
        skip = ("price", "url", "image", "sku_url", "source")
        keys = {k for k in set(p.attributes) | set(b.attributes) if not k.startswith(skip)}
        c["same_family"] = bool(p.family_id and p.family_id == b.family_id)
        differ = sorted(
            k for k in keys if str(p.attributes.get(k, "")) != str(b.attributes.get(k, ""))
        )
        if p.name != b.name:
            differ.append("name")
        if p.category_path != b.category_path:
            differ.append("category")
        c["differ_by"] = differ
        c["equivalent"] = c["same_family"] and not differ  # another number for the same spec
    # how many wrong top answers were the same spec under another part number: not a
    # miss a photo could have avoided, and worth knowing before blaming the model
    conf = a.get("confusions", [])
    total = sum(c.get("times", 0) for c in conf)
    equiv = sum(c.get("times", 0) for c in conf if c.get("equivalent"))
    a["confusions_equivalent_share"] = round(equiv / total, 3) if total else None
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
                lever = (
                    "photograph the end of the part with a coin and Measure: the bore tells "
                    "thin-wall from thick-wall"
                    if any(k in ("wall_thickness", "schedule", "wall") for k in differ)
                    else "put a coin next to the part and Measure when it is a size"
                )
                do = (
                    f"look-alikes that differ by {', '.join(differ[:3])}: the family answer asks "
                    f"for it; {lever}"
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
    for cat, v in list(a.get("category_precision_bought", {}).items())[:2]:
        if v["bought"] >= 5 and v["precision"] < 0.5:
            out.append(
                {
                    "severity": "medium",
                    "what": f"in {cat} the top answer was bought only {v['precision']:.0%} of "
                    f"the time ({v['bought']} purchases)",
                    "do": "the weakest category: more photos of its parts (`mcv learn` after "
                    "purchases), and check its attributes and images in the catalog",
                }
            )
    for seg, v in list(a.get("segment_precision_bought", {}).items())[:1]:
        if v["bought"] >= 10 and v["precision"] < 0.5:
            out.append(
                {
                    "severity": "medium",
                    "what": f"shops that buy {v.get('label', seg)} got the top answer only "
                    f"{v['precision']:.0%} "
                    f"of the time ({v['bought']} purchases)",
                    "do": "the segment the ranking serves worst: their purchase photos teach the "
                    "most (`mcv learn`), and their categories deserve a look in the catalog",
                }
            )
    eq = a.get("confusions_equivalent_share")
    if eq is not None and eq >= 0.3:
        out.append(
            {
                "severity": "low",
                "what": f"{eq:.0%} of the wrong top answers were the same spec under another "
                "part number",
                "do": "no photo tells those apart: merge duplicate listings in the catalog, or "
                "show both numbers when the family and every attribute match",
            }
        )
    rec = a.get("recommendations") or {}
    if rec.get("orders_after_strip", 0) >= 30 and (rec.get("take_rate") or 0) < 0.2:
        out.append(
            {
                "severity": "low",
                "what": f"the For-you strip was bought from in only {rec['take_rate']:.0%} of the "
                f"{rec['orders_after_strip']} orders that followed it",
                "do": "the strip should lead with what this shop re-orders; check that orders "
                "carry the client id the phone sends and that the customer model rebuilds",
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
