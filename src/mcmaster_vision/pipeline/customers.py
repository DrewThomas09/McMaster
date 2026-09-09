"""What the orders say about each customer, and how that reshapes ranking.

A customer is a client id and its orders. From the parts bought we keep histograms over
categories, families, materials and sizes; customers with similar category mixes form
*segments* (plain k-means over their category vectors), which stand in for the industry
a shop is in (a plumber buys pipe fittings and nipples, a machine shop socket screws and
dowel pins). A customer's prior over categories blends personal history, segment and
global mix, weighted by how much history they have, so a first-time buyer gets the
global ranking and a repeat customer gets theirs.

The prior is a *tie-breaker* on ranking (search results, identification candidates),
capped so visual evidence or a text match is never overturned; complements come from
what other customers bought in the same order (co-purchase lift) and drive the
"for you" suggestions and re-order prompts.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from mcmaster_vision.schemas import Order, Part

SIZE_KEYS = ("thread_size", "pipe_size", "od", "length")
MATERIAL_KEYS = ("material",)


def category_key(part: Part, depth: int = 2) -> str:
    return " > ".join(part.category_path[:depth]) if part.category_path else "?"


@dataclass
class Profile:
    client_id: str
    orders: int = 0
    items: int = 0
    first_at: str | None = None
    last_at: str | None = None
    categories: Counter = field(default_factory=Counter)
    families: Counter = field(default_factory=Counter)
    materials: Counter = field(default_factory=Counter)
    sizes: Counter = field(default_factory=Counter)
    parts: Counter = field(default_factory=Counter)
    recent: list[str] = field(default_factory=list)  # part numbers of the last order
    bought_at: dict[str, list[datetime]] = field(default_factory=dict)  # part -> when
    segment: int | None = None

    def as_dict(self, top: int = 5) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "orders": self.orders,
            "items": self.items,
            "first_at": self.first_at,
            "last_at": self.last_at,
            "categories": self.categories.most_common(top),
            "families": self.families.most_common(top),
            "materials": self.materials.most_common(top),
            "sizes": self.sizes.most_common(top),
            "repeat_parts": [pn for pn, n in self.parts.most_common(top) if n >= 2],
            "segment": self.segment,
        }


class CustomerBook:
    """Profiles, segments, priors and complements from a list of orders."""

    def __init__(
        self,
        orders: list[Order],
        parts: dict[str, Part],
        *,
        k: int = 8,
        seed: int = 0,
        min_orders_for_segment: int = 1,
    ):
        self.parts = parts
        self.profiles: dict[str, Profile] = {}
        self.global_categories: Counter = Counter()
        self.pair_counts: Counter = Counter()
        self.pair_customers: dict[tuple[str, str], set[str]] = {}
        self.part_counts: Counter = Counter()
        self.n_orders = 0
        self._ingest(orders)
        self.categories = sorted(self.global_categories)
        self.segments: list[dict[str, Any]] = []
        self.centroids: np.ndarray | None = None
        self._segment(k, seed, min_orders_for_segment)

    # ----------------------------------------------------------------- build
    def _ingest(self, orders: list[Order]) -> None:
        for o in sorted(orders, key=lambda x: x.created_at):
            prof = self.profiles.setdefault(o.client_id, Profile(o.client_id))
            at = (
                o.created_at.isoformat()
                if isinstance(o.created_at, datetime)
                else str(o.created_at)
            )
            prof.orders += 1
            prof.first_at = prof.first_at or at
            prof.last_at = at
            prof.recent = [it.part_number for it in o.items]
            self.n_orders += 1
            pns = []
            when = o.created_at if isinstance(o.created_at, datetime) else None
            for it in o.items:
                part = self.parts.get(it.part_number)
                prof.items += it.quantity
                prof.parts[it.part_number] += 1
                if when is not None:
                    prof.bought_at.setdefault(it.part_number, []).append(when)
                self.part_counts[it.part_number] += 1
                pns.append(it.part_number)
                if part is None:
                    continue
                cat = category_key(part)
                prof.categories[cat] += 1
                self.global_categories[cat] += 1
                if part.family_id:
                    prof.families[part.family_id] += 1
                attrs = {k.lower(): str(v) for k, v in part.attributes.items()}
                for key in MATERIAL_KEYS:
                    if attrs.get(key):
                        prof.materials[attrs[key]] += 1
                for key in SIZE_KEYS:
                    if attrs.get(key):
                        prof.sizes[f"{key}={attrs[key]}"] += 1
            uniq = sorted(set(pns))
            for i, a in enumerate(uniq):
                for b in uniq[i + 1 :]:
                    self.pair_counts[(a, b)] += 1
                    self.pair_customers.setdefault((a, b), set()).add(o.client_id)

    def _vector(self, counts: Counter) -> np.ndarray:
        v = np.array([counts.get(c, 0) for c in self.categories], dtype=np.float64)
        s = v.sum()
        return v / s if s else v

    def _segment(self, k: int, seed: int, min_orders: int) -> None:
        ids = [cid for cid, p in self.profiles.items() if p.orders >= min_orders and p.categories]
        if not ids or not self.categories:
            return
        X = np.stack([self._vector(self.profiles[c].categories) for c in ids])
        k = max(1, min(k, len(ids)))
        rng = np.random.default_rng(seed)
        # k-means++ seeding, then a few Lloyd iterations: small data, no dependency
        cent = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
            cent.append(X[rng.choice(len(X), p=probs)])
        C = np.array(cent)
        labels = np.zeros(len(X), dtype=int)
        for _ in range(25):
            labels = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array(
                [X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)]
            )
            if np.allclose(newC, C):
                break
            C = newC
        self.centroids = C
        for cid, lab in zip(ids, labels, strict=True):
            self.profiles[cid].segment = int(lab)
        for j in range(k):
            members = [cid for cid, lab in zip(ids, labels, strict=True) if lab == j]
            if not members:
                continue
            mix = Counter()
            for cid in members:
                mix.update(self.profiles[cid].categories)
            top = [c for c, _ in mix.most_common(2)]
            self.segments.append(
                {
                    "segment": j,
                    "customers": len(members),
                    "orders": sum(self.profiles[c].orders for c in members),
                    "label": " + ".join(top) if top else "?",
                    "top_categories": mix.most_common(4),
                }
            )

    # ----------------------------------------------------------------- priors
    def category_prior(self, client_id: str | None) -> dict[str, float]:
        """p(category) for this customer: personal history, the segment's mix and the
        global mix blended by evidence (an unknown or new customer gets the global mix)."""
        total = sum(self.global_categories.values())
        if not total:
            return {}
        glob = {c: n / total for c, n in self.global_categories.items()}
        prof = self.profiles.get(client_id or "")
        if prof is None or not prof.categories:
            return glob
        n = sum(prof.categories.values())
        seg_mix: dict[str, float] = {}
        if prof.segment is not None and self.centroids is not None:
            seg_mix = dict(zip(self.categories, self.centroids[prof.segment], strict=True))
        m_seg, m_glob = 5.0, 2.0
        out = {}
        for c in self.categories:
            personal = prof.categories.get(c, 0)
            out[c] = (personal + m_seg * seg_mix.get(c, glob[c]) + m_glob * glob[c]) / (
                n + m_seg + m_glob
            )
        return out

    def boosts(
        self, client_id: str | None, part_numbers: list[str], weight: float = 1.0
    ) -> dict[str, float]:
        """Per-part boost in [-1, 1]. Soft evidence (the customer's category prior against
        the global mix, materials and sizes they tend to buy) is scaled by ``weight``, which
        grows with their history; strong evidence (this very part, or its family, bought
        before) counts in full from the first order. Zero for an unknown customer."""
        prof = self.profiles.get(client_id or "")
        if prof is None or not prof.categories:
            return dict.fromkeys(part_numbers, 0.0)
        prior = self.category_prior(client_id)
        total = sum(self.global_categories.values()) or 1
        n_items = max(1, sum(prof.categories.values()))
        out: dict[str, float] = {}
        for pn in part_numbers:
            part = self.parts.get(pn)
            if part is None:
                out[pn] = 0.0
                continue
            cat = category_key(part)
            g = self.global_categories.get(cat, 0) / total
            p = prior.get(cat, g)
            soft = math.log((p + 1e-3) / (g + 1e-3)) if g or p else 0.0
            soft = max(-1.5, min(1.5, soft)) / 1.5  # -> [-1, 1]
            attrs = {k.lower(): str(v) for k, v in part.attributes.items()}
            for key in SIZE_KEYS:
                if attrs.get(key) and prof.sizes.get(f"{key}={attrs[key]}"):
                    soft += 0.15 * min(1.0, prof.sizes[f"{key}={attrs[key]}"] / n_items * 4)
            for key in MATERIAL_KEYS:
                if attrs.get(key) and prof.materials.get(attrs[key]):
                    soft += 0.1 * min(1.0, prof.materials[attrs[key]] / n_items * 4)
            strong = 0.0
            if part.family_id and prof.families.get(part.family_id):
                strong += 0.3 * min(1.0, prof.families[part.family_id] / 3)
            if prof.parts.get(pn):
                strong += 0.5 + 0.1 * min(3, prof.parts[pn])
            out[pn] = max(-1.0, min(1.0, weight * soft + strong))
        return out

    def usual_values(self, client_id: str | None) -> dict[str, str]:
        """attribute -> the value this customer buys most (sizes and material), when it
        is clearly their habit (at least twice and over 40% of the times it appeared)."""
        prof = self.profiles.get(client_id or "")
        if prof is None:
            return {}
        per_key: dict[str, Counter] = defaultdict(Counter)
        for token, n in prof.sizes.items():
            key, _, val = token.partition("=")
            per_key[key][val] += n
        for val, n in prof.materials.items():
            per_key["material"][val] += n
        out = {}
        for key, c in per_key.items():
            val, n = c.most_common(1)[0]
            if n >= 2 and n / sum(c.values()) >= 0.4:
                out[key] = val
        return out

    def due(self, client_id: str | None, now: datetime | None = None) -> list[dict[str, Any]]:
        """Staples whose usual re-order interval has passed: a part bought three times or
        more, with the median gap between purchases shorter than the time since the last."""
        prof = self.profiles.get(client_id or "")
        if prof is None:
            return []
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        out = []
        for pn, times in prof.bought_at.items():
            if len(times) < 3:
                continue
            ts = sorted(t if t.tzinfo else t.replace(tzinfo=timezone.utc) for t in times)
            gaps = sorted((b - a).total_seconds() for a, b in zip(ts, ts[1:], strict=False))
            median = gaps[len(gaps) // 2]
            since = (now - ts[-1]).total_seconds()
            # a rhythm needs real days between orders; orders minutes apart are one visit
            if median >= 3600 and since >= median:
                out.append(
                    {
                        "part_number": pn,
                        "every_days": round(median / 86400, 1),
                        "last_days_ago": round(since / 86400, 1),
                        "times": len(ts),
                    }
                )
        out.sort(key=lambda d: -d["last_days_ago"] / max(d["every_days"], 0.01))
        return out

    # ------------------------------------------------------------ complements
    def complements(self, part_number: str, n: int = 5) -> list[tuple[str, float]]:
        """Parts bought in the same order as ``part_number`` by at least two different
        customers, by lift (co-purchases over what chance would give)."""
        base = self.part_counts.get(part_number, 0)
        if not base or self.n_orders < 2:
            return []
        out = []
        for (a, b), c in self.pair_counts.items():
            if part_number not in (a, b) or c < 2:
                continue
            if len(self.pair_customers.get((a, b), ())) < 2:
                continue
            other = b if a == part_number else a
            expected = base * self.part_counts.get(other, 0) / self.n_orders
            out.append((other, round(c / expected, 2) if expected else 0.0))
        out.sort(key=lambda x: -x[1])
        return out[:n]

    def recommend(
        self,
        client_id: str | None,
        n: int = 6,
        *,
        browse: Callable[[str], list[Part]] | None = None,
        new_slots: int = 1,
    ) -> list[dict[str, Any]]:
        """What to show this customer before they search: parts they re-order, then
        complements of what they bought last, then their segment's favourites.

        A list of a shop's own history fills itself after a few orders, so ``new_slots``
        places at the end are kept for things the shop has never bought (complements,
        segment favourites, and, given ``browse(category) -> parts``, unbought parts in
        its usual category and material): one thing new next to the re-orders.
        """
        prof = self.profiles.get(client_id or "")
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(pn: str, why: str, score: float, known: bool = False) -> None:
            if pn in seen or (pn not in self.parts and not known):
                return
            seen.add(pn)
            out.append({"part_number": pn, "why": why, "score": round(score, 3)})

        if prof is not None:
            for d in self.due(client_id):
                add(
                    d["part_number"],
                    f"usually every {d['every_days']:g} days, last {d['last_days_ago']:g} days ago",
                    2.0,
                )
            for pn, c in prof.parts.most_common():
                if c >= 2:
                    add(pn, f"you ordered this {c} times", 1.0 + c / 10)
            # a part bought once is still the likeliest thing to be bought again: the
            # most recent first, below the staples and above every guess. The scores
            # are tiers: due (2.0) > staples (1.0-1.3) > recent (0.8) > complements
            # (0.5-0.75) > segment favourites (0.2-0.4) > popular (0-0.2), and no
            # count, however large, crosses into the tier above
            once = sorted(
                (pn for pn, c in prof.parts.items() if c == 1),
                key=lambda pn: prof.bought_at.get(pn, [prof.last_at])[-1],
                reverse=True,
            )
            for i, pn in enumerate(once[: 2 * n]):
                why = (
                    "you ordered this recently" if pn in prof.recent else "you ordered this before"
                )
                add(pn, why, 0.8 - 0.01 * i)
            for pn in prof.recent:
                for other, lift in self.complements(pn, 4):
                    if not prof.parts.get(other):
                        add(other, f"often bought with {pn}", 0.5 + min(lift, 5) / 20)
        if prof is not None and prof.segment is not None:
            members = [p for p in self.profiles.values() if p.segment == prof.segment]
            fav: Counter = Counter()
            for p in members:
                fav.update(p.parts)
            for pn, c in fav.most_common(2 * n):
                if prof is None or not prof.parts.get(pn):
                    add(pn, "popular with shops like yours", 0.2 + 0.2 * min(1.0, c / 50))
        if prof is not None and browse is not None and prof.categories:
            # something new in the shop's usual aisle, in the material it prefers
            mats = {m for m, _ in prof.materials.most_common(2)}
            for cat, _ in prof.categories.most_common(2):
                for part in browse(cat):
                    if prof.parts.get(part.part_number):
                        continue
                    mat = str(part.attributes.get("material") or "")
                    if mats and mat not in mats:
                        continue
                    add(
                        part.part_number,
                        f"new in {cat.split(' > ')[-1]}" + (f", {mat}" if mat else ""),
                        0.45,
                        known=True,
                    )
        if not out:
            for pn, c in self.part_counts.most_common(n):
                add(pn, "popular", 0.2 * min(1.0, c / 50))
        out.sort(key=lambda r: -r["score"])  # stable: equal scores keep their reason order
        if prof is None or new_slots <= 0:
            return out[:n]
        own = [r for r in out if prof.parts.get(r["part_number"])]
        new = [r for r in out if not prof.parts.get(r["part_number"])]
        keep = max(0, n - new_slots) if new else n
        head = own[:keep]
        return (head + new[: n - len(head)] + own[keep:])[:n]

    def summary(self) -> dict[str, Any]:
        return {
            "customers": len(self.profiles),
            "orders": self.n_orders,
            "repeat_customers": sum(1 for p in self.profiles.values() if p.orders >= 2),
            "segments": self.segments,
        }


def rerank_within_tiers(
    scored: list[tuple[Any, float]],
    boosts: dict[str, float],
    *,
    tolerance: float = 0.1,
    min_boost: float = 0.05,
    pinned_below: float = -1e8,
) -> list[Any]:
    """Re-order text hits by a customer's boosts without overturning the text match.

    Hits arrive best first with their text score (bm25: more negative is stronger). A
    *tier* is a run of hits whose score is within ``tolerance`` of the tier's leader:
    size, material and finish variants of one name land in a tier together, a hit that
    matches the words less well starts a new one. Only hits inside a tier trade places,
    by boost (a history below ``min_boost`` does not count; a negative one never sinks
    a hit); part-number matches (scores below ``pinned_below``) keep their order.
    """
    out: list[Any] = []
    tier: list[Any] = []
    leader: float | None = None

    def flush() -> None:
        if not tier:
            return
        keyed = [
            (-(b if (b := boosts.get(p.part_number, 0.0)) >= min_boost else 0.0), i, p)
            for i, p in enumerate(tier)
        ]
        keyed.sort(key=lambda t: t[:2])
        out.extend(p for _, _, p in keyed)
        tier.clear()

    for part, score in scored:
        if score <= pinned_below:
            flush()
            out.append(part)
            leader = None
            continue
        if leader is None or score > leader * (1.0 - tolerance):
            flush()
            leader = score
        tier.append(part)
    flush()
    return out


def customer_boost_weight(n_orders: int) -> float:
    """How much of the boost to apply: nothing for a first order, full after five."""
    return min(1.0, n_orders / 5.0)


def segment_precision(book: CustomerBook, events) -> dict[int, dict[str, Any]]:
    """Bought-top-1 precision per customer segment (keyed by segment id, worst first) from
    the event log: which kind of shop the ranking serves worst."""
    ident_rows = {r.get("request_id"): r for r in events.rows("identify")}
    seg_of = {cid: p.segment for cid, p in book.profiles.items() if p.segment is not None}
    labels = {s["segment"]: s["label"] for s in book.segments}
    tally: dict[int, list[int]] = {}
    for c in events.rows("checkout"):
        seg = seg_of.get(c.get("client_id"))
        if seg is None:
            continue
        for it in c.get("items", []):
            src = ident_rows.get(it.get("request_id"))
            if not src:
                continue
            t = tally.setdefault(seg, [0, 0])
            t[0] += 1
            t[1] += int(src.get("best") == it.get("part_number"))
    return {
        seg: {
            "label": labels.get(seg, str(seg)),
            "bought": n,
            "top1_right": k,
            "precision": round(k / n, 3),
        }
        for seg, (n, k) in sorted(tally.items(), key=lambda kv: kv[1][1] / kv[1][0])
        if n
    }


def public_segments(book: CustomerBook, min_customers: int = 5) -> dict[str, Any]:
    """The summary for a public endpoint: segments too small to hide a single shop's
    category mix are folded into one line."""
    summ = book.summary()
    big = [s for s in summ["segments"] if s["customers"] >= min_customers]
    small = [s for s in summ["segments"] if s["customers"] < min_customers]
    if small:
        big.append(
            {
                "segment": None,
                "customers": sum(s["customers"] for s in small),
                "orders": sum(s["orders"] for s in small),
                "label": f"{len(small)} small segments",
                "top_categories": [],
            }
        )
    summ["segments"] = big
    return summ
