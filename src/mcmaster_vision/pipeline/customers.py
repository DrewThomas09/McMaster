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
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
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
            for it in o.items:
                part = self.parts.get(it.part_number)
                prof.items += it.quantity
                prof.parts[it.part_number] += 1
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
            for i, a in enumerate(sorted(set(pns))):
                for b in sorted(set(pns))[i + 1 :]:
                    self.pair_counts[(a, b)] += 1

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

    def boosts(self, client_id: str | None, part_numbers: list[str]) -> dict[str, float]:
        """Per-part boost in [-1, 1]: log-odds of the customer's category prior against
        the global mix, plus a nudge for a family, material or size they buy, and for a
        part they bought before. Zero for an unknown customer."""
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
            b = math.log((p + 1e-3) / (g + 1e-3)) if g or p else 0.0
            b = max(-1.5, min(1.5, b)) / 1.5  # -> [-1, 1]
            if part.family_id and prof.families.get(part.family_id):
                b += 0.3 * min(1.0, prof.families[part.family_id] / 3)
            attrs = {k.lower(): str(v) for k, v in part.attributes.items()}
            for key in SIZE_KEYS:
                if attrs.get(key) and prof.sizes.get(f"{key}={attrs[key]}"):
                    b += 0.15 * min(1.0, prof.sizes[f"{key}={attrs[key]}"] / n_items * 4)
            for key in MATERIAL_KEYS:
                if attrs.get(key) and prof.materials.get(attrs[key]):
                    b += 0.1 * min(1.0, prof.materials[attrs[key]] / n_items * 4)
            if prof.parts.get(pn):
                b += 0.3
            out[pn] = max(-1.0, min(1.0, b))
        return out

    # ------------------------------------------------------------ complements
    def complements(self, part_number: str, n: int = 5) -> list[tuple[str, float]]:
        """Parts bought in the same order as ``part_number``, by lift (co-purchases over
        what chance would give), at least twice."""
        base = self.part_counts.get(part_number, 0)
        if not base or self.n_orders < 2:
            return []
        out = []
        for (a, b), c in self.pair_counts.items():
            if part_number not in (a, b) or c < 2:
                continue
            other = b if a == part_number else a
            expected = base * self.part_counts.get(other, 0) / self.n_orders
            out.append((other, round(c / expected, 2) if expected else 0.0))
        out.sort(key=lambda x: -x[1])
        return out[:n]

    def recommend(self, client_id: str | None, n: int = 6) -> list[dict[str, Any]]:
        """What to show this customer before they search: parts they re-order, then
        complements of what they bought last, then their segment's favourites."""
        prof = self.profiles.get(client_id or "")
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(pn: str, why: str, score: float) -> None:
            if pn in seen or pn not in self.parts:
                return
            seen.add(pn)
            out.append({"part_number": pn, "why": why, "score": round(score, 3)})

        if prof is not None:
            for pn, c in prof.parts.most_common():
                if c >= 2:
                    add(pn, f"you ordered this {c} times", 1.0 + c / 10)
            for pn in prof.recent:
                for other, lift in self.complements(pn, 4):
                    if not prof.parts.get(other):
                        add(other, f"often bought with {pn}", 0.5 + min(lift, 5) / 10)
        if prof is not None and prof.segment is not None:
            members = [p for p in self.profiles.values() if p.segment == prof.segment]
            fav: Counter = Counter()
            for p in members:
                fav.update(p.parts)
            for pn, c in fav.most_common(2 * n):
                if prof is None or not prof.parts.get(pn):
                    add(pn, "popular with shops like yours", 0.2 + c / 50)
        if not out:
            for pn, c in self.part_counts.most_common(n):
                add(pn, "popular", c / 50)
        return out[:n]

    def summary(self) -> dict[str, Any]:
        return {
            "customers": len(self.profiles),
            "orders": self.n_orders,
            "repeat_customers": sum(1 for p in self.profiles.values() if p.orders >= 2),
            "segments": self.segments,
        }


def customer_boost_weight(n_orders: int) -> float:
    """How much of the boost to apply: nothing for a first order, full after five."""
    return min(1.0, n_orders / 5.0)
