"""A demo marketplace: industries, shops, orders, and the numbers that say whether the
ranking learns from them.

Shops belong to an *industry* (a persona over catalog categories and materials) and
come back for 10-20 orders. Each item is found either by a text search or by a photo,
then bought, so the backend sees exactly what a real marketplace would: searches,
identifications, carts, checkouts. Every search is asked twice, with and without the
shop's id, so the lift from personalisation is measured on the same query; identify is
asked twice the same way. At the end the customer model's segments are compared with
the true industries, and the recommendations with what the shop bought next.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from mcmaster_vision.schemas import Part

INDUSTRIES: dict[str, dict[str, float]] = {
    "plumbing": {"Pipe, Tubing, Hose & Fittings": 0.7, "Fastening & Joining": 0.2, "Sealing": 0.1},
    "machine shop": {
        "Fastening & Joining": 0.6,
        "Power Transmission": 0.25,
        "Sawing & Cutting": 0.15,
    },
    "maintenance": {"Power Transmission": 0.5, "Fastening & Joining": 0.3, "Hardware": 0.2},
    "cabinetry": {"Hardware": 0.6, "Fastening & Joining": 0.3, "Hand Tools": 0.1},
    "fluid systems": {
        "Sealing": 0.4,
        "Pipe, Tubing, Hose & Fittings": 0.4,
        "Fastening & Joining": 0.2,
    },
    "general": {},
}


@dataclass
class Shop:
    client_id: str
    industry: str
    weights: dict[str, float]
    materials: list[str]
    staples: list[str]
    n_orders: int
    ranks: list[dict] = field(default_factory=list)


def _tokens(part: Part) -> str:
    """A search a shop would type: the part's kind words without the size."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]+", part.name) if len(w) > 2]
    return " ".join(words[-3:]) if words else part.part_number


def make_shops(
    parts: list[Part], n: int, *, seed: int, orders: tuple[int, int] = (10, 20), industries=None
) -> list[Shop]:
    rng = random.Random(seed)
    industries = industries or INDUSTRIES
    by_cat: dict[str, list[Part]] = defaultdict(list)
    for p in parts:
        by_cat[p.category_path[0] if p.category_path else "?"].append(p)
    cats = sorted(by_cat)
    materials = sorted(
        {str(p.attributes.get("material")) for p in parts if p.attributes.get("material")}
    )
    shops = []
    names = list(industries)
    for i in range(n):
        ind = names[i % len(names)]
        base = industries[ind] or dict.fromkeys(cats, 1.0)
        w = {c: max(0.0, base.get(c, 0.02) * rng.uniform(0.6, 1.4)) for c in cats}
        tot = sum(w.values()) or 1.0
        w = {c: v / tot for c, v in w.items()}
        mats = rng.sample(materials, k=min(2, len(materials))) if materials else []
        pool = [p for c, ps in by_cat.items() for p in ps if w.get(c, 0) > 0.05]
        pool = [p for p in pool if not mats or p.attributes.get("material") in mats] or pool
        staples = [p.part_number for p in rng.sample(pool, k=min(4, len(pool)))]
        shops.append(
            Shop(
                client_id=f"shop-{seed}-{i:04d}",
                industry=ind,
                weights=w,
                materials=mats,
                staples=staples,
                n_orders=rng.randint(*orders),
            )
        )
    return shops


def next_basket(shop: Shop, parts: list[Part], rng: random.Random, size: int) -> list[Part]:
    by_cat: dict[str, list[Part]] = defaultdict(list)
    for p in parts:
        by_cat[p.category_path[0] if p.category_path else "?"].append(p)
    by_pn = {p.part_number: p for p in parts}
    basket: list[Part] = []
    for _ in range(size):
        if shop.staples and rng.random() < 0.5:
            basket.append(by_pn[rng.choice(shop.staples)])
            continue
        cats, ws = zip(*shop.weights.items(), strict=True)
        cat = rng.choices(cats, weights=ws)[0]
        pool = by_cat.get(cat) or parts
        liked = [p for p in pool if p.attributes.get("material") in shop.materials]
        basket.append(rng.choice(liked if liked and rng.random() < 0.7 else pool))
    seen = set()
    return [p for p in basket if not (p.part_number in seen or seen.add(p.part_number))]


def _rank(part_numbers: list[str], pn: str) -> int | None:
    return part_numbers.index(pn) + 1 if pn in part_numbers else None


def run_market(
    client,
    parts: list[Part],
    shops: list[Shop],
    *,
    seed: int = 0,
    search_rate: float = 0.6,
    tta: str = "none",
    echo=None,
) -> dict[str, Any]:
    """Drive the shops through their orders against a TestClient of the API."""
    rng = random.Random(seed + 1)
    say = echo or (lambda *_: None)
    rec_hits = rec_shown = 0
    checkouts = 0
    max_orders = max(s.n_orders for s in shops)
    for k in range(max_orders):  # interleave shops so the model learns across all of them
        active = [s for s in shops if k < s.n_orders]
        if not active:
            break
        for shop in active:
            basket = next_basket(shop, parts, rng, rng.randint(1, 4))
            if k >= 2:
                try:
                    recs = client.get(f"/recommend?client_id={shop.client_id}&n=6").json()
                    rec_shown += 1
                    if any(r["part_number"] in {p.part_number for p in basket} for r in recs):
                        rec_hits += 1
                except Exception:  # noqa: BLE001
                    pass
            for part in basket:
                if rng.random() < search_rate:
                    q = _tokens(part)
                    plain = client.get(f"/search?q={q}&limit=30").json()
                    pers = client.get(f"/search?q={q}&limit=30&client_id={shop.client_id}").json()
                    shop.ranks.append(
                        {
                            "order": k + 1,
                            "kind": "search",
                            "plain": _rank([p["part_number"] for p in plain], part.part_number),
                            "personal": _rank([p["part_number"] for p in pers], part.part_number),
                        }
                    )
                    req = None
                else:
                    d0 = client.post(
                        f"/demo/try/{part.part_number}?seed={k * 7 + 1}&tta={tta}"
                    ).json()
                    d1 = client.post(
                        f"/demo/try/{part.part_number}?seed={k * 7 + 1}&tta={tta}&client_id={shop.client_id}"
                    ).json()
                    shop.ranks.append(
                        {
                            "order": k + 1,
                            "kind": "identify",
                            "plain": d0["rank"],
                            "personal": d1["rank"],
                        }
                    )
                    req = d1["result"]["request_id"]
                client.post(
                    "/cart",
                    json={
                        "client_id": shop.client_id,
                        "part_number": part.part_number,
                        "quantity": rng.randint(1, 3),
                        "request_id": req,
                    },
                )
            r = client.post("/checkout", json={"client_id": shop.client_id})
            checkouts += r.status_code == 200
        say(f"  round {k + 1}/{max_orders}: {len(active)} shops ordered")
    return {"checkouts": checkouts, "recommend_shown": rec_shown, "recommend_hits": rec_hits}


def _summ(rows: list[dict], key: str) -> dict[str, Any]:
    ranks = [r[key] for r in rows]
    n = len(ranks)
    return {
        "n": n,
        "top1": round(sum(r == 1 for r in ranks) / n, 3) if n else None,
        "top5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3) if n else None,
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3) if n else None,
    }


def market_report(shops: list[Shop], book, run: dict[str, Any]) -> dict[str, Any]:
    rows = [r for s in shops for r in s.ranks]
    out: dict[str, Any] = {"shops": len(shops), "orders": sum(s.n_orders for s in shops), **run}
    for kind in ("search", "identify"):
        sub = [r for r in rows if r["kind"] == kind]
        out[kind] = {"plain": _summ(sub, "plain"), "personal": _summ(sub, "personal")}
        by_bucket = {}
        for lo, hi in ((1, 3), (4, 8), (9, 99)):
            b = [r for r in sub if lo <= r["order"] <= hi]
            if b:
                by_bucket[f"orders {lo}-{hi if hi < 99 else '+'}"] = {
                    "plain_top1": _summ(b, "plain")["top1"],
                    "personal_top1": _summ(b, "personal")["top1"],
                    "n": len(b),
                }
        out[kind]["by_order"] = by_bucket
    # segments vs the true industries: weighted purity
    truth = {s.client_id: s.industry for s in shops}
    seg_members: dict[int, Counter] = defaultdict(Counter)
    for cid, prof in book.profiles.items():
        if prof.segment is not None and cid in truth:
            seg_members[prof.segment][truth[cid]] += 1
    n_assigned = sum(sum(c.values()) for c in seg_members.values())
    purity = (
        round(sum(max(c.values()) for c in seg_members.values()) / n_assigned, 3)
        if n_assigned
        else None
    )
    out["segments"] = {
        "k": len(seg_members),
        "purity": purity,
        "industries": len(set(truth.values())),
        "table": {int(k): dict(v.most_common(3)) for k, v in seg_members.items()},
    }
    out["recommend_hit_rate"] = (
        round(run["recommend_hits"] / run["recommend_shown"], 3)
        if run.get("recommend_shown")
        else None
    )
    return out


def simulate_market(
    settings,
    *,
    shops: int = 200,
    orders: tuple[int, int] = (10, 20),
    seed: int = 0,
    search_rate: float = 0.6,
    tta: str = "none",
    scratch=None,
    live: bool = False,
    echo=None,
) -> dict[str, Any]:
    import tempfile

    from fastapi.testclient import TestClient

    from mcmaster_vision.api import create_app
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.pipeline.simulate import scratch_settings

    say = echo or (lambda *_: None)
    if not live:
        scratch = scratch or tempfile.mkdtemp(prefix="mcv-market-")
        settings = scratch_settings(settings, scratch)
        say(f"scratch copy of the deployment in {scratch}")
    s = settings.model_copy(
        update={"demo_mode": True, "rate_limit_per_minute": 1_000_000, "customers_refresh_s": 5.0}
    )
    with CatalogStore(s.catalog_db) as store:
        parts = list(store.iter_parts(with_images_only=True))
    crowd = make_shops(parts, shops, seed=seed, orders=orders)
    app = create_app(s)
    with TestClient(app) as client:
        say(
            f"{len(crowd)} shops in {len(set(x.industry for x in crowd))} industries, {sum(x.n_orders for x in crowd)} orders ..."
        )
        run = run_market(
            client, parts, crowd, seed=seed, search_rate=search_rate, tta=tta, echo=say
        )
        book = app.state.customer_book()
    rep = market_report(crowd, book, run)
    for kind in ("search", "identify"):
        k = rep[kind]
        if k["plain"]["n"]:
            say(
                f"  {kind}: top-1 plain {k['plain']['top1']:.0%} -> personalised {k['personal']['top1']:.0%}, "
                f"MRR {k['plain']['mrr']} -> {k['personal']['mrr']} ({k['plain']['n']} lookups)"
            )
            for b, v in k["by_order"].items():
                say(f"    {b}: {v['plain_top1']:.0%} -> {v['personal_top1']:.0%} (n={v['n']})")
    sg = rep["segments"]
    say(f"  segments: {sg['k']} found for {sg['industries']} industries, purity {sg['purity']}")
    say(
        f"  recommendations: hit rate {rep['recommend_hit_rate']} over {rep['recommend_shown']} orders"
    )
    return rep
