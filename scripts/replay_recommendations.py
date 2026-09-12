"""Replay a market's order log through the recommender, round by round, and score
variants against the order-again baseline without re-running the simulation.

    python3 scripts/replay_recommendations.py ORDERS.jsonl CATALOG.sqlite [--n 6]

Round k is every shop's k-th order. The customer book is built from the orders of the
rounds before, as the nightly rebuild in ``mcv simulate-market`` does, and each variant
is asked for ``n`` parts for the shop; a hit is a recommended part in that order, a new
hit one the shop had never bought. Variants are named in ``VARIANTS`` below.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from mcmaster_vision.catalog import CatalogStore
from mcmaster_vision.pipeline.customers import CustomerBook
from mcmaster_vision.schemas import Order


def load_orders(path: Path) -> list[Order]:
    with open(path, encoding="utf-8") as fh:
        return [Order.model_validate_json(ln) for ln in fh if ln.strip()]


def by_round(orders: list[Order]) -> dict[int, list[Order]]:
    per_shop: dict[str, list[Order]] = defaultdict(list)
    for o in orders:  # the log is in placement order
        per_shop[o.client_id].append(o)
    rounds: dict[int, list[Order]] = defaultdict(list)
    for shop_orders in per_shop.values():
        for k, o in enumerate(shop_orders):
            rounds[k].append(o)
    return rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("orders")
    ap.add_argument("catalog")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--first-round", type=int, default=2, help="score from this order on")
    args = ap.parse_args()
    orders = load_orders(Path(args.orders))
    rounds = by_round(orders)
    with CatalogStore(args.catalog) as store:
        parts = {p.part_number: p for p in store.iter_parts()}

        def browse(path, mat):
            return store.by_category(path, limit=40, material=mat)

        variants = {
            "order_again (baseline)": None,
            "recommend new_slots=1 (shipped)": dict(new_slots=1, browse=browse),
            "recommend new_slots=0": dict(new_slots=0, browse=browse),
            "recommend new_slots=1, no aisle": dict(new_slots=1, browse=None),
            "recommend new_slots=2": dict(new_slots=2, browse=browse),
            "new slot only over a stale once-bought": "conditional",
        }

        def conditional(book, client_id, n):
            """The new slot only when the re-order it would displace is a part bought
            once and not recently (the weakest re-order); a staple keeps its place."""
            own = book.recommend(client_id, n, browse=browse, new_slots=0)
            if len(own) < n or own[-1]["why"] == "you ordered this before":
                return book.recommend(client_id, n, browse=browse, new_slots=1)
            return own

        hits: Counter = Counter()
        new_hits: Counter = Counter()
        slot_hits: dict[str, Counter] = defaultdict(Counter)
        shown = 0
        history: list[Order] = []
        bought_before: dict[str, Counter] = defaultdict(Counter)
        for k in sorted(rounds):
            if k >= args.first_round and history:
                book = CustomerBook(history, parts)
                for o in rounds[k]:
                    want = {it.part_number for it in o.items}
                    before = bought_before[o.client_id]
                    shown += 1
                    for name, kw in variants.items():
                        if kw is None:
                            recs = [pn for pn, _ in before.most_common(args.n)]
                        elif kw == "conditional":
                            recs = [
                                r["part_number"] for r in conditional(book, o.client_id, args.n)
                            ]
                        else:
                            recs = [
                                r["part_number"] for r in book.recommend(o.client_id, args.n, **kw)
                            ]
                        got = [pn for pn in recs if pn in want]
                        hits[name] += bool(got)
                        new_hits[name] += any(pn not in before for pn in got)
                        for i, pn in enumerate(recs):
                            slot_hits[name][i] += pn in want
            for o in rounds[k]:
                history.append(o)
                bought_before[o.client_id].update(it.part_number for it in o.items)
        print(f"{len(orders)} orders, {len(rounds)} rounds, {shown} orders scored (n={args.n})")
        for name in variants:
            slots = " ".join(f"{slot_hits[name][i] / shown:.0%}" for i in range(args.n))
            print(
                f"  {name:36s} hit {hits[name] / shown:.1%}  never-bought hit "
                f"{new_hits[name] / shown:.1%}  by slot: {slots}"
            )


if __name__ == "__main__":
    main()
