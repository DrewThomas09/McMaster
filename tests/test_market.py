"""The demo marketplace: shops from several industries, orders over time, and the report
that says what personalised ranking gains."""

from __future__ import annotations

import random

from mcmaster_vision.data.market import INDUSTRIES, make_shops, next_basket, simulate_market


def test_shops_follow_their_industry(store):
    parts = list(store.iter_parts(with_images_only=True))
    shops = make_shops(parts, 12, seed=3, orders=(2, 4))
    assert len(shops) == 12 and {s.industry for s in shops} <= set(INDUSTRIES)
    assert all(2 <= s.n_orders <= 4 and s.staples for s in shops)
    rng = random.Random(1)
    plumber = next(s for s in shops if s.industry == "plumbing")
    cats = [p.category_path[0] for _ in range(30) for p in next_basket(plumber, parts, rng, 3)]
    top = max(set(cats), key=cats.count)
    assert top in ("Pipe, Tubing, Hose & Fittings", "Fastening & Joining")
    basket = next_basket(plumber, parts, rng, 4)
    assert len({p.part_number for p in basket}) == len(basket)


def test_simulate_market_reports_lift_and_segments(tmp_path, demo_dir, index):
    from mcmaster_vision.config import Settings

    s = Settings(
        data_dir=tmp_path,
        catalog_db=demo_dir / "catalog.sqlite",
        index_dir=tmp_path / "index",
        model_dir=tmp_path / "models",
        queries_dir=tmp_path / "q",
        backbone="hash",
        index_gallery_augment=0,
    )
    s.ensure_dirs()
    index.save(s.index_path)
    rep = simulate_market(
        s,
        shops=8,
        orders=(3, 5),
        seed=1,
        tta="none",
        scratch=tmp_path / "m",
        coin_rate=0.5,
        learn_every=2,
    )
    # the learning loop ran after rounds 2 and 4 and the gallery grew with the purchases
    assert [x["round"] for x in rep["learned"]] == [2, 4]
    assert any(x.get("action") == "index" for x in rep["learned"])
    assert rep["served_rows"] and rep["served_rows"] > len(index)
    assert rep["shops"] == 8 and rep["checkouts"] >= 20
    for kind in ("search", "identify"):
        k = rep[kind]
        assert k["plain"]["n"] > 0 and 0 <= k["personal"]["top1"] <= 1
        assert k["by_order"]
        # the honest split: parts the shop bought before vs parts it never bought
        assert k["seen_before"]["n"] + k["new_part"]["n"] == k["plain"]["n"]
        assert k["seen_before"]["n"] > 0 and k["new_part"]["n"] > 0
    # the facet chips: tapped only when the part was off the page, so top-1 cannot fall
    # (the honest columns are how often it changed anything and how often it hurt MRR)
    chips = rep["search"]["chips"]
    assert 0 <= chips["tapped"] <= chips["offered"] <= chips["off_page"]
    assert chips["narrowed_top1"] >= chips["personal_top1"] and chips["tapped_worse"] >= 0
    assert rep["search"]["new_part"]["narrowed_top1"] is not None
    photo = rep["identify"]
    assert photo["with_coin"]["n"] + photo["no_coin"]["n"] == photo["plain"]["n"]
    assert photo["with_coin"]["n"] > 0 and photo["no_coin"]["n"] > 0
    assert (
        photo["new_part_with_coin"]["n"] + photo["new_part_no_coin"]["n"] == photo["new_part"]["n"]
    )
    assert rep["segments"]["industries"] >= 1 and rep["segments"]["k"] >= 1
    assert 0 <= rep["recommend_hit_rate"] <= 1
    assert 0 <= rep["baseline_hit_rate"] <= 1  # the order-again list the model must beat
    assert 0 <= rep["recommend_new_hit_rate"] <= rep["recommend_hit_rate"]
    kinds = rep["recommend_new_by_kind"]
    assert kinds and all(v["bought"] <= v["shown"] for v in kinds.values())
    # the comparison arm of every photo lookup is not logged: one identify row per photo
    n_photo = rep["identify"]["plain"]["n"]
    events = (tmp_path / "m" / "logs" / "events.jsonl").read_text().splitlines()
    assert sum('"kind": "identify"' in ln for ln in events) == n_photo
    # nothing landed in the live data directory
    assert not (tmp_path / "logs" / "orders.jsonl").exists()
    assert (tmp_path / "m" / "logs" / "orders.jsonl").exists()
