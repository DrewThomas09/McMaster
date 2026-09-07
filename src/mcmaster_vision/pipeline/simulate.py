"""Self-running demos: synthetic customers walk the whole journey in-process.

Each customer photographs a catalog part (a photo-style render with its own seed),
identifies it, and if the right part is in the list adds it to the cart and, most of
the time, checks out. That leaves behind exactly what real traffic leaves behind:
identify / cart / checkout events, orders, and ``checkout`` confirmations. The
report is the same ``analytics()`` + ``issues()`` the dashboard shows, so a person
(or ``mcv simulate --learn``) can see what is wrong and whether learning fixed it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcmaster_vision.config import Settings


@dataclass
class Customer:
    part_number: str
    photo_seed: int
    client_id: str
    abandons: bool = False  # adds to the cart, never checks out
    walks_away: bool = False  # never even taps "none of these"


@dataclass
class SimReport:
    customers: int
    identified: int = 0
    found_in_list: int = 0
    top1: int = 0
    carts: int = 0
    checkouts: int = 0
    none_of_these: int = 0
    errors: list[str] = field(default_factory=list)
    ranks: dict[str, int | None] = field(default_factory=dict)  # client_id -> rank
    bought: set[str] = field(default_factory=set)  # client_ids that checked out

    @property
    def top1_rate(self) -> float | None:
        return round(self.top1 / self.identified, 3) if self.identified else None

    @property
    def found_rate(self) -> float | None:
        return round(self.found_in_list / self.identified, 3) if self.identified else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "customers": self.customers,
            "identified": self.identified,
            "found_in_list": self.found_in_list,
            "found_rate": self.found_rate,
            "top1": self.top1,
            "top1_rate": self.top1_rate,
            "carts": self.carts,
            "checkouts": self.checkouts,
            "none_of_these": self.none_of_these,
            "errors": self.errors[:10],
        }


def make_customers(
    part_numbers: list[str],
    n: int,
    *,
    seed: int = 0,
    abandon_rate: float = 0.15,
    walk_away_rate: float = 0.3,
    photo_seed_base: int = 0,
) -> list[Customer]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(
            Customer(
                part_number=rng.choice(part_numbers),
                photo_seed=photo_seed_base + rng.randrange(1, 10_000),
                client_id=f"sim-{seed}-{i}",
                abandons=rng.random() < abandon_rate,
                walks_away=rng.random() < walk_away_rate,
            )
        )
    return out


def run_customers(
    client, customers: list[Customer], *, top_n: int = 5, tta: str = "fast"
) -> SimReport:
    """Drive ``customers`` through a TestClient (or anything with .get/.post) of the API."""
    rep = SimReport(customers=len(customers))
    for c in customers:
        r = client.post(f"/demo/try/{c.part_number}?seed={c.photo_seed}&top_n={top_n}&tta={tta}")
        if r.status_code != 200:
            rep.errors.append(f"try {c.part_number}: {r.status_code} {r.text[:80]}")
            continue
        d = r.json()
        rep.identified += 1
        rank = d.get("rank")
        rep.ranks[c.client_id] = rank
        req = d["result"]["request_id"]
        if rank is None:
            if not c.walks_away:
                fr = client.post(
                    "/feedback",
                    data={
                        "request_id": req,
                        "predicted": (d["result"].get("best") or {}).get("part_number") or "",
                    },
                )
                if fr.status_code == 200:
                    rep.none_of_these += 1
                else:
                    rep.errors.append(f"feedback: {fr.status_code} {fr.text[:80]}")
            continue
        rep.found_in_list += 1
        if rank == 1:
            rep.top1 += 1
        cr = client.post(
            "/cart",
            json={"client_id": c.client_id, "part_number": c.part_number, "request_id": req},
        )
        if cr.status_code != 200:
            rep.errors.append(f"cart: {cr.status_code} {cr.text[:80]}")
            continue
        rep.carts += 1
        if c.abandons:
            continue
        ch = client.post("/checkout", json={"client_id": c.client_id})
        if ch.status_code != 200:
            rep.errors.append(f"checkout: {ch.status_code} {ch.text[:80]}")
            continue
        rep.checkouts += 1
        rep.bought.add(c.client_id)
    return rep


def _pc(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "n/a"


def _top1(ranks: dict[str, int | None], who: set[str]) -> float:
    hit = [ranks.get(c) == 1 for c in who if c in ranks]
    return round(sum(hit) / len(hit), 3) if hit else 0.0


def scratch_settings(settings: Settings, root: str | Path) -> Settings:
    """A copy of the deployment to simulate against: the catalog is shared read-only,
    the index and calibration are copied (learning rewrites them), and events, orders
    and confirmed photos go to the scratch directory, so synthetic customers never
    become real purchases, real training data or a real retrain trigger."""
    import shutil

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index_dir = root / "index"
    if settings.index_dir.exists() and not index_dir.exists():
        shutil.copytree(settings.index_dir, index_dir)
    model_dir = root / "models"
    if settings.model_dir.exists() and not model_dir.exists():
        shutil.copytree(settings.model_dir, model_dir)
    s = settings.model_copy(
        update={
            "data_dir": root,
            "index_dir": index_dir,
            "model_dir": model_dir,
            "queries_dir": root / "queries",
        }
    )
    s.ensure_dirs()
    return s


def simulate(
    settings: Settings,
    *,
    customers: int = 40,
    seed: int = 0,
    learn: bool = False,
    top_n: int = 5,
    tta: str = "fast",
    echo=None,
    live: bool = False,
    scratch: str | Path | None = None,
) -> dict[str, Any]:
    """Run customers against an in-process API; with ``learn`` fold the purchases into
    the index and run the same customers again (same photos: what the loop promises)
    plus new photos of the same parts (what generalises). Runs on a scratch copy of the
    deployment unless ``live`` (then the synthetic purchases become real data)."""
    import tempfile

    from fastapi.testclient import TestClient

    from mcmaster_vision.api import create_app
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.pipeline.events import analytics, issues
    from mcmaster_vision.pipeline.learn import learn_index

    say = echo or (lambda *_: None)
    if not live:
        scratch = scratch or tempfile.mkdtemp(prefix="mcv-simulate-")
        settings = scratch_settings(settings, scratch)
        say(f"scratch copy of the deployment in {scratch} (use --live to write real data)")
    s = settings.model_copy(update={"demo_mode": True, "rate_limit_per_minute": 100_000})
    with CatalogStore(s.catalog_db) as store:
        pns = [p.part_number for p in store.iter_parts(with_images_only=True)]
    if not pns:
        raise RuntimeError("no catalog parts with images; run mcv demo or mcv bootstrap first")
    app = create_app(s)
    out: dict[str, Any] = {"seed": seed, "customers": customers}
    with TestClient(app) as client:
        crowd = make_customers(pns, customers, seed=seed)
        say(f"{len(crowd)} customers on {len(pns)} parts ...")
        rep = run_customers(client, crowd, top_n=top_n, tta=tta)
        out["before"] = rep.as_dict()
        a = analytics(app.state.events, app.state.feedback.stats())
        out["analytics"] = a
        out["issues"] = issues(a)
        say(
            f"  found in list {_pc(rep.found_rate)}, top-1 {_pc(rep.top1_rate)}, "
            f"{rep.carts} carts, {rep.checkouts} checkouts, {rep.none_of_these} 'none of these'"
            + (f", {len(rep.errors)} errors (first: {rep.errors[0]})" if rep.errors else "")
        )
        for it in out["issues"]:
            say(f"  [{it['severity']}] {it['what']} -> {it['do']}")
        if learn:
            say("learning from the purchases (index rebuild with confirmed photos) ...")
            res = learn_index(s)
            out["learn"] = res
            say(f"  {res}")
            if res.get("action") == "index":
                # serve the learned index now (the API polls meta.json every 15 s)
                app.state.last_index_check = 0.0
                app.state.identifier = app.state.get_identifier()
                served = len(app.state.identifier.index)
                out["served_rows_after_learn"] = served
                same = run_customers(client, crowd, top_n=top_n, tta=tta)
                fresh_crowd = make_customers(pns, customers, seed=seed + 1, photo_seed_base=50_000)
                # same parts as before, new photos: what the gallery photos generalise to
                for f, c in zip(fresh_crowd, crowd, strict=True):
                    f.part_number = c.part_number
                fresh = run_customers(client, fresh_crowd, top_n=top_n, tta=tta)
                # the promise is about the photos that were bought: score those
                buyers = rep.bought
                before_b = _top1(rep.ranks, buyers)
                after_b = _top1(same.ranks, buyers)
                out["after_same_photos"] = {
                    **same.as_dict(),
                    "bought_top1_before": before_b,
                    "bought_top1_after": after_b,
                }
                out["after_new_photos"] = fresh.as_dict()
                say(
                    f"  after: the {len(buyers)} bought photos top-1 {_pc(after_b)} (was "
                    f"{_pc(before_b)}); all photos top-1 {_pc(same.top1_rate)} (was "
                    f"{_pc(rep.top1_rate)}); new photos of the same parts top-1 "
                    f"{_pc(fresh.top1_rate)}, found {_pc(fresh.found_rate)}"
                )
    return out
