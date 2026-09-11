"""Cart and checkout: the purchase loop that teaches the model.

A cart is keyed by a client id the phone generates; items carry the identification
they came from. A checkout writes an order, files every item's query photo as a
``checkout`` confirmation (the strongest evidence there is: the customer paid for
it), and logs events for the analytics. This is a demo storefront, not a payment
system: no money moves and no personal data is collected.

Carts live on disk (one small JSON file per client) so every worker process sees
the same cart, and a photo's identification is looked up in the shared event log,
so a cart add and its checkout may land on different workers. A request id is a
random 128-bit token only the phone that made the identification holds, which is
what ties a purchase to a photo (the same trust as ``POST /feedback``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from mcmaster_vision.schemas import CartItem, Order

router = APIRouter(tags=["commerce"])


def _rate(request: Request) -> None:
    """Cart calls are cheap and frequent (every page load): they get their own budget,
    ten times the photo limit, instead of spending the /identify one."""
    from mcmaster_vision.api.ratelimit import RateLimiter

    limiter = getattr(request.app.state, "cart_limiter", None)
    if limiter is None:
        limiter = RateLimiter(10 * request.app.state.settings.rate_limit_per_minute)
        request.app.state.cart_limiter = limiter
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(RateLimiter.bucket(client)):
        raise HTTPException(429, "rate limit exceeded; try again in a minute")


_SAFE_CLIENT = 64
_CLIENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _client(client_id: str | None) -> str:
    cid = (client_id or "").strip()
    if not _CLIENT_RE.match(cid):
        raise HTTPException(400, "give a client_id (letters, digits, - or _, up to 64)")
    return cid


class Carts:
    """Carts as one JSON file per client under ``carts/`` (shared by all workers, pruned
    by age) and an append-only orders file."""

    MAX_LINES = 50  # distinct parts in one cart: a real order, not a scraper's dump
    MAX_CARTS = 20_000  # cart files on disk before the oldest go, whatever their age

    def __init__(self, orders_path: str | Path, *, max_age_s: float = 14 * 86400):
        self.orders_path = Path(orders_path)
        self.orders_path.parent.mkdir(parents=True, exist_ok=True)
        self.dir = self.orders_path.parent / "carts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._last_prune = 0.0

    def _path(self, cid: str) -> Path:
        return self.dir / f"{cid}.json"

    def _read(self, cid: str) -> list[CartItem]:
        try:
            raw = json.loads(self._path(cid).read_text(encoding="utf-8"))
            return [CartItem.model_validate(x) for x in raw]
        except (OSError, ValueError):
            return []

    def _write(self, cid: str, cart: list[CartItem]) -> None:
        path = self._path(cid)
        if not cart:
            path.unlink(missing_ok=True)
            return
        tmp = path.with_name(f"{cid}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(json.dumps([it.model_dump(mode="json") for it in cart]), encoding="utf-8")
        tmp.replace(path)

    @contextlib.contextmanager
    def _locked(self, cid: str):
        """One read-modify-write at a time per cart, across worker processes."""
        import fcntl

        with self._lock, open(self.dir / f"{cid}.lock", "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def prune(self) -> int:
        """Drop carts nobody touched for ``max_age_s`` (at most once a minute)."""
        now = time.time()
        if now - self._last_prune < 60:
            return 0
        self._last_prune = now
        n = 0
        carts = list(self.dir.glob("*.json"))
        for f in carts + list(self.dir.glob("*.lock")):
            try:
                old = now - f.stat().st_mtime > self.max_age_s
                # a lock whose cart is gone (checked out, emptied) is a stray after a minute
                stray = f.suffix == ".lock" and not f.with_suffix(".json").exists()
                if old or (stray and now - f.stat().st_mtime > 60):
                    f.unlink()
                    n += 1
            except OSError:
                continue
        if len(carts) > self.MAX_CARTS:  # a flood of throwaway ids: the oldest tenth goes
            carts.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0)
            for f in carts[: len(carts) // 10]:
                try:
                    f.unlink()
                    n += 1
                except OSError:
                    continue
        return n

    def get(self, cid: str) -> list[CartItem]:
        with self._lock:
            return self._read(cid)

    def add(self, cid: str, item: CartItem, *, set_quantity: bool = False) -> list[CartItem]:
        with self._locked(cid):
            cart = self._read(cid)
            for it in cart:
                if it.part_number == item.part_number:
                    it.quantity = min(
                        999, item.quantity if set_quantity else it.quantity + item.quantity
                    )
                    it.request_id = it.request_id or item.request_id
                    if item.request_id and it.confidence is None:
                        it.confidence, it.tier = item.confidence, item.tier
                    break
            else:
                if len(cart) >= self.MAX_LINES:
                    raise ValueError(f"a cart holds at most {self.MAX_LINES} different parts")
                cart.append(item)
            cart = [it for it in cart if it.quantity > 0]
            self._write(cid, cart)
            self.prune()
            return list(cart)

    def remove(self, cid: str, part_number: str) -> list[CartItem]:
        if not self._path(cid).exists():
            return []  # nothing to remove, and no files for an id that never had a cart
        with self._locked(cid):
            cart = [it for it in self._read(cid) if it.part_number != part_number]
            self._write(cid, cart)
            return list(cart)

    def clear(self, cid: str) -> None:
        with self._locked(cid):
            self._write(cid, [])

    def place(self, cid: str) -> Order:
        with self._locked(cid):
            items = self._read(cid)
            if not items:
                raise HTTPException(400, "the cart is empty")
            self._write(cid, [])
        total = sum((it.price_usd or 0.0) * it.quantity for it in items)
        order = Order(
            order_id=uuid.uuid4().hex[:10].upper(),
            client_id=cid,
            items=items,
            total_usd=round(total, 2) if any(it.price_usd for it in items) else None,
        )
        return order

    def save_order(self, order: Order) -> None:
        with self._lock, open(self.orders_path, "a", encoding="utf-8") as fh:
            fh.write(order.model_dump_json() + "\n")

    def all_orders(self) -> list[Order]:
        """Every order on disk (for the customer model); cached until the file changes."""
        if not self.orders_path.exists():
            return []
        try:
            st = self.orders_path.stat()
        except OSError:
            return []
        sig = (st.st_mtime_ns, st.st_size)
        cached = getattr(self, "_all_cache", None)
        if cached and cached[0] == sig:
            return cached[1]
        out: list[Order] = []
        with self._lock, open(self.orders_path, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                try:
                    out.append(Order.model_validate_json(ln))
                except ValueError:
                    continue
        self._all_cache = (sig, out)
        return out

    def orders(self, limit: int = 50, client_id: str | None = None) -> list[Order]:
        if not self.orders_path.exists():
            return []
        with self._lock, open(self.orders_path, "rb") as fh:
            # the newest orders are at the end: read a bounded tail, never the whole history
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 512 * 1024 - 1))
            raw = fh.read()
        if size > 512 * 1024:
            raw = raw.partition(b"\n")[2]  # drop the fragment before the first newline
        lines = raw.decode("utf-8", errors="replace").splitlines()
        out: list[Order] = []
        for ln in reversed(lines):
            if not ln.strip():
                continue
            try:
                o = Order.model_validate_json(ln)
            except ValueError:  # a half-written line never breaks the list
                continue
            if client_id is None or o.client_id == client_id:
                out.append(o)
            if len(out) >= limit:
                break
        return out


class AddToCart(BaseModel):
    client_id: str
    part_number: str
    quantity: int = Field(1, ge=1, le=999)
    request_id: str | None = None
    set_quantity: bool = Field(False, description="Replace the line's quantity instead of adding")


def _price(part) -> float | None:
    raw = part.attributes.get("price_usd") if part else None
    try:
        return float(str(raw).replace("$", "").replace(",", "")) if raw not in (None, "") else None
    except ValueError:
        return None


@router.get("/cart", response_model=list[CartItem])
def get_cart(request: Request, client_id: str = Query(...)):
    _rate(request)
    return request.app.state.carts.get(_client(client_id))


@router.post("/cart", response_model=list[CartItem])
def add_to_cart(body: AddToCart, request: Request):
    _rate(request)
    cid = _client(body.client_id)
    ident = request.app.state.get_identifier()
    part = ident.store.get(body.part_number.upper())
    if part is None:
        raise HTTPException(404, "unknown part number")
    # what did the identification say about this part? (rank 1 = the top answer)
    rank, confidence, tier = None, None, None
    src = request.app.state.events.identify_row(body.request_id) if body.request_id else None
    if src:
        ranked = src.get("candidates") or []
        rank = ranked.index(part.part_number) + 1 if part.part_number in ranked else None
        confidence = src.get("confidence") if src.get("best") == part.part_number else None
        tier = src.get("tier")
    item = CartItem(
        part_number=part.part_number,
        quantity=body.quantity,
        request_id=body.request_id if src else None,  # an unknown id ties nothing to the photo
        confidence=confidence,
        tier=tier,
        name=part.name,
        price_usd=_price(part),
    )
    try:
        cart = request.app.state.carts.add(cid, item, set_quantity=body.set_quantity)
    except ValueError as e:  # the cart is full
        raise HTTPException(400, str(e)) from e
    if not body.set_quantity:
        request.app.state.events.log(
            "cart_add",
            client_id=cid,
            part_number=part.part_number,
            request_id=item.request_id,
            rank=rank,
            was_top=bool(src and src.get("best") == part.part_number),
            quantity=body.quantity,
        )
        # a cart add is weak evidence (weight 1) that the photo showed this part; a
        # checkout upgrades the same request id to weight 3, an abandoned cart keeps it.
        # Two different parts from one photo (a customer comparing) is no evidence at all
        if src and item.request_id and not _ambiguous(cart, item.request_id):
            _file(request, item.request_id, part.part_number, src, source="cart")
    return cart


def _ambiguous(items: list[CartItem], request_id: str) -> bool:
    return len({it.part_number for it in items if it.request_id == request_id}) > 1


def _file(request: Request, request_id: str, part_number: str, src: dict, *, source: str) -> bool:
    photo = request.app.state.recent.get(request_id)
    if photo is None:
        return False
    try:
        request.app.state.feedback.record(
            photo,
            request_id,
            part_number,
            predicted=src.get("best"),
            tier=src.get("tier"),
            source=source,
        )
        return True
    except ValueError:
        return False


@router.delete("/cart/{part_number}", response_model=list[CartItem])
def remove_from_cart(part_number: str, request: Request, client_id: str = Query(...)):
    _rate(request)
    cid = _client(client_id)
    cart = request.app.state.carts.remove(cid, part_number.upper())
    request.app.state.events.log("cart_remove", client_id=cid, part_number=part_number.upper())
    return cart


class CheckoutBody(BaseModel):
    client_id: str


@router.post("/checkout", response_model=Order)
def checkout(body: CheckoutBody, request: Request):
    """Place the order. Every item that came from an identification becomes a
    ``checkout`` confirmation of that photo: the model learns from what was bought."""
    _rate(request)
    cid = _client(body.client_id)
    order = request.app.state.carts.place(cid)
    learned = 0
    for it in order.items:
        if not it.request_id or _ambiguous(order.items, it.request_id):
            continue  # one photo, two parts bought: no label is better than a coin flip
        src = request.app.state.events.identify_row(it.request_id) or {}
        if _file(request, it.request_id, it.part_number, src, source="checkout"):
            learned += 1
    order.learned = learned
    request.app.state.carts.save_order(order)
    inv = getattr(request.app.state, "customers_invalidate", None)
    if inv is not None:
        inv()
    request.app.state.events.log(
        "checkout",
        client_id=cid,
        order_id=order.order_id,
        items=[
            {"part_number": it.part_number, "quantity": it.quantity, "request_id": it.request_id}
            for it in order.items
        ],
        learned=learned,
        total_usd=order.total_usd,
    )
    return order


@router.get("/orders", response_model=list[Order])
def orders(
    request: Request,
    limit: int = Query(20, ge=1, le=200),
    client_id: str | None = Query(None, description="Your own orders; all orders need the token"),
):
    """A phone sees its own orders; the full list (every client id) is an admin view."""
    _rate(request)
    if client_id:
        return request.app.state.carts.orders(limit, client_id=_client(client_id))
    request.app.state.check_admin(request)
    return request.app.state.carts.orders(limit)
