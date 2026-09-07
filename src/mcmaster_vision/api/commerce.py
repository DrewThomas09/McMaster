"""Cart and checkout: the purchase loop that teaches the model.

A cart is keyed by a client id the phone generates; items carry the identification
they came from. A checkout writes an order, files every item's query photo as a
``checkout`` confirmation (the strongest evidence there is: the customer paid for
it), and logs events for the analytics. This is a demo storefront, not a payment
system: no money moves and no personal data is collected.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from mcmaster_vision.schemas import CartItem, Order

router = APIRouter(tags=["commerce"])
_SAFE_CLIENT = 64


def _client(client_id: str | None) -> str:
    cid = (client_id or "").strip()
    if not cid or len(cid) > _SAFE_CLIENT or not cid.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "give a client_id (letters, digits, - or _)")
    return cid


class Carts:
    """In-memory carts (ephemeral by design) and an append-only orders file."""

    def __init__(self, orders_path: str | Path):
        self.orders_path = Path(orders_path)
        self.orders_path.parent.mkdir(parents=True, exist_ok=True)
        self._carts: dict[str, list[CartItem]] = {}
        self._lock = threading.Lock()

    def get(self, cid: str) -> list[CartItem]:
        with self._lock:
            return list(self._carts.get(cid, []))

    def add(self, cid: str, item: CartItem) -> list[CartItem]:
        with self._lock:
            cart = self._carts.setdefault(cid, [])
            for it in cart:
                if it.part_number == item.part_number:
                    it.quantity = min(999, it.quantity + item.quantity)
                    it.request_id = it.request_id or item.request_id
                    break
            else:
                cart.append(item)
            return list(cart)

    def remove(self, cid: str, part_number: str) -> list[CartItem]:
        with self._lock:
            cart = [it for it in self._carts.get(cid, []) if it.part_number != part_number]
            self._carts[cid] = cart
            return list(cart)

    def clear(self, cid: str) -> None:
        with self._lock:
            self._carts.pop(cid, None)

    def place(self, cid: str) -> Order:
        with self._lock:
            items = list(self._carts.get(cid, []))
            if not items:
                raise HTTPException(400, "the cart is empty")
            self._carts.pop(cid, None)
        total = sum((it.price_usd or 0.0) * it.quantity for it in items)
        order = Order(
            order_id=uuid.uuid4().hex[:10].upper(),
            client_id=cid,
            items=items,
            total_usd=round(total, 2) if any(it.price_usd for it in items) else None,
        )
        with self._lock, open(self.orders_path, "a", encoding="utf-8") as fh:
            fh.write(order.model_dump_json() + "\n")
        return order

    def orders(self, limit: int = 50) -> list[Order]:
        if not self.orders_path.exists():
            return []
        lines = self.orders_path.read_text(encoding="utf-8").splitlines()
        return [Order.model_validate_json(ln) for ln in lines[-limit:] if ln.strip()][::-1]


class AddToCart(BaseModel):
    client_id: str
    part_number: str
    quantity: int = Field(1, ge=1, le=999)
    request_id: str | None = None


def _price(part) -> float | None:
    raw = part.attributes.get("price_usd") if part else None
    try:
        return float(str(raw).replace("$", "")) if raw not in (None, "") else None
    except ValueError:
        return None


@router.get("/cart", response_model=list[CartItem])
def get_cart(request: Request, client_id: str = Query(...)):
    return request.app.state.carts.get(_client(client_id))


@router.post("/cart", response_model=list[CartItem])
def add_to_cart(body: AddToCart, request: Request):
    cid = _client(body.client_id)
    ident = request.app.state.get_identifier()
    part = ident.store.get(body.part_number.upper())
    if part is None:
        raise HTTPException(404, "unknown part number")
    # what did the identification say about this part? (rank 1 = the top answer)
    rank, confidence, tier = None, None, None
    src = None
    if body.request_id:
        for r in request.app.state.events.rows("identify"):
            if r.get("request_id") == body.request_id:
                src = r
        if src:
            ranked = src.get("candidates") or []
            rank = ranked.index(part.part_number) + 1 if part.part_number in ranked else None
            confidence = src.get("confidence") if src.get("best") == part.part_number else None
            tier = src.get("tier")
    item = CartItem(
        part_number=part.part_number,
        quantity=body.quantity,
        request_id=body.request_id,
        confidence=confidence,
        tier=tier,
        name=part.name,
        price_usd=_price(part),
    )
    cart = request.app.state.carts.add(cid, item)
    request.app.state.events.log(
        "cart_add",
        client_id=cid,
        part_number=part.part_number,
        request_id=body.request_id,
        rank=rank,
        was_top=bool(src and src.get("best") == part.part_number),
        quantity=body.quantity,
    )
    return cart


@router.delete("/cart/{part_number}", response_model=list[CartItem])
def remove_from_cart(part_number: str, request: Request, client_id: str = Query(...)):
    cid = _client(client_id)
    cart = request.app.state.carts.remove(cid, part_number.upper())
    request.app.state.events.log("cart_remove", client_id=cid, part_number=part_number.upper())
    return cart


class CheckoutBody(BaseModel):
    client_id: str


@router.post("/checkout", response_model=Order)
async def checkout(body: CheckoutBody, request: Request):
    """Place the order. Every item that came from an identification becomes a
    ``checkout`` confirmation of that photo: the model learns from what was bought."""
    cid = _client(body.client_id)
    order = request.app.state.carts.place(cid)
    learned = 0
    ident_rows = {r.get("request_id"): r for r in request.app.state.events.rows("identify")}
    for it in order.items:
        if not it.request_id:
            continue
        photo = request.app.state.recent.get(it.request_id)
        if photo is None:
            continue
        src = ident_rows.get(it.request_id, {})
        try:
            request.app.state.feedback.record(
                photo,
                it.request_id,
                it.part_number,
                predicted=src.get("best"),
                tier=src.get("tier"),
                source="checkout",
            )
            learned += 1
        except ValueError:
            continue
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
def orders(request: Request, limit: int = Query(20, ge=1, le=200)):
    return request.app.state.carts.orders(limit)
