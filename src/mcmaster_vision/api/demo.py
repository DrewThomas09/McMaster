"""Demo helpers: sample parts to try, a printable sheet, and the connect page.

Enabled with ``MCV_DEMO_MODE=true`` (``mcv demo`` / ``mcv up`` set it). None of
this is needed in production; it exists so a demo works with no physical parts
at hand: tap a sample to identify a photo-style render of it, or print the
sheet and photograph the paper with the phone.
"""

from __future__ import annotations

import html
import io
import random

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageDraw

from mcmaster_vision.data.augment import AugmentConfig, PhotoAugmenter
from mcmaster_vision.pipeline.identify import Identifier

router = APIRouter(tags=["demo"])


def _ident(request: Request) -> Identifier:
    # through the app's accessor so a rebuilt index (mcv learn / retrain) is picked up
    ident = request.app.state.get_identifier()
    if not request.app.state.settings.demo_mode:
        raise HTTPException(404, "demo mode is off (MCV_DEMO_MODE=true)")
    return ident


def _sample_parts(ident: Identifier, n: int, seed: int | None):
    parts = [p for p in ident.store.iter_parts(with_images_only=True)]
    rng = random.Random(seed)
    rng.shuffle(parts)
    # spread across families so the strip shows variety
    seen: set[str] = set()
    out = []
    for p in parts:
        key = p.family_id or p.part_number
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= n:
            break
    return out


@router.get("/demo/samples")
def samples(
    n: int = Query(12, ge=1, le=48), seed: int | None = None, ident: Identifier = Depends(_ident)
) -> list[dict]:
    """Random parts (one per family) to try without a physical part."""
    return [
        {
            "part_number": p.part_number,
            "name": p.name,
            "category": p.category,
            "thumb": f"/parts/{p.part_number}/thumb?size=160",
        }
        for p in _sample_parts(ident, n, seed)
    ]


@router.get("/demo/query/{part_number}")
def query_image(part_number: str, seed: int = Query(0, ge=0), ident: Identifier = Depends(_ident)):
    """A photo-style augmented render of the part (what the demo identifies)."""
    part = ident.store.get(part_number)
    if part is None or not part.image_paths:
        raise HTTPException(404, "unknown part")
    aug = PhotoAugmenter(AugmentConfig.evaluation(), seed=seed)
    img = aug(Image.open(part.image_paths[seed % len(part.image_paths)]), out_size=512)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


COIN_MM = 24.26  # a US quarter


def with_coin(
    render: Image.Image, part_mm: float, seed: int = 0
) -> tuple[Image.Image, Image.Image] | None:
    """The catalog render next to a quarter at the scale the part's own dimension
    implies, so a coin-and-part photo is as consistent as a real one: the part's long
    axis is ``part_mm`` long, the coin ``COIN_MM`` across. None when the render has no
    measurable silhouette."""
    import numpy as np

    from mcmaster_vision.pipeline.measure import object_extent_px

    render = render.convert("RGB")
    ext = object_extent_px(render)
    if ext is None or part_mm <= 0:
        return None
    # crop the render to its content so the coin and the part fill the frame together,
    # as a phone photo taken close would
    arr = np.asarray(render)
    ink = (arr < 245).any(axis=-1)
    ys, xs = np.nonzero(ink)
    if len(xs):
        pad = 6
        render = render.crop(
            (
                max(0, xs.min() - pad),
                max(0, ys.min() - pad),
                min(render.size[0], xs.max() + pad),
                min(render.size[1], ys.max() + pad),
            )
        )
    mm_per_px = part_mm / ext[0]
    coin_px = COIN_MM / mm_per_px
    margin = 30
    total = coin_px + render.size[0] + 3 * margin
    f = min(2.0, 480 / total)
    coin_d = max(12, int(coin_px * f))
    part_img = render.resize((max(1, int(render.size[0] * f)), max(1, int(render.size[1] * f))))
    canvas = Image.new("RGB", (512, 512), (255, 255, 255))
    mask = Image.new("RGB", (512, 512), (0, 0, 0))
    rng = random.Random(seed)
    tint = (196 + rng.randint(-10, 10), 184 + rng.randint(-10, 10), 128 + rng.randint(-10, 10))
    cy = 256
    cx = margin + coin_d // 2
    box = (cx - coin_d // 2, cy - coin_d // 2, cx + coin_d // 2, cy + coin_d // 2)
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(box, fill=tint)
    draw.ellipse((box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3), outline=(160, 150, 100), width=2)
    ImageDraw.Draw(mask).ellipse(box, fill=(255, 255, 255))
    px = margin * 2 + coin_d
    canvas.paste(part_img, (px, cy - part_img.size[1] // 2))
    return canvas, mask


def coin_segment(mask: Image.Image) -> tuple[float, float, float, float] | None:
    """The segment a user would draw across the coin: the horizontal diameter of the
    bright disc in the (augmented) coin mask."""
    import numpy as np

    m = np.asarray(mask.convert("L")) > 128
    if m.sum() < 30:
        return None
    ys, xs = np.nonzero(m)
    cx, cy = float(xs.mean()), float(ys.mean())
    d = 2.0 * float(np.sqrt(m.sum() / np.pi))
    return (cx - d / 2, cy, cx + d / 2, cy)


def part_long_mm(part) -> float | None:
    """The part's stated long dimension (a length, else an OD / width) in mm."""
    from mcmaster_vision.pipeline.measure import parse_length_mm

    attrs = {k.lower(): str(v) for k, v in part.attributes.items()}
    for k in ("length", "overall_length", "od", "outside_diameter", "diameter", "width"):
        if k in attrs:
            mm = parse_length_mm(attrs[k])
            if mm:
                return mm
    return None


@router.post("/demo/try/{part_number}")
async def try_part(
    request: Request,
    part_number: str,
    seed: int = Query(0, ge=0),
    top_n: int = Query(5, ge=1, le=20),
    tta: str = Query("full", pattern="^(full|fast|none)$"),
    coin: bool = Query(False, description="Photograph the part next to a quarter and use it"),
    ident: Identifier = Depends(_ident),
) -> dict:
    """Identify a photo-style render of a catalog part; returns the result and whether it was right."""
    part = ident.store.get(part_number)
    if part is None or not part.image_paths:
        raise HTTPException(404, "unknown part")
    request.app.state.check_rate(request)  # same limits and CPU gate as /identify
    aug = PhotoAugmenter(AugmentConfig.evaluation(), seed=seed)
    src = Image.open(part.image_paths[seed % len(part.image_paths)])
    ref = None
    coin_mask = None
    if coin:
        mm = part_long_mm(part)
        staged = with_coin(src, mm, seed) if mm else None
        if staged is not None:
            src, coin_mask = staged
    if coin_mask is not None:
        # the coin mask rides through the same geometry: the segment is where the user
        # would draw it across the coin they put down
        img, moved = aug(src, out_size=512, mask=coin_mask)
        ref = coin_segment(moved)
    else:
        img = aug(src, out_size=512)
    measured_with_coin = ref is not None
    async with request.app.state.gate:
        if ref is not None:
            d = ref[2] - ref[0]
            res = await run_in_threadpool(
                ident.identify, img, top_n=top_n, tta=tta, mm_per_px=COIN_MM / d, reference=ref
            )
        else:
            res = await run_in_threadpool(ident.identify, img, top_n=top_n, tta=tta)
    # a sample is a real identification: log it and keep the render so a "This is it"
    # on it files feedback like any photo
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    await request.app.state.record(res, buf.getvalue())
    ranked = [c.part_number for c in res.candidates]
    return {
        "truth": part.part_number,
        "rank": ranked.index(part.part_number) + 1 if part.part_number in ranked else None,
        "query_image": f"/demo/query/{part.part_number}?seed={seed}",
        "coin": measured_with_coin,
        "result": res.model_dump(mode="json"),
    }


@router.get("/demo/sheet", response_class=HTMLResponse)
def sheet(
    n: int = Query(12, ge=1, le=48),
    seed: int | None = 1,
    key: bool = Query(True, description="Print part numbers under the images"),
    ident: Identifier = Depends(_ident),
) -> str:
    """Printable sheet of catalog images: print it, then photograph the paper with the phone."""
    parts = _sample_parts(ident, n, seed)
    cells = "".join(
        f"<figure><img src='/parts/{html.escape(p.part_number)}/image'><figcaption>{html.escape(p.part_number) if key else '&nbsp;'}<br><small>{html.escape(p.name)}</small></figcaption></figure>"
        for p in parts
    )
    from mcmaster_vision.api.pages import layout

    head = """<style>
 .sheet { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8mm; }
 figure { margin: 0; text-align: center; page-break-inside: avoid; }
 figure img { width: 100%; aspect-ratio: 1; object-fit: contain; border: 1px solid #ddd; background: #fff; }
 figcaption { font-size: 12px; margin-top: 4px; } small { color: #666; }
 @media print { .mc-header, .mc-footer, .noprint { display: none !important; } main.mc { padding: 0; max-width: none; } }
</style>"""
    nxt = f"/demo/sheet?n={n}&seed={(seed or 0) + 1}&key={'true' if key else 'false'}"
    flip = f"/demo/sheet?n={n}&seed={seed}&key={'false' if key else 'true'}"
    intro = (
        '<p class="noprint crumbs">Print this page (Ctrl/Cmd+P), lay it flat, and photograph one part at a time '
        f'with the phone app. <a href="{nxt}">another set</a> &middot; <a href="{flip}">{"hide" if key else "show"} part numbers</a></p>'
    )
    return layout(
        "Demo sheet",
        f'<h1 class="page">Demo sheet</h1>{intro}<div class="sheet">{cells}</div>',
        head=head,
    )


@router.get("/connect", response_class=HTMLResponse, include_in_schema=False)
def connect(request: Request) -> str:
    """Show the phone URL(s) and a QR code; open this on the laptop and scan it."""
    from mcmaster_vision.api.app import lan_urls

    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    urls = lan_urls(port, request.url.scheme)
    qr_svg = ""
    try:
        import qrcode
        import qrcode.image.svg

        img = qrcode.make(
            urls[0], image_factory=qrcode.image.svg.SvgPathImage, box_size=12, border=2
        )
        buf = io.BytesIO()
        img.save(buf)
        qr_svg = buf.getvalue().decode()
    except ImportError:
        qr_svg = "<p>(pip install qrcode for a QR code)</p>"
    links = "".join(f"<li><a href='{html.escape(u)}'>{html.escape(u)}</a></li>" for u in urls)
    from mcmaster_vision.api.pages import layout

    warn = (
        'Camera preview and "Add to Home Screen" work: this page is served over HTTPS.'
        if request.url.scheme == "https"
        else "Plain HTTP: the photo button works; for the live camera and app install start with <code>mcv serve --https</code>."
    )
    body = f'<h1 class="page">Open on your phone</h1><div style="text-align:center">{qr_svg}<ul style="list-style:none;padding:0;font-size:18px">{links}</ul><p class="notice" style="display:inline-block">{warn}</p></div>'
    return layout(
        "Connect a phone", body, head="<style>svg{width:min(70vw,360px);height:auto}</style>"
    )
