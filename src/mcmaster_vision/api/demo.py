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
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from mcmaster_vision.data.augment import AugmentConfig, PhotoAugmenter
from mcmaster_vision.pipeline.identify import Identifier

router = APIRouter(tags=["demo"])


def _ident(request: Request) -> Identifier:
    ident = getattr(request.app.state, "identifier", None)
    if ident is None:
        raise HTTPException(503, "index not built yet")
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
def query_image(part_number: str, seed: int = 0, ident: Identifier = Depends(_ident)):
    """A photo-style augmented render of the part (what the demo identifies)."""
    part = ident.store.get(part_number)
    if part is None or not part.image_paths:
        raise HTTPException(404, "unknown part")
    aug = PhotoAugmenter(AugmentConfig.evaluation(), seed=seed)
    img = aug(Image.open(part.image_paths[seed % len(part.image_paths)]), out_size=512)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/demo/try/{part_number}")
def try_part(
    part_number: str,
    seed: int = 0,
    top_n: int = Query(5, ge=1, le=20),
    tta: str = Query("full", pattern="^(full|fast|none)$"),
    ident: Identifier = Depends(_ident),
) -> dict:
    """Identify a photo-style render of a catalog part; returns the result and whether it was right."""
    part = ident.store.get(part_number)
    if part is None or not part.image_paths:
        raise HTTPException(404, "unknown part")
    aug = PhotoAugmenter(AugmentConfig.evaluation(), seed=seed)
    img = aug(Image.open(part.image_paths[seed % len(part.image_paths)]), out_size=512)
    res = ident.identify(img, top_n=top_n, tta=tta)
    ranked = [c.part_number for c in res.candidates]
    return {
        "truth": part.part_number,
        "rank": ranked.index(part.part_number) + 1 if part.part_number in ranked else None,
        "query_image": f"/demo/query/{part.part_number}?seed={seed}",
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
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>McMaster-Vision demo sheet</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 12mm; }}
 h1 {{ font-size: 16px; margin: 0 0 8px; }} p {{ color:#555; font-size: 12px; margin: 0 0 12px; }}
 .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10mm; }}
 figure {{ margin: 0; text-align: center; page-break-inside: avoid; }}
 img {{ width: 100%; aspect-ratio: 1; object-fit: contain; border: 1px solid #ddd; background: #fff; }}
 figcaption {{ font-size: 12px; margin-top: 4px; }} small {{ color: #666; }}
 @media print {{ .noprint {{ display: none; }} }}
</style></head><body>
<h1>McMaster-Vision demo sheet</h1>
<p class="noprint">Print this page (Ctrl/Cmd+P), lay it flat, and photograph one part at a time with the phone app. <a href="/demo/sheet?n={n}&seed={(seed or 0) + 1}&key={"true" if key else "false"}">another set</a> &middot; <a href="/demo/sheet?n={n}&seed={seed}&key={"false" if key else "true"}">{"hide" if key else "show"} part numbers</a></p>
<div class="grid">{cells}</div></body></html>"""


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
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Connect a phone</title>
<style>body{{font-family:system-ui,sans-serif;text-align:center;padding:24px}} svg{{width:min(70vw,360px);height:auto}} ul{{list-style:none;padding:0;font-size:18px}} .warn{{color:#b45309;font-size:14px}}</style></head>
<body><h1>Open on your phone</h1>{qr_svg}<ul>{links}</ul>
<p class="warn">{'Camera preview and "Add to Home Screen" work: this page is served over HTTPS.' if request.url.scheme == "https" else "Plain HTTP: the photo button works; for the live camera and app install, start with <code>mcv serve --https</code>."}</p>
</body></html>"""
