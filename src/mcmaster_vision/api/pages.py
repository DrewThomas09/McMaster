"""Server-rendered pages sharing one layout and the theme: browse, part detail, dashboard.

Plain HTML from Python (no template engine) keeps the dependency list short and
the pages cacheable; the identify app itself lives in ``static/index.html``.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from mcmaster_vision import __version__
from mcmaster_vision.pipeline.identify import Identifier

router = APIRouter(include_in_schema=False)
NAV = [("/", "Identify"), ("/browse", "Browse"), ("/dashboard", "Dashboard")]


def e(x) -> str:
    return html.escape(str(x if x is not None else ""))


_ABBREV = {"od": "OD", "id": "ID", "pipe_size": "pipe size (NPT)"}


def label(key: str) -> str:
    """Attribute key -> human label; keeps OD / ID upper-case."""
    return e(_ABBREV.get(key, key.replace("_", " ")))


def layout(title: str, body: str, *, active: str = "", head: str = "", status: str = "") -> str:
    nav = "".join(
        f'<a href="{href}" class="{"on" if href == active else ""}">{label}</a>'
        for href, label in NAV
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><meta name="theme-color" content="#0b5d3b">
<link rel="stylesheet" href="/static/theme.css"><link rel="icon" href="/static/icon.svg" type="image/svg+xml"><link rel="manifest" href="/static/manifest.webmanifest">
<title>{e(title)} · McMaster-Vision</title>{head}</head><body>
<header class="mc-header"><a class="brand" href="/">McMaster-Vision<small>unofficial part identifier</small></a>{f'<span class="pill ok">{e(status)}</span>' if status else ""}<nav>{nav}</nav></header>
<main class="mc">{body}</main>
<footer class="mc-footer">McMaster-Vision v{__version__} · not affiliated with McMaster-Carr; part numbers link to mcmaster.com for reference.</footer>
</body></html>"""


def _ident(request: Request) -> Identifier:
    ident = getattr(request.app.state, "identifier", None)
    if ident is None:
        raise HTTPException(503, "index not built yet")
    return ident


def _not_ready(title: str) -> str:
    return layout(
        title,
        '<div class="notice">Nothing is built yet. Run <code>mcv bootstrap &lt;your images&gt;</code> or <code>mcv up</code> for a demo catalog, then reload.</div>',
    )


def _tile(part) -> str:
    pn = e(part.part_number)
    return f'<a class="card tile" href="/part/{pn}"><img src="/parts/{pn}/thumb?size=200" loading="lazy" alt=""><div class="pn">{pn}</div><div class="meta">{e(part.name)}</div></a>'


@router.get("/browse", response_class=HTMLResponse)
def browse(request: Request, category: str = Query(""), page: int = Query(1, ge=1)) -> str:
    ident = getattr(request.app.state, "identifier", None)
    if ident is None:
        return _not_ready("Browse")
    store = ident.store
    prefix = [c.strip() for c in category.split(">") if c.strip()]
    tax = store.taxonomy()
    crumbs = '<a href="/browse">All categories</a>'
    for i, c in enumerate(prefix):
        crumbs += f' › <a href="/browse?category={e(" > ".join(prefix[: i + 1]))}">{e(c)}</a>'
    children = tax.children(prefix)
    chips = ""
    for c in children:
        href = e(" > ".join([*prefix, c]))
        chips += f'<a class="chip" href="/browse?category={href}">{e(c)} <span style="color:var(--muted)">{tax.count([*prefix, c])}</span></a>'
    per = 48
    parts = (
        store.by_category(prefix, limit=per, offset=(page - 1) * per)
        if (prefix or page > 1)
        else []
    )
    total = tax.count(prefix) if prefix else store.count()
    tiles = "".join(_tile(p) for p in parts)
    pager = ""
    if prefix:
        pager = f'<p class="crumbs">{(page - 1) * per + 1}-{min(page * per, total)} of {total}'
        if page > 1:
            pager += f' · <a href="/browse?category={e(category)}&page={page - 1}">previous</a>'
        if page * per < total:
            pager += f' · <a href="/browse?category={e(category)}&page={page + 1}">next</a>'
        pager += "</p>"
    body = f'''<h1 class="page">Browse the catalog</h1><div class="crumbs">{crumbs}</div>
<form action="/browse" method="get" class="chips" style="margin-bottom:12px"><input type="hidden" name="category" value="{e(category)}"><input name="q" placeholder="search within" style="display:none"></form>
{f'<div class="chips" style="margin-bottom:14px">{chips}</div>' if chips else ""}
{f'<div class="grid">{tiles}</div>{pager}' if parts else ('<p class="crumbs">Pick a category above.</p>' if not prefix else '<p class="crumbs">No parts here.</p>')}'''
    return layout("Browse", body, active="/browse", status=f"{store.count()} parts")


@router.get("/part/{part_number}", response_class=HTMLResponse)
def part_page(part_number: str, request: Request) -> str:
    ident = getattr(request.app.state, "identifier", None)
    if ident is None:
        return _not_ready("Part")
    part = ident.store.get(part_number)
    if part is None:
        raise HTTPException(404, "unknown part number")
    pn = e(part.part_number)
    n_img = sum(1 for p in part.image_paths if Path(p).exists())
    gallery = (
        "".join(
            f'<a href="/parts/{pn}/image?i={i}" target="_blank"><img src="/parts/{pn}/thumb?i={i}&size=256" alt="" style="width:100%;aspect-ratio:1;object-fit:contain;background:#fff;border:1px solid var(--rule);border-radius:6px"></a>'
            for i in range(n_img)
        )
        or '<p class="crumbs">No images.</p>'
    )
    specs = "".join(
        f"<tr><td>{e(k.replace('_', ' '))}</td><td>{e(v)}</td></tr>"
        for k, v in part.attributes.items()
    )
    crumb_links = []
    for i, c in enumerate(part.category_path):
        href = e(" > ".join(part.category_path[: i + 1]))
        crumb_links.append(f'<a href="/browse?category={href}">{e(c)}</a>')
    crumbs = " › ".join(crumb_links)
    family = (
        [p for p in ident.store.family(part.family_id) if p.part_number != part.part_number]
        if part.family_id
        else []
    )
    fam_html = ""
    if family:
        keys = sorted({k for p in [part, *family] for k in p.attributes})
        differing = [
            k for k in keys if len({str(p.attributes.get(k, "")) for p in [part, *family]}) > 1
        ]
        rows = ""
        for p in family[:40]:
            cells = "".join(f"<td>{e(p.attributes.get(k, ''))}</td>" for k in differing)
            rows += f'<tr><td><a href="/part/{e(p.part_number)}">{e(p.part_number)}</a></td>{cells}</tr>'
        heads = "".join(f"<th>{label(k)}</th>" for k in differing)
        fam_html = (
            f'<h2 class="page">Look-alike SKUs in this family ({len(family)})</h2>'
            f'<div class="card" style="padding:8px 12px;overflow-x:auto"><table class="spec"><tr><th>part</th>{heads}</tr>{rows}</table></div>'
        )
    demo = ""
    if request.app.state.settings.demo_mode:
        demo = f'<a class="ghost" href="/?try={pn}">Identify a photo-style render</a> '
    body = f"""<div class="crumbs">{crumbs or "&nbsp;"}</div>
<h1 class="page">{pn} <span style="font-weight:400;color:var(--muted)">{e(part.name)}</span></h1>
<p>{demo}<a class="ghost" href="https://www.mcmaster.com/{pn}/" target="_blank" rel="noopener">Open on mcmaster.com ↗</a> <button class="ghost" onclick="navigator.clipboard&&navigator.clipboard.writeText('{pn}')">Copy part number</button></p>
<div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(140px,1fr))">{gallery}</div>
<h2 class="page">Specifications</h2><div class="card" style="padding:8px 12px"><table class="spec">{specs or '<tr><td class="msg" colspan="2">No attributes recorded.</td></tr>'}</table>{f'<p class="crumbs">{e(part.description)}</p>' if part.description else ""}</div>
{fam_html}"""
    return layout(part.part_number, body, active="/browse")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request) -> str:
    from mcmaster_vision.pipeline.manifest import status as _status

    settings = request.app.state.settings
    st = _status(settings)
    metrics = request.app.state.requests.summary(request.app.state.feedback)
    idx = st.get("index") or {}
    cat = st.get("catalog") or {}
    fb = metrics.get("feedback") or {}
    tiers = metrics.get("tiers") or {}
    total_t = sum(tiers.values()) or 1
    tier_rows = ""
    for t, n in sorted(tiers.items(), key=lambda kv: -kv[1]):
        width = f"{100 * n / total_t:.0f}%"
        tier_rows += (
            f'<tr><td><span class="tier {e(t)}">{e(t)}</span></td><td>{n}</td>'
            f'<td style="width:50%"><div class="bar"><i style="width:{width}"></i></div></td></tr>'
        )
    recent = list(request.app.state.requests._recent)[-15:][::-1]
    recent_rows = ""
    for r in recent:
        when = e(str(r.get("created_at", ""))[11:19])
        best = r.get("best")
        best_html = f'<a href="/part/{e(best)}">{e(best)}</a>' if best else "—"
        conf = e(round((r.get("confidence") or 0) * 100))
        recent_rows += (
            f'<tr><td>{when}</td><td><span class="tier {e(r["tier"])}">{e(r["tier"])}</span></td>'
            f"<td>{best_html}</td><td>{conf}%</td><td>{e(r.get('latency_ms'))} ms</td></tr>"
        )
    stale = st.get("index_stale")
    body = f"""<h1 class="page">Dashboard</h1>
{'<div class="notice">The catalog changed after the index was built: run <code>mcv build-index --only-new</code> then <code>POST /admin/reload</code>.</div>' if stale else ""}
<div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0">
  <div class="card stat"><b>{e(cat.get("parts", 0))}</b><span>parts in catalog</span></div>
  <div class="card stat"><b>{e(idx.get("vectors", 0))}</b><span>index rows ({e(idx.get("backend", "-"))})</span></div>
  <div class="card stat"><b>{e(metrics.get("requests_total", 0))}</b><span>identifications</span></div>
  <div class="card stat"><b>{e(metrics.get("latency_ms", {}).get("p50") or "—")}</b><span>ms p50 (p95 {e(metrics.get("latency_ms", {}).get("p95") or "—")})</span></div>
  <div class="card stat"><b>{e(fb.get("confirmed", 0))}</b><span>confirmed photos</span></div>
  <div class="card stat"><b>{e(f"{metrics['confirmed_top1_rate']:.0%}" if metrics.get("confirmed_top1_rate") is not None else "—")}</b><span>confirmed top-1 rate</span></div>
</div>
<h2 class="page">Answer tiers (recent window)</h2><div class="card" style="padding:8px 12px"><table class="spec">{tier_rows or "<tr><td>No requests yet.</td></tr>"}</table></div>
<h2 class="page">Recent identifications</h2><div class="card" style="padding:8px 12px;overflow-x:auto"><table class="spec"><tr><th>time</th><th>tier</th><th>best</th><th>conf.</th><th>latency</th></tr>{recent_rows or '<tr><td class="msg" colspan="5">None yet.</td></tr>'}</table></div>
<h2 class="page">Build</h2><div class="card" style="padding:8px 12px"><table class="spec">
<tr><td>model</td><td>{e(idx.get("backbone") or st.get("settings", {}).get("backbone"))}</td></tr>
<tr><td>index built</td><td>{e(idx.get("built_at", "—"))}</td></tr>
<tr><td>catalog updated</td><td>{e(cat.get("updated_at", "—"))}</td></tr>
<tr><td>calibration</td><td>{e(st.get("calibration") or "not fitted")}</td></tr>
<tr><td>manifest</td><td>{e(st.get("manifest", {}).get("updated_at", "—"))}</td></tr>
<tr><td>server time</td><td>{e(datetime.now(timezone.utc).isoformat(timespec="seconds"))}</td></tr>
</table></div>
<p class="crumbs"><a href="/status">/status</a> · <a href="/metrics">/metrics</a> · <a href="/docs">API docs</a> · <a href="/connect">open on a phone</a></p>"""
    return layout(
        "Dashboard", body, active="/dashboard", status="ready" if st.get("ready") else "not ready"
    )
