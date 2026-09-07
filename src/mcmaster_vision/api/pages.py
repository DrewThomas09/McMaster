"""Server-rendered pages sharing one layout and the theme: browse, part detail, dashboard.

Plain HTML from Python (no template engine) keeps the dependency list short and
the pages cacheable; the identify app itself lives in ``static/index.html``.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

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
    # through the app's accessor so a rebuilt index (mcv learn / retrain) is picked up
    ident = request.app.state.get_identifier()
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
    fits_html = _fits_with(part)
    # inside <script> entities are not decoded, so JSON (with "</" broken up) not html.escape
    pn_js = json.dumps(part.part_number).replace("</", "<\\/")
    demo = ""
    if request.app.state.settings.demo_mode:
        demo = f'<a class="ghost" href="/?try={pn}">Identify a photo-style render</a> '
    body = f"""<div class="crumbs">{crumbs or "&nbsp;"}</div>
<h1 class="page">{pn} <span style="font-weight:400;color:var(--muted)">{e(part.name)}</span></h1>
<p>{demo}<button class="btn small" id="addcart" type="button">&#128722; Add to cart</button> <a class="ghost" href="/#cart" id="opencart" hidden>view cart</a> <a class="ghost" href="https://www.mcmaster.com/{quote(part.part_number, safe="")}/" target="_blank" rel="noopener">Open on mcmaster.com ↗</a> <button class="ghost" onclick="navigator.clipboard&&navigator.clipboard.writeText({e(json.dumps(part.part_number))})">Copy part number</button></p>
<script>
(() => {{
  const PN = {pn_js};
  const cid = () => {{ let id = null; try {{ id = localStorage.getItem('mcv.client'); }} catch (_) {{}}
    if (!id) {{ id = 'c-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36); try {{ localStorage.setItem('mcv.client', id); }} catch (_) {{}} }} return id; }};
  const b = document.getElementById('addcart');
  b.onclick = async () => {{
    b.disabled = true;
    try {{
      const r = await fetch('/cart', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ client_id: cid(), part_number: PN, quantity: 1 }}) }});
      if (!r.ok) throw new Error(r.status);
      const cart = await r.json();
      const n = cart.reduce((t, it) => t + it.quantity, 0);
      b.textContent = '\u2713 In cart (' + n + ' item' + (n === 1 ? '' : 's') + ')';
      document.getElementById('opencart').hidden = false;
    }} catch (err) {{ b.textContent = 'Could not add (' + err.message + ')'; }}
    b.disabled = false;
  }};
}})();
</script>
<div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(140px,1fr))">{gallery}</div>
<h2 class="page">Specifications</h2><div class="card" style="padding:8px 12px"><table class="spec">{specs or '<tr><td class="msg" colspan="2">No attributes recorded.</td></tr>'}</table>{f'<p class="crumbs">{e(part.description)}</p>' if part.description else ""}</div>
{fits_html}{fam_html}"""
    return layout(part.part_number, body, active="/browse")


def _fits_with(part) -> str:
    """Pipe sizing facts for a threaded fitting: real OD/ID of its nominal size, thread
    pitch, and which thread types mate with it (from the catalog's measuring pages)."""
    from mcmaster_vision.pipeline.pipe import (
        THREADS_PER_INCH,
        compatible_threads,
        normalise_pipe_size,
        pipe_id_mm,
        pipe_od_mm,
    )

    attrs = {k.lower().replace(" ", "_"): str(v) for k, v in part.attributes.items()}
    size_raw = next(
        (attrs[k] for k in ("pipe_size", "pipe", "nominal_pipe_size") if k in attrs), None
    )
    key = normalise_pipe_size(size_raw) if size_raw else None
    text = " ".join([part.name, part.description, *attrs.values()]).upper()
    thread = next(
        (t for t in ("NPTF", "NPT", "BSPT", "BSPP", "NPSM", "NPSH", "NPSL") if t in text), None
    )
    if not key and not thread:
        return ""
    rows = []
    if key:
        od, pid = pipe_od_mm(key), pipe_id_mm(key)
        if od:
            rows.append(("male thread / pipe OD", f"{od / 25.4:.3f} in ({od:.1f} mm)"))
        if pid:
            rows.append(("female thread / pipe ID", f"{pid / 25.4:.3f} in ({pid:.1f} mm)"))
        npt, bsp = THREADS_PER_INCH.get(key, (None, None))
        if npt or bsp:
            rows.append(
                (
                    "threads per inch",
                    " · ".join(f"{n} {f}" for n, f in ((npt, "NPT"), (bsp, "BSP")) if n),
                )
            )
    if thread:
        for gender in ("male", "female"):
            mates = compatible_threads(thread, gender)
            if mates:
                rows.append(
                    (
                        f"{thread} {gender} fits female"
                        if gender == "male"
                        else f"{thread} female fits male",
                        ", ".join(mates),
                    )
                )
    if not rows:
        return ""
    trs = "".join(f"<tr><td>{e(k)}</td><td>{e(v)}</td></tr>" for k, v in rows)
    return (
        f'<h2 class="page">Pipe size {e(key) if key else ""} in real dimensions</h2>'
        f'<div class="card" style="padding:8px 12px"><table class="spec">{trs}</table>'
        '<p class="crumbs">Pipe size is a nominal designation: measure the OD across male threads or the ID of female ones.</p></div>'
    )


def _learning_loop_html(request: Request) -> str:
    """The purchase funnel, what customers bought vs what was predicted, the issues the
    analytics found, and how much the model has learned from it."""
    from mcmaster_vision.pipeline.events import analytics, enrich_confusions, issues
    from mcmaster_vision.pipeline.learn import learning_state

    st = request.app.state
    a = analytics(st.events, st.feedback.stats())
    try:
        enrich_confusions(a, st.get_identifier().store)
    except HTTPException:  # nothing built yet
        pass
    found = issues(a)
    learn = learning_state(st.settings, st.feedback)
    w = a["window"]
    f = a["funnel"]

    def pc(x):
        return f"{x:.0%}" if x is not None else "—"

    issue_rows = (
        "".join(
            f'<tr><td><span class="tier {"unknown" if i["severity"] == "high" else "candidate" if i["severity"] == "medium" else "likely"}">{e(i["severity"])}</span></td>'
            f'<td>{e(i["what"])}</td><td class="crumbs">{e(i["do"])}</td></tr>'
            for i in found
        )
        or '<tr><td class="msg" colspan="3">No problems found in this window.</td></tr>'
    )
    from urllib.parse import quote

    def part_link(pn: str) -> str:
        return f'<a href="/part/{quote(pn)}">{e(pn)}</a>' if pn and pn != "(none)" else e(pn)

    conf_rows = "".join(
        f"<tr><td>{part_link(c['predicted'])}</td><td>{part_link(c['bought'])}</td>"
        f"<td>{c['times']}</td></tr>"
        for c in a["confusions"][:6]
    )
    tier_rows = "".join(
        f'<tr><td><span class="tier {e(t)}">{e(t)}</span></td><td>{v["bought"]}</td><td>{pc(v["precision"])}</td></tr>'
        for t, v in a["tier_precision_bought"].items()
    )
    conf = a["confidence"]
    due = learn["retrain_due"]
    day_rows = "".join(
        f"<tr><td>{e(d['day'])}</td><td>{d['identify']}</td><td>{d['bought']}</td>"
        f"<td>{pc(d['bought_top1_rate'])}</td><td>{d['learns']}{' + ' + str(d['retrains']) + ' retrain' if d['retrains'] else ''}</td></tr>"
        for d in a.get("daily", [])[-14:]
    )
    daily_html = (
        f'<div class="card" style="padding:8px 12px;margin-top:10px"><b>By day</b><table class="spec"><tr><th>day</th><th>identified</th><th>bought</th><th>bought top-1</th><th>learns</th></tr>{day_rows}</table></div>'
        if day_rows
        else ""
    )
    return f"""<h2 class="page">Learning loop (photo &rarr; cart &rarr; checkout &rarr; model)</h2>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0">
  <div class="card stat"><b>{w["identify"]}</b><span>identifications</span></div>
  <div class="card stat"><b>{pc(f["identify_to_cart"])}</b><span>&rarr; added to cart</span></div>
  <div class="card stat"><b>{pc(f["cart_to_checkout"])}</b><span>&rarr; checked out</span></div>
  <div class="card stat"><b>{w["items_bought"]}</b><span>parts bought</span></div>
  <div class="card stat"><b>{pc(a["bought_top1_rate"])}</b><span>bought the top answer</span></div>
  <div class="card stat"><b>{learn["new_confirmations"]}</b><span>new confirmations since last learn ({learn["new_purchases"]} purchases)</span></div>
</div>
<div class="card" style="padding:8px 12px"><table class="spec"><tr><th>severity</th><th>what the data says</th><th>what to do</th></tr>{issue_rows}</table></div>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">
  <div class="card" style="padding:8px 12px;flex:1;min-width:260px"><b>Predicted vs bought</b><table class="spec"><tr><th>predicted</th><th>bought</th><th>times</th></tr>{conf_rows or '<tr><td class="msg" colspan="3">No wrong purchases recorded.</td></tr>'}</table></div>
  <div class="card" style="padding:8px 12px;flex:1;min-width:260px"><b>When customers bought, was the tier right?</b><table class="spec"><tr><th>tier</th><th>bought</th><th>top-1 right</th></tr>{tier_rows or '<tr><td class="msg" colspan="3">Nothing bought yet.</td></tr>'}</table>
  <div class="crumbs">confidence when right {e(conf["when_right"] if conf["when_right"] is not None else "—")} · when wrong {e(conf["when_wrong"] if conf["when_wrong"] is not None else "—")} · p95 latency {e(a["latency_ms"]["p95"] or "—")} ms · errors {w["errors"]}</div></div>
</div>
{daily_html}
<p class="crumbs" id="learnline">last learned {e((learn["learned_at"] or "never")[:19])} ({e(learn["learned_photos"] or 0)} photos) · {learn["since_retrain"]}/{learn["retrain_threshold"]} towards a retrain{' · <span class="tier candidate">retrain due: run mcv learn</span>' if due else ""} ·
<button class="btn small" type="button" id="learnbtn">Learn now</button> <code>mcv learn</code> · <code>mcv simulate --learn</code> · <a href="/analytics">/analytics</a> · <a href="/orders">/orders</a></p>
<script>
document.getElementById('learnbtn').onclick = async () => {{
  const b = document.getElementById('learnbtn'); b.disabled = true; b.textContent = 'Learning…';
  try {{
    const tok = localStorage.getItem('mcv_token') || '';
    const r = await fetch('/admin/learn', {{method: 'POST', headers: tok ? {{'X-API-Token': tok}} : {{}}}});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.status);
    document.getElementById('learnline').firstChild.textContent = j.action === 'none' ? 'nothing new to learn ' : `learned ${{j.photos}} photos of ${{j.parts}} parts (${{j.how}}, ${{j.seconds}} s) `;
  }} catch (err) {{ alert('Learn failed: ' + err.message + (String(err.message).includes('Token') ? ' — set localStorage.mcv_token' : '')); }}
  b.disabled = false; b.textContent = 'Learn now';
}};
</script>"""


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
    recent = request.app.state.requests.recent(15)
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
    ev = (
        (st.get("manifest") or {}).get("retrain_eval")
        or (st.get("manifest") or {}).get("evaluation")
        or {}
    )
    eval_html = ""
    if ev.get("queries"):
        r = ev.get("recall_at") or {}
        cat_rows = "".join(
            f"<tr><td>{e(c)}</td><td>{v['recall_1']:.0%}</td><td>{v['recall_5']:.0%}</td><td>{v['queries']}</td></tr>"
            for c, v in list((ev.get("by_category") or {}).items())[:8]
        )
        eval_html = f"""<h2 class="page">Measured accuracy (last evaluation, {e(ev.get("queries"))} queries)</h2>
<div class="card" style="padding:8px 12px"><table class="spec">
<tr><td>Recall@1</td><td>{e(f"{float(r.get('1', r.get(1, 0))):.0%}")}</td></tr>
<tr><td>Recall@5</td><td>{e(f"{float(r.get('5', r.get(5, 0))):.0%}")}</td></tr>
<tr><td>Family Recall@1</td><td>{e(f"{float((ev.get('family_recall_at') or {}).get('1', 0)):.0%}")}</td></tr>
<tr><td>MRR</td><td>{e(ev.get("mrr"))}</td></tr>
</table>{f'<table class="spec" style="margin-top:8px"><tr><th>weakest categories</th><th>R@1</th><th>R@5</th><th>queries</th></tr>{cat_rows}</table>' if cat_rows else ""}</div>"""
    sto = st.get("storage") or {}
    storage_rows = ""
    for name, c in (sto.get("components") or {}).items():
        nbytes = c.get("bytes", 0)
        size = f"{nbytes / 1e6:.1f} MB" if nbytes >= 1e6 else f"{nbytes / 1e3:.0f} KB"
        storage_rows += (
            f"<tr><td>{e(name)}</td><td>{size}</td>"
            f"<td>{e((c.get('updated_at') or '—')[:19])}</td></tr>"
        )
    last = sto.get("last_backup")
    backup_line = (
        f"last backup {e(last['created_at'][:19])} ({last['bytes'] / 1e6:.1f} MB)"
        if last
        else "no backup yet"
    )
    if sto.get("backup_stale"):
        backup_line += ' <span class="tier candidate">changed since</span>'
    loop_html = _learning_loop_html(request)
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
{loop_html}
{eval_html}
<h2 class="page">Storage</h2><div class="card" style="padding:8px 12px"><table class="spec"><tr><th>state</th><th>size</th><th>updated</th></tr>{storage_rows}</table>
<p class="crumbs" id="backupline">{backup_line} · <button class="btn small" type="button" id="backupbtn">Back up now</button> <code>mcv backup</code> / <code>mcv restore</code></p></div>
<script>
document.getElementById('backupbtn').onclick = async () => {{
  const b = document.getElementById('backupbtn'); b.disabled = true; b.textContent = 'Backing up…';
  try {{
    const tok = localStorage.getItem('mcv_token') || '';
    const r = await fetch('/admin/backup', {{method: 'POST', headers: tok ? {{'X-API-Token': tok}} : {{}}}});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.status);
    const line = document.getElementById('backupline');
    line.firstChild.textContent = 'backed up to ' + j.path + ' (' + (j.bytes / 1e6).toFixed(1) + ' MB) ';
    const badge = line.querySelector('.tier'); if (badge) badge.remove();
  }} catch (err) {{ alert('Backup failed: ' + err.message + (String(err.message).includes('Token') ? ' — set localStorage.mcv_token' : '')); }}
  b.disabled = false; b.textContent = 'Back up now';
}};
</script>
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
