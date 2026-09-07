"""HTTP API.

POST /identify           multipart image -> IdentificationResult
GET  /parts/{pn}         catalog entry
GET  /parts/{pn}/image   first catalog image
GET  /search?q=          keyword search over the catalog
GET  /health, /stats
GET  /                   upload UI
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mcmaster_vision import __version__
from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.feedback import FeedbackStore, RecentPhotos
from mcmaster_vision.pipeline.identify import Identifier, load_identifier
from mcmaster_vision.pipeline.requestlog import RequestLog
from mcmaster_vision.schemas import Feedback, IdentificationResult, IndexStats, Part

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
THUMB_SIZES = (96, 200, 400)


def create_app(settings: Settings | None = None, identifier: Identifier | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="McMaster-Vision", version=__version__, description=__doc__)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if settings.cors_origins:
        origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
        )
    from mcmaster_vision.api.commerce import Carts
    from mcmaster_vision.api.commerce import router as commerce_router
    from mcmaster_vision.api.demo import router as demo_router
    from mcmaster_vision.api.pages import router as pages_router
    from mcmaster_vision.pipeline.events import EventLog, analytics, issues

    app.include_router(demo_router)
    app.include_router(pages_router)
    app.include_router(commerce_router)
    app.state.events = EventLog(settings.data_dir / "logs" / "events.jsonl")
    app.state.carts = Carts(settings.data_dir / "logs" / "orders.jsonl")
    app.state.settings = settings
    app.state.identifier = identifier
    app.state.feedback = FeedbackStore(settings.queries_dir)
    app.state.requests = RequestLog(settings.data_dir / "logs" / "requests.jsonl")
    # query photos wait on disk (not in memory) so a confirmation still lands after a
    # restart, a redeploy, or on a different worker process
    app.state.recent = RecentPhotos(settings.data_dir / "cache" / "recent")
    app.state.started_at = time.time()
    app.state.index_mtime: float | None = None
    app.state.last_index_check = 0.0
    index_meta = settings.index_path / "meta.json"

    from mcmaster_vision.api.ratelimit import RateLimiter

    limiter = RateLimiter(settings.rate_limit_per_minute)
    # CPU-bound identifications are serialised to the core count: more threads than cores
    # only adds contention, and a phone's live preview must not starve a real photo. The
    # gate is awaited *before* taking a threadpool worker, so queued identifications never
    # exhaust the pool and stall health checks, thumbnails and pages.
    app.state.gate = asyncio.Semaphore(settings.max_concurrency or max(1, os.cpu_count() or 1))
    backup_lock = threading.Lock()

    def check_rate(request: Request, cost: int = 1) -> None:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(RateLimiter.bucket(client), cost):
            raise HTTPException(429, "rate limit exceeded; try again in a minute")

    app.state.check_rate = check_rate
    app.state.get_identifier = lambda: get_identifier()
    app.state.check_admin = lambda request: check_admin(request)

    async def record(result: IdentificationResult, photo: bytes | None) -> None:
        """Append to the request log and keep the photo, off the event loop."""

        def _do() -> None:
            app.state.requests.log(result)
            app.state.events.log(
                "identify",
                request_id=result.request_id,
                tier=result.tier.value,
                best=result.best.part_number if result.best else None,
                confidence=result.best.confidence if result.best else None,
                candidates=[c.part_number for c in result.candidates],
                latency_ms=result.timings_ms.get("total"),
                photos=result.photos,
                measured=bool(result.measured),
                family=result.family.family_id if result.family else None,
            )
            if photo is not None:
                app.state.recent.put(result.request_id, photo)

        await run_in_threadpool(_do)

    app.state.record = record

    def _load() -> Identifier:
        # sample the mtime *before* reading: a swap that lands during the read is then
        # noticed on the next poll instead of being recorded against the old data
        mtime = index_meta.stat().st_mtime if index_meta.exists() else None
        try:
            ident = load_identifier(settings)
        except Exception:
            app.state.failed_mtime = mtime  # do not retry this exact index every 15 s
            raise
        app.state.index_mtime = mtime
        app.state.failed_mtime = None
        app.state.last_index_check = time.time()
        return ident

    def _maybe_reload() -> None:
        """Pick up a rebuilt index (``mcv build-index`` / ``mcv retrain`` / ``mcv restore``)
        without a restart or a token: the on-disk meta.json is polled at most every 15 s."""
        now = time.time()
        if now - app.state.last_index_check < 15:
            return
        app.state.last_index_check = now
        try:
            mtime = index_meta.stat().st_mtime if index_meta.exists() else None
        except OSError:
            return
        # any change counts (a restore puts back files with *older* mtimes); an index that
        # already failed to load is left alone until it changes again
        if (
            mtime
            and app.state.index_mtime
            and mtime != app.state.index_mtime
            and mtime != getattr(app.state, "failed_mtime", None)
        ):
            try:
                app.state.identifier = _load()
                log.info("index changed on disk; reloaded")
            except Exception:  # keep serving the old index
                log.exception("auto-reload failed; still serving the previous index")

    def get_identifier() -> Identifier:
        if app.state.identifier is None:
            try:
                app.state.identifier = _load()
            except FileNotFoundError as e:
                raise HTTPException(503, f"index not built yet: {e}") from e
            except (RuntimeError, ValueError) as e:  # index / backbone mismatch, corrupt index
                raise HTTPException(503, str(e)) from e
        elif settings.auto_reload:
            _maybe_reload()
        return app.state.identifier

    @app.on_event("startup")
    def warm_up() -> None:
        """Load catalog + index + backbone at boot so the first photo is fast; if
        nothing is built yet the endpoints report 503 until `mcv bootstrap` runs."""
        if app.state.identifier is None and settings.warm_up:
            try:
                app.state.identifier = _load()
                log.info("identifier ready: %s", app.state.identifier.index.stats().model_dump())
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                log.warning("not ready: %s", e)

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/sw.js", include_in_schema=False)
    def service_worker():
        return FileResponse(
            STATIC / "sw.js",
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/"},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        ident = app.state.identifier
        return {
            "status": "ok",
            "version": __version__,
            "ready": ident is not None,
            "model": ident.embedder.version if ident else None,
            "index_built_at": ident.index.meta.get("built_at") if ident else None,
            "parts": len(set(ident.index.ids)) if ident else None,
            "demo_mode": settings.demo_mode,
            "secure": False,  # the client checks window.isSecureContext itself
            "uptime_s": round(time.time() - app.state.started_at, 1),
            "requests_total": app.state.requests.total,
            "index_backbone_mismatch": bool(
                ident
                and ident.index.meta.get("backbone")
                and ident.index.meta.get("backbone") != ident.embedder.version
            ),
        }

    @app.get("/status")
    def status_view() -> dict:
        from mcmaster_vision.pipeline.manifest import status as _status

        out = _status(settings)
        out["loaded"] = app.state.identifier is not None
        return out

    def check_admin(request: Request) -> None:
        token = settings.api_token
        if token and not secrets.compare_digest(request.headers.get("x-api-token", ""), token):
            raise HTTPException(401, "bad or missing X-API-Token")

    @app.post("/admin/reload")
    def reload(request: Request) -> dict:
        """Re-open the catalog and index after `mcv build-index` without restarting.
        Protected by MCV_API_TOKEN (header X-API-Token) when that is set."""
        check_admin(request)
        try:
            app.state.identifier = _load()
        except FileNotFoundError as e:
            raise HTTPException(503, f"index not built yet: {e}") from e
        except (RuntimeError, ValueError) as e:  # keep serving the previous index
            raise HTTPException(409, str(e)) from e
        return {
            "reloaded": True,
            "index": app.state.identifier.index.stats().model_dump(mode="json"),
        }

    @app.post("/admin/backup")
    async def backup(request: Request) -> dict:
        """Bundle catalog, index, calibration, confirmed photos, logs and manifest into
        ``data/backups/mcv-<timestamp>.tar.gz`` (same as ``mcv backup``)."""
        check_admin(request)
        from mcmaster_vision.pipeline.backup import create_backup, read_inventory

        if not backup_lock.acquire(blocking=False):
            raise HTTPException(409, "a backup is already running")
        try:
            path = await run_in_threadpool(create_backup, settings)
        finally:
            backup_lock.release()
        inv = read_inventory(path)
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "components": sorted(inv["components"]),
            "created_at": inv["created_at"],
        }

    @app.get("/admin/backups")
    def list_backups(request: Request) -> list[dict]:
        check_admin(request)
        root = settings.data_dir / "backups"
        if not root.exists():
            return []
        out = []
        for f in sorted(root.glob("*.tar.gz"), reverse=True):
            out.append({"path": str(f), "bytes": f.stat().st_size, "mtime": f.stat().st_mtime})
        return out

    @app.get("/stats", response_model=IndexStats)
    def stats(ident: Identifier = Depends(get_identifier)) -> IndexStats:
        return ident.index.stats()

    @app.post("/identify", response_model=IdentificationResult)
    async def identify(
        request: Request,
        file: UploadFile | None = File(None, description="One photo"),
        files: list[UploadFile] | None = File(None, description="Several photos of the same part"),
        top_n: int = Query(5, ge=1, le=50),
        use_llm: bool | None = Query(
            None, description="Override the configured vision-LLM reranker"
        ),
        constraints: str | None = Query(
            None, description='JSON object of known attributes, e.g. {"thread_size": "M6"}'
        ),
        tta: str = Query(
            "full",
            pattern="^(full|fast|none)$",
            description="Test-time augmentation: full (8 views), fast (2), none",
        ),
        log: bool = Query(
            True,
            description="false for live-preview frames: not written to the request log or metrics",
        ),
        mm_per_px: float | None = Query(
            None,
            gt=0,
            description="Scale of the first photo (mm per pixel of the uploaded image), "
            "from a coin / card / ruler marked in the app; enables size matching",
        ),
        ref: str | None = Query(
            None,
            pattern=r"^-?\d+(\.\d+)?(,-?\d+(\.\d+)?){3}$",
            description="x1,y1,x2,y2 (uploaded pixels) of the line drawn across the reference "
            "object, so the coin / card is not measured instead of the part",
        ),
        ident: Identifier = Depends(get_identifier),
    ) -> IdentificationResult:
        reference = tuple(float(v) for v in ref.split(",")) if ref else None
        try:
            cons = json.loads(constraints) if constraints else {}
            if not isinstance(cons, dict):
                raise ValueError("constraints must be a JSON object")
        except ValueError as e:
            raise HTTPException(400, f"bad constraints: {e}") from e
        check_rate(request)
        uploads = [u for u in ([file] if file else []) + (files or []) if u is not None]
        if not uploads:
            raise HTTPException(400, "upload at least one image as 'file' or 'files'")
        if len(uploads) > 6:
            raise HTTPException(400, "at most 6 photos per query")
        limit = settings.max_upload_mb * 1024 * 1024
        blobs = []
        for u in uploads:
            if u.size is not None and u.size > limit:  # refuse before buffering it
                raise HTTPException(413, f"upload exceeds {settings.max_upload_mb} MB")
            data = await u.read()
            if len(data) > limit:
                raise HTTPException(413, f"upload exceeds {settings.max_upload_mb} MB")
            if not data:
                raise HTTPException(400, "empty upload")
            blobs.append(data)
        try:
            async with app.state.gate:
                result = await run_in_threadpool(
                    ident.identify_many_bytes,
                    blobs,
                    top_n=top_n,
                    use_llm=use_llm,
                    constraints=cons,
                    tta=tta,
                    mm_per_px=mm_per_px,
                    reference=reference,
                    suggest_reference=log,  # not for live-preview frames
                )
        except (OSError, ValueError) as e:
            raise HTTPException(400, f"could not decode image: {e}") from e
        if log:  # keep the first photo so /feedback can file it under the confirmed part
            await record(result, blobs[0])
        return result

    @app.post("/feedback", response_model=Feedback)
    async def feedback(
        request: Request,
        request_id: str = Form(...),
        part_number: str | None = Form(
            None, description="Confirmed part number, or omit for 'none of these'"
        ),
        predicted: str | None = Form(None),
        tier: str | None = Form(None),
        file: UploadFile | None = File(None, description="Photo, if the server no longer holds it"),
        ident: Identifier = Depends(get_identifier),
    ) -> Feedback:
        check_rate(request)
        data = app.state.recent.get(request_id)
        if data is None and file is not None:
            # a resent photo is accepted only for an id this server actually issued, and
            # within the upload limit: /feedback must not be a free file drop
            if not app.state.requests.known(request_id):
                raise HTTPException(404, "unknown request_id")
            limit = settings.max_upload_mb * 1024 * 1024
            if file.size is not None and file.size > limit:
                raise HTTPException(413, f"upload exceeds {settings.max_upload_mb} MB")
            data = await file.read()
            if len(data) > limit:
                raise HTTPException(413, f"upload exceeds {settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(
                404, "photo for this request_id is no longer available; resend it as 'file'"
            )
        if part_number and ident.store.get(part_number.upper()) is None:
            raise HTTPException(404, "unknown part number")
        try:
            fb = await run_in_threadpool(
                app.state.feedback.record,
                data,
                request_id,
                part_number,
                predicted=predicted,
                tier=tier,
                source="tap",
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        app.state.events.log(
            "feedback",
            request_id=request_id,
            part_number=fb.part_number,
            predicted=predicted,
            correct=bool(fb.part_number and predicted == fb.part_number),
            source="tap",
        )
        return fb

    @app.get("/analytics")
    def analytics_view() -> dict:
        """The purchase funnel, confusion pairs, tier precision when bought, latency,
        errors, and a plain-language issues list."""
        a = analytics(app.state.events, app.state.feedback.stats())
        a["issues"] = issues(a)
        a["learning"] = _learning_state()
        return a

    def _learning_state() -> dict:
        from mcmaster_vision.pipeline.learn import learning_state

        return learning_state(settings, app.state.feedback)

    @app.post("/admin/learn")
    async def learn(request: Request) -> dict:
        """Fold new confirmations into the gallery now (`mcv learn --index-only`): the
        rebuilt index is picked up automatically."""
        check_admin(request)
        from mcmaster_vision.pipeline.learn import LearnBusy, learn_index

        try:
            result = await run_in_threadpool(learn_index, settings)
        except LearnBusy as e:
            raise HTTPException(409, str(e)) from e
        return result

    @app.get("/feedback/stats")
    def feedback_stats() -> dict:
        return app.state.feedback.stats()

    @app.get("/metrics")
    def metrics() -> dict:
        """Request volume, tier distribution, latency percentiles, and confirmed top-1 rate."""
        return app.state.requests.summary(app.state.feedback)

    @app.post("/identify/batch")
    async def identify_batch(
        request: Request,
        files: list[UploadFile] = File(
            ..., description="One photo per part (a bin, a drawer, a BOM)"
        ),
        top_n: int = Query(3, ge=1, le=20),
        ident: Identifier = Depends(get_identifier),
    ) -> list[dict]:
        """Identify many *different* parts in one call; returns one row per photo."""
        if len(files) > 200:
            raise HTTPException(400, "at most 200 photos per batch")
        check_rate(request, cost=len(files))
        rows = []
        limit = settings.max_upload_mb * 1024 * 1024
        for u in files:
            data = await u.read()
            if not data or len(data) > limit:
                msg = "empty upload" if not data else f"exceeds {settings.max_upload_mb} MB"
                rows.append({"file": u.filename, "error": msg})
                continue
            try:
                async with app.state.gate:
                    res = await run_in_threadpool(ident.identify_bytes, data, top_n=top_n)
            except (OSError, ValueError):
                rows.append({"file": u.filename, "error": "could not decode image"})
                continue
            await record(res, data)
            rows.append(
                {
                    "file": u.filename,
                    "request_id": res.request_id,
                    "tier": res.tier.value,
                    "best": res.best.part_number if res.best else None,
                    "confidence": res.best.confidence if res.best else None,
                    "candidates": [c.part_number for c in res.candidates],
                    "family": res.family.family_id if res.family else None,
                }
            )
        return rows

    @app.get("/parts/{part_number}", response_model=Part)
    def get_part(part_number: str, ident: Identifier = Depends(get_identifier)) -> Part:
        part = ident.store.get(part_number)
        if part is None:
            raise HTTPException(404, "unknown part number")
        return part

    @app.get("/parts/{part_number}/image", include_in_schema=False)
    def get_part_image(
        part_number: str, i: int = Query(0, ge=0), ident: Identifier = Depends(get_identifier)
    ):
        part = ident.store.get(part_number)
        if part is None or i >= len(part.image_paths) or not Path(part.image_paths[i]).exists():
            raise HTTPException(404, "no image")
        return FileResponse(part.image_paths[i], headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/parts/{part_number}/thumb", include_in_schema=False)
    def get_part_thumb(
        part_number: str,
        i: int = Query(0, ge=0),
        size: int = Query(200, ge=48, le=512),
        ident: Identifier = Depends(get_identifier),
    ):
        """Small JPEG thumbnails (cached on disk) keep the phone UI fast on cellular."""
        part = ident.store.get(part_number)
        if part is None or i >= len(part.image_paths):
            raise HTTPException(404, "no image")
        src = Path(part.image_paths[i])
        if not src.exists():
            raise HTTPException(404, "no image")
        size = min(THUMB_SIZES, key=lambda s: abs(s - size))  # a few sizes, not 465 files
        cache = settings.data_dir / "cache" / "thumbs"
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(part.part_number.encode("utf-8")).hexdigest()[:16]
        out = cache / f"{key}_{i}_{size}.jpg"
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            from PIL import Image, ImageOps

            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((size, size))
                canvas = Image.new("RGB", (size, size), (255, 255, 255))
                canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
                tmp = out.with_name(f".{out.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                canvas.save(tmp, format="JPEG", quality=85, optimize=True)
                tmp.replace(out)  # readers never see a half-written thumbnail
        return FileResponse(
            out, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800"}
        )

    @app.get("/parts/{part_number}/images")
    def list_part_images(part_number: str, ident: Identifier = Depends(get_identifier)) -> dict:
        part = ident.store.get(part_number)
        if part is None:
            raise HTTPException(404, "unknown part number")
        present = [i for i, p in enumerate(part.image_paths) if Path(p).exists()]
        pn = quote(part.part_number, safe="")
        return {
            "part_number": part.part_number,
            "count": len(present),
            "thumbs": [f"/parts/{pn}/thumb?i={i}" for i in present],
            "images": [f"/parts/{pn}/image?i={i}" for i in present],
        }

    @app.get("/categories")
    def categories(
        depth: int = Query(2, ge=1, le=4), ident: Identifier = Depends(get_identifier)
    ) -> list[dict]:
        """Catalog taxonomy with part counts, to the requested depth."""
        tax = ident.store.taxonomy()

        def walk(prefix: tuple[str, ...]) -> list[dict]:
            out = []
            for child in tax.children(prefix):
                path = (*prefix, child)
                node = {"name": child, "path": list(path), "parts": tax.count(path)}
                if len(path) < depth:
                    node["children"] = walk(path)
                out.append(node)
            return out

        return walk(())

    @app.get("/search", response_model=list[Part])
    def search(
        q: str = Query("", description="Keyword / part number; empty lists a category"),
        category: str = Query("", description="Category path prefix joined with ' > '"),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        ident: Identifier = Depends(get_identifier),
    ):
        prefix = [c.strip() for c in category.split(">") if c.strip()]
        if q.strip():
            hits = ident.store.search_text(q, limit + offset)
            if prefix:
                hits = [p for p in hits if p.category_path[: len(prefix)] == prefix]
            return hits[offset : offset + limit]
        if prefix or offset:
            return ident.store.by_category(prefix, limit=limit, offset=offset)
        raise HTTPException(400, "give q or category")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        try:
            app.state.events.log(
                "error", status=500, path=request.url.path, error=type(exc).__name__
            )
        except Exception:
            pass
        # no internals (paths, messages) leave the server; the log has the traceback
        return JSONResponse(
            status_code=500, content={"detail": "internal error; see the server log"}
        )

    return app


def get_app() -> FastAPI:
    """Factory for `uvicorn --factory mcmaster_vision.api.app:get_app` (multi-worker serving)."""
    return create_app(Settings())


def lan_urls(port: int, scheme: str = "http") -> list[str]:
    """URLs a phone on the same network can open."""
    import socket

    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except socket.gaierror:
        pass
    try:  # the interface that routes to the internet is usually the Wi-Fi one
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            ip = sock.getsockname()[0]
            if ip not in ips and not ip.startswith("127."):
                ips.insert(0, ip)
    except OSError:
        pass
    return [f"{scheme}://{ip}:{port}/" for ip in ips] or [f"{scheme}://localhost:{port}/"]


def self_signed_cert(cert_dir: Path, hosts: list[str]) -> tuple[Path, Path]:
    """Create (once) a self-signed certificate with `openssl`, valid for the LAN IPs."""
    import subprocess

    cert_dir.mkdir(parents=True, exist_ok=True)
    crt, key = cert_dir / "mcv.crt", cert_dir / "mcv.key"
    if crt.exists() and key.exists():
        return crt, key
    san = ",".join(["DNS:localhost", *[f"IP:{h}" for h in hosts]])
    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-sha256",
        "-nodes",
        "-days",
        "825",
        "-keyout",
        str(key),
        "-out",
        str(crt),
        "-subj",
        "/CN=mcmaster-vision",
        "-addext",
        f"subjectAltName={san}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise SystemExit(
            "--https needs the `openssl` command (brew install openssl / apt install openssl), "
            "or put the server behind a TLS proxy: see deploy/README.md"
        ) from e
    return crt, key


def port_in_use(host: str, port: int) -> str | None:
    """A short reason when nothing can listen on host:port, else None."""
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    bind_host = "" if host in ("0.0.0.0", "::") else host
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError as e:  # no IPv6 on this machine at all
        return e.strerror or str(e)
    with sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError as e:
            return e.strerror or str(e)
    return None


def print_qr(url: str) -> None:
    try:
        import qrcode  # type: ignore
    except ImportError:
        print("(pip install qrcode for a scannable QR code)")
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.print_ascii(invert=True)


def run(
    settings: Settings | None = None,
    host: str | None = None,
    port: int | None = None,
    workers: int = 1,
    https: bool = False,
    qr: bool = False,
) -> None:
    import uvicorn

    settings = settings or Settings()
    host = host or settings.api_host
    port = port or settings.api_port
    busy = port_in_use(host, port)
    if busy:
        raise SystemExit(
            f"port {port} is already in use on {host} ({busy}); stop the other server or pass "
            f"--port {port + 1}"
        )
    ssl: dict = {}
    if https:
        urls = lan_urls(port, "https")
        crt, key = self_signed_cert(
            settings.data_dir / "certs", [u.split("//")[1].split(":")[0] for u in urls]
        )
        ssl = {"ssl_certfile": str(crt), "ssl_keyfile": str(key)}
    if host in ("0.0.0.0", "::"):
        urls = lan_urls(port, "https" if https else "http")
        print("Open on your phone (same network): " + "  ".join(urls))
        if qr:
            print_qr(urls[0])
    # behind Caddy / nginx the client address comes from X-Forwarded-For; only trusted
    # proxies may set it (MCV_FORWARDED_ALLOW_IPS="*" in the compose deployment)
    proxy = {"proxy_headers": True, "forwarded_allow_ips": settings.forwarded_allow_ips}
    if workers > 1:
        # Each worker process loads its own copy of the index; size RAM accordingly. The
        # factory builds Settings() from the environment, so the resolved settings (a YAML
        # config, programmatic overrides) are exported first or the workers would serve
        # the defaults.
        for key, value in settings.model_dump(mode="json").items():
            if value is None or isinstance(value, (dict, list)):
                continue
            os.environ[f"MCV_{key.upper()}"] = (
                str(value).lower() if isinstance(value, bool) else str(value)
            )
        uvicorn.run(
            "mcmaster_vision.api.app:get_app",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            **proxy,
            **ssl,
        )
    else:
        uvicorn.run(create_app(settings), host=host, port=port, **proxy, **ssl)
