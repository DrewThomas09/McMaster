"""HTTP API.

POST /identify           multipart image -> IdentificationResult
GET  /parts/{pn}         catalog entry
GET  /parts/{pn}/image   first catalog image
GET  /search?q=          keyword search over the catalog
GET  /health, /stats
GET  /                   upload UI
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mcmaster_vision import __version__
from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.feedback import FeedbackStore
from mcmaster_vision.pipeline.identify import Identifier, load_identifier
from mcmaster_vision.pipeline.requestlog import RequestLog
from mcmaster_vision.schemas import Feedback, IdentificationResult, IndexStats, Part

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, identifier: Identifier | None = None) -> FastAPI:
    settings = settings or Settings()
    _recent: dict[str, bytes] = {}  # request_id -> first photo (bounded, in-memory)
    app = FastAPI(title="McMaster-Vision", version=__version__, description=__doc__)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if settings.cors_origins:
        origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
        )
    from mcmaster_vision.api.demo import router as demo_router

    app.include_router(demo_router)
    app.state.settings = settings
    app.state.identifier = identifier
    app.state.feedback = FeedbackStore(settings.queries_dir)
    app.state.requests = RequestLog(settings.data_dir / "logs" / "requests.jsonl")

    from mcmaster_vision.api.ratelimit import RateLimiter

    limiter = RateLimiter(settings.rate_limit_per_minute)

    def check_rate(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "rate limit exceeded; try again in a minute")

    def get_identifier() -> Identifier:
        if app.state.identifier is None:
            try:
                app.state.identifier = load_identifier(settings)
            except FileNotFoundError as e:
                raise HTTPException(503, f"index not built yet: {e}") from e
        return app.state.identifier

    @app.on_event("startup")
    def warm_up() -> None:
        """Load catalog + index + backbone at boot so the first photo is fast; if
        nothing is built yet the endpoints report 503 until `mcv bootstrap` runs."""
        if app.state.identifier is None and settings.warm_up:
            try:
                app.state.identifier = load_identifier(settings)
                log.info("identifier ready: %s", app.state.identifier.index.stats().model_dump())
            except FileNotFoundError as e:
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
        }

    @app.get("/status")
    def status_view() -> dict:
        from mcmaster_vision.pipeline.manifest import status as _status

        out = _status(settings)
        out["loaded"] = app.state.identifier is not None
        return out

    @app.post("/admin/reload")
    def reload(request: Request) -> dict:
        """Re-open the catalog and index after `mcv build-index` without restarting.
        Protected by MCV_API_TOKEN (header X-API-Token) when that is set."""
        token = settings.api_token
        if token and request.headers.get("x-api-token") != token:
            raise HTTPException(401, "bad or missing X-API-Token")
        try:
            app.state.identifier = load_identifier(settings)
        except FileNotFoundError as e:
            raise HTTPException(503, f"index not built yet: {e}") from e
        return {
            "reloaded": True,
            "index": app.state.identifier.index.stats().model_dump(mode="json"),
        }

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
        ident: Identifier = Depends(get_identifier),
    ) -> IdentificationResult:
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
            data = await u.read()
            if len(data) > limit:
                raise HTTPException(413, f"upload exceeds {settings.max_upload_mb} MB")
            if not data:
                raise HTTPException(400, "empty upload")
            blobs.append(data)
        try:
            result = await run_in_threadpool(
                ident.identify_many_bytes,
                blobs,
                top_n=top_n,
                use_llm=use_llm,
                constraints=cons,
                tta=tta,
            )
        except OSError as e:
            raise HTTPException(400, f"could not decode image: {e}") from e
        app.state.requests.log(result)
        # keep the first photo briefly so /feedback can file it under the confirmed part
        _recent[result.request_id] = blobs[0]
        if len(_recent) > 200:
            _recent.pop(next(iter(_recent)))
        return result

    @app.post("/feedback", response_model=Feedback)
    async def feedback(
        request_id: str = Form(...),
        part_number: str | None = Form(
            None, description="Confirmed part number, or omit for 'none of these'"
        ),
        predicted: str | None = Form(None),
        tier: str | None = Form(None),
        file: UploadFile | None = File(None, description="Photo, if the server no longer holds it"),
        ident: Identifier = Depends(get_identifier),
    ) -> Feedback:
        data = _recent.get(request_id)
        if data is None and file is not None:
            data = await file.read()
        if not data:
            raise HTTPException(
                404, "photo for this request_id is no longer available; resend it as 'file'"
            )
        if part_number and ident.store.get(part_number.upper()) is None:
            raise HTTPException(404, "unknown part number")
        return app.state.feedback.record(
            data, request_id, part_number, predicted=predicted, tier=tier
        )

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
        check_rate(request)
        rows = []
        limit = settings.max_upload_mb * 1024 * 1024
        for u in files:
            data = await u.read()
            if not data or len(data) > limit:
                msg = "empty upload" if not data else f"exceeds {settings.max_upload_mb} MB"
                rows.append({"file": u.filename, "error": msg})
                continue
            try:
                res = await run_in_threadpool(ident.identify_bytes, data, top_n=top_n)
            except OSError:
                rows.append({"file": u.filename, "error": "could not decode image"})
                continue
            app.state.requests.log(res)
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
        cache = settings.data_dir / "cache" / "thumbs"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / f"{part.part_number}_{i}_{size}.jpg"
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            from PIL import Image, ImageOps

            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((size, size))
                canvas = Image.new("RGB", (size, size), (255, 255, 255))
                canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
                canvas.save(out, format="JPEG", quality=85, optimize=True)
        return FileResponse(
            out, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800"}
        )

    @app.get("/parts/{part_number}/images")
    def list_part_images(part_number: str, ident: Identifier = Depends(get_identifier)) -> dict:
        part = ident.store.get(part_number)
        if part is None:
            raise HTTPException(404, "unknown part number")
        n = sum(1 for p in part.image_paths if Path(p).exists())
        return {
            "part_number": part.part_number,
            "count": n,
            "thumbs": [f"/parts/{part.part_number}/thumb?i={i}" for i in range(n)],
            "images": [f"/parts/{part.part_number}/image?i={i}" for i in range(n)],
        }

    @app.get("/search", response_model=list[Part])
    def search(
        q: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=100),
        ident: Identifier = Depends(get_identifier),
    ):
        return ident.store.search_text(q, limit)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

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
    subprocess.run(cmd, check=True, capture_output=True)
    return crt, key


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
    if workers > 1:
        # Each worker process loads its own copy of the index; size RAM accordingly.
        uvicorn.run(
            "mcmaster_vision.api.app:get_app",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            **ssl,
        )
    else:
        uvicorn.run(create_app(settings), host=host, port=port, **ssl)
