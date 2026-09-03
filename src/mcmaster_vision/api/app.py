"""HTTP API.

POST /identify           multipart image -> IdentificationResult
GET  /parts/{pn}         catalog entry
GET  /parts/{pn}/image   first catalog image
GET  /search?q=          keyword search over the catalog
GET  /health, /stats
GET  /                   upload UI
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from mcmaster_vision import __version__
from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.feedback import FeedbackStore
from mcmaster_vision.pipeline.identify import Identifier, load_identifier
from mcmaster_vision.schemas import Feedback, IdentificationResult, IndexStats, Part

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, identifier: Identifier | None = None) -> FastAPI:
    settings = settings or Settings()
    _recent: dict[str, bytes] = {}  # request_id -> first photo (bounded, in-memory)
    app = FastAPI(title="McMaster-Vision", version=__version__, description=__doc__)
    app.state.settings = settings
    app.state.identifier = identifier
    app.state.feedback = FeedbackStore(settings.queries_dir)

    def get_identifier() -> Identifier:
        if app.state.identifier is None:
            try:
                app.state.identifier = load_identifier(settings)
            except FileNotFoundError as e:
                raise HTTPException(503, f"index not built yet: {e}") from e
        return app.state.identifier

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "ready": app.state.identifier is not None}

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
        ident: Identifier = Depends(get_identifier),
    ) -> IdentificationResult:
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
            result = ident.identify_many_bytes(blobs, top_n=top_n, use_llm=use_llm)
        except OSError as e:
            raise HTTPException(400, f"could not decode image: {e}") from e
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

    @app.get("/parts/{part_number}", response_model=Part)
    def get_part(part_number: str, ident: Identifier = Depends(get_identifier)) -> Part:
        part = ident.store.get(part_number)
        if part is None:
            raise HTTPException(404, "unknown part number")
        return part

    @app.get("/parts/{part_number}/image", include_in_schema=False)
    def get_part_image(part_number: str, ident: Identifier = Depends(get_identifier)):
        part = ident.store.get(part_number)
        if part is None or not part.image_paths or not Path(part.image_paths[0]).exists():
            raise HTTPException(404, "no image")
        return FileResponse(part.image_paths[0])

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


def run(settings: Settings | None = None, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    settings = settings or Settings()
    uvicorn.run(
        create_app(settings), host=host or settings.api_host, port=port or settings.api_port
    )
