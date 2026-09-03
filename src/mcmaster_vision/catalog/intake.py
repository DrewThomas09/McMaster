"""Image intake: validate, normalise, de-duplicate, and download catalog images.

Real catalog drops are messy: mixed formats, EXIF-rotated phone shots, 20 MP
scans, duplicates, broken files, rows pointing at URLs instead of files. Every
path into the store goes through here first so the rest of the pipeline can
assume clean RGB images of a sane size.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

from mcmaster_vision.catalog.sources import CatalogSource, open_source
from mcmaster_vision.schemas import Part

log = logging.getLogger(__name__)

MAX_SIDE = 1024  # catalog images are stored at most this large; models use <= 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    parts: int = 0
    with_images: int = 0
    without_images: int = 0
    images: int = 0
    missing_files: int = 0
    unreadable: int = 0
    duplicate_part_numbers: int = 0
    duplicate_images: int = 0
    without_name: int = 0
    without_category: int = 0
    categories: int = 0
    families: int = 0
    tiny_images: int = 0  # < 64 px on the long side: useless for retrieval
    examples: dict[str, list[str]] = field(default_factory=dict)

    def note(self, key: str, value: str, limit: int = 5) -> None:
        bucket = self.examples.setdefault(key, [])
        if len(bucket) < limit:
            bucket.append(value)

    def ok(self) -> bool:
        return self.parts > 0 and self.with_images > 0 and self.duplicate_part_numbers == 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _image_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_source(
    source: str | Path | CatalogSource, *, check_images: bool = True, max_parts: int | None = None
) -> ValidationReport:
    """Walk a source once and report everything that would go wrong at ingest / index time."""
    src = source if isinstance(source, CatalogSource) else open_source(source)
    rep = ValidationReport()
    seen_pn: set[str] = set()
    seen_hash: set[str] = set()
    cats: set[str] = set()
    fams: set[str] = set()
    for i, part in enumerate(src):
        if max_parts and i >= max_parts:
            break
        rep.parts += 1
        if part.part_number in seen_pn:
            rep.duplicate_part_numbers += 1
            rep.note("duplicate_part_numbers", part.part_number)
        seen_pn.add(part.part_number)
        if not part.name or part.name == part.part_number:
            rep.without_name += 1
        if part.category_path:
            cats.add(" > ".join(part.category_path))
        else:
            rep.without_category += 1
        if part.family_id:
            fams.add(part.family_id)
        good = 0
        for p in part.image_paths:
            rep.images += 1
            path = Path(p)
            if not path.exists():
                rep.missing_files += 1
                rep.note("missing_files", p)
                continue
            if check_images:
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        if max(im.size) < 64:
                            rep.tiny_images += 1
                            rep.note("tiny_images", p)
                except (UnidentifiedImageError, OSError) as e:
                    rep.unreadable += 1
                    rep.note("unreadable", f"{p}: {e}")
                    continue
                digest = _image_hash(path)
                if digest in seen_hash:
                    rep.duplicate_images += 1
                    rep.note("duplicate_images", p)
                seen_hash.add(digest)
            good += 1
        if good:
            rep.with_images += 1
        else:
            rep.without_images += 1
            rep.note("without_images", part.part_number)
    rep.categories = len(cats)
    rep.families = len(fams)
    return rep


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def prepare_image(
    src: str | Path, dst_dir: str | Path, *, stem: str, max_side: int = MAX_SIDE
) -> Path | None:
    """Decode, fix EXIF orientation, flatten alpha onto white, cap the size, save as
    JPEG (or PNG when the source had transparency). Returns None if unreadable."""
    src, dst_dir = Path(src), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
            if has_alpha:
                rgba = im.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                bg.alpha_composite(rgba)
                im = bg.convert("RGB")
            else:
                im = im.convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            out = dst_dir / f"{stem}.jpg"
            im.save(out, format="JPEG", quality=92, optimize=True)
            return out
    except (UnidentifiedImageError, OSError) as e:
        log.warning("cannot prepare %s: %s", src, e)
        return None


def normalise_parts(
    parts: Iterable[Part], images_dir: str | Path, *, progress: Callable[[int], None] | None = None
) -> Iterator[Part]:
    """Copy every part's images into ``<images_dir>/<part_number>/`` in normalised
    form and drop unreadable / duplicate files. Yields updated parts."""
    images_dir = Path(images_dir)
    for i, part in enumerate(parts, 1):
        seen: set[str] = set()
        kept: list[str] = []
        for j, p in enumerate(part.image_paths):
            path = Path(p)
            if not path.exists():
                continue
            digest = _image_hash(path)
            if digest in seen:
                continue
            seen.add(digest)
            out = prepare_image(path, images_dir / part.part_number, stem=f"{part.part_number}_{j}")
            if out is not None:
                kept.append(str(out.resolve()))
        if progress and i % 1000 == 0:
            progress(i)
        yield part.model_copy(update={"image_paths": kept})


# ---------------------------------------------------------------------------
# URL lists -> files
# ---------------------------------------------------------------------------
def fetch_image_urls(
    records: Iterable[dict],
    images_dir: str | Path,
    *,
    client=None,
    delay_s: float = 0.2,
    max_per_part: int = 6,
    progress: Callable[[int], None] | None = None,
) -> Iterator[dict]:
    """For records carrying ``image_urls`` (list or ``;``-joined string), download the
    files into ``<images_dir>/<part_number>/`` and yield records with ``image_paths``.
    Already-downloaded files are skipped, so the call is resumable."""
    import time

    import httpx

    images_dir = Path(images_dir)
    own_client = client is None
    client = client or httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "mcmaster-vision/0.1 image fetch"},
    )
    try:
        for i, rec in enumerate(records, 1):
            urls = rec.get("image_urls") or []
            if isinstance(urls, str):
                urls = [u.strip() for u in urls.split(";") if u.strip()]
            pn = str(rec["part_number"]).strip()
            folder = images_dir / pn
            paths: list[str] = [p for p in (rec.get("image_paths") or []) if Path(p).exists()]
            for j, url in enumerate(urls[:max_per_part]):
                ext = Path(urlparse(url).path).suffix.lower() or ".jpg"
                if ext not in IMAGE_EXTS:
                    ext = ".jpg"
                out = folder / f"{pn}_{j}{ext}"
                if not out.exists():
                    try:
                        r = client.get(url)
                        if r.status_code != 200 or len(r.content) < 256:
                            log.warning("image fetch %s -> %s", url, r.status_code)
                            continue
                        folder.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(r.content)
                        if delay_s:
                            time.sleep(delay_s)
                    except httpx.HTTPError as e:
                        log.warning("image fetch failed %s: %s", url, e)
                        continue
                paths.append(str(out.resolve()))
            rec = dict(rec)
            rec["image_paths"] = paths
            rec.pop("image_urls", None)
            if progress and i % 100 == 0:
                progress(i)
            yield rec
    finally:
        if own_client:
            client.close()


def read_records(path: str | Path) -> Iterator[dict]:
    """Raw rows from a JSONL or CSV file (no Part validation)."""
    import csv

    path = Path(path)
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8", newline="") as fh:
            yield from csv.DictReader(fh)
    else:
        raise ValueError(f"expected .jsonl or .csv, got {path}")


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
            n += 1
    return n
