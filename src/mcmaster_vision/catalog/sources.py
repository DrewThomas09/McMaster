"""Catalog data sources.

McMaster-Carr does not publish a bulk product-data API for the general public, and
their terms of use prohibit scraping. Every source below therefore consumes data
you are entitled to use:

* ``JSONLSource`` / ``CSVSource`` - exports you already hold (one row per SKU).
* ``DirectorySource`` - a folder tree ``<root>/<part_number>/{meta.json, *.jpg}``.
* ``McMasterApiSource`` - the account-holder Product Information API. It is a stub
  that documents the contract; wire in your credentials and endpoint mapping.

All sources yield :class:`~mcmaster_vision.schemas.Part` objects.
"""

from __future__ import annotations

import csv
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mcmaster_vision.schemas import Part

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class CatalogSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Part]: ...

    def __len__(self) -> int:  # optional; used for progress bars
        return 0


def _split_path(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [p.strip() for p in str(value).replace("|", ">").split(">") if p.strip()]


def _split_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [p.strip() for p in str(value).split(";") if p.strip()]


def part_from_record(rec: dict[str, Any], base_dir: Path | None = None) -> Part:
    """Normalise a loosely-typed row into a Part."""
    attributes = rec.get("attributes") or {}
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except json.JSONDecodeError:
            attributes = {}
    # Any unknown column becomes an attribute; that keeps CSV exports simple.
    known = {
        "part_number",
        "name",
        "category_path",
        "category",
        "description",
        "attributes",
        "image_paths",
        "images",
        "family_id",
        "url",
    }
    for k, v in rec.items():
        if k not in known and v not in (None, ""):
            attributes.setdefault(k, v)

    images = _split_list(rec.get("image_paths") or rec.get("images"))
    if base_dir is not None:
        images = [str((base_dir / p).resolve()) if not Path(p).is_absolute() else p for p in images]

    return Part(
        part_number=str(rec["part_number"]).strip(),
        name=str(rec.get("name") or rec.get("description") or rec["part_number"]).strip(),
        category_path=_split_path(rec.get("category_path") or rec.get("category")),
        description=str(rec.get("description") or ""),
        attributes=attributes,
        image_paths=images,
        family_id=rec.get("family_id") or None,
        url=rec.get("url") or None,
    )


class JSONLSource(CatalogSource):
    """One JSON object per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[Part]:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield part_from_record(json.loads(line), self.path.parent)

    def __len__(self) -> int:
        with open(self.path, encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())


class CSVSource(CatalogSource):
    """CSV with at least ``part_number`` and ``name`` columns.

    ``category_path`` uses ``>`` separators; ``image_paths`` uses ``;``.
    Extra columns are stored as attributes.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[Part]:
        with open(self.path, encoding="utf-8", newline="") as fh:
            for rec in csv.DictReader(fh):
                yield part_from_record(rec, self.path.parent)

    def __len__(self) -> int:
        with open(self.path, encoding="utf-8", newline="") as fh:
            return max(0, sum(1 for _ in fh) - 1)


class DirectorySource(CatalogSource):
    """``<root>/<part_number>/`` folders, each with images and an optional ``meta.json``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __iter__(self) -> Iterator[Part]:
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            meta_path = folder / "meta.json"
            rec: dict[str, Any] = {"part_number": folder.name, "name": folder.name}
            if meta_path.exists():
                rec.update(json.loads(meta_path.read_text(encoding="utf-8")))
            rec["image_paths"] = [
                str(p.resolve()) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS
            ]
            yield part_from_record(rec)

    def __len__(self) -> int:
        return sum(1 for p in self.root.iterdir() if p.is_dir())


class FlatImageSource(CatalogSource):
    """A folder of screenshots / photos named by part number: ``<root>/<pn>.png``,
    ``<root>/<pn>_1.jpg`` ... Optional ``<root>/meta.jsonl`` adds names, categories
    and attributes keyed by part number."""

    _NAME_RE = re.compile(r"^([0-9]{4,5}[A-Za-z][0-9A-Za-z]{1,5})(?:[_\- ].*)?$")

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _groups(self) -> dict[str, list[Path]]:
        groups: dict[str, list[Path]] = {}
        for f in sorted(self.root.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            m = self._NAME_RE.match(f.stem)
            if m:
                groups.setdefault(m.group(1).upper(), []).append(f)
        return groups

    def __iter__(self) -> Iterator[Part]:
        meta: dict[str, dict[str, Any]] = {}
        meta_path = self.root / "meta.jsonl"
        if meta_path.exists():
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    meta[str(rec["part_number"]).upper()] = rec
        for pn, files in self._groups().items():
            rec: dict[str, Any] = {"part_number": pn, "name": pn}
            rec.update(meta.get(pn, {}))
            rec["image_paths"] = [str(f.resolve()) for f in files]
            yield part_from_record(rec)

    def __len__(self) -> int:
        return len(self._groups())


class McMasterApiSource(CatalogSource):
    """Adapter for the McMaster-Carr Product Information API (account holders only).

    The API requires a client certificate and credentials issued by McMaster-Carr.
    Implement :meth:`fetch_page` against your approved endpoint; the rest of the
    pipeline needs nothing else.
    """

    def __init__(self, base_url: str, username: str, password: str, cert_path: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.cert_path = cert_path

    def fetch_page(self, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        raise NotImplementedError(
            "Wire this to your McMaster-Carr API credentials. Return (records, next_cursor)."
        )

    def __iter__(self) -> Iterator[Part]:
        cursor: str | None = None
        while True:
            records, cursor = self.fetch_page(cursor)
            for rec in records:
                yield part_from_record(rec)
            if not cursor:
                break


def open_source(path: str | Path) -> CatalogSource:
    p = Path(path)
    if p.is_dir():
        has_subdirs = any(c.is_dir() for c in p.iterdir())
        return DirectorySource(p) if has_subdirs else FlatImageSource(p)
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        return JSONLSource(p)
    if p.suffix.lower() == ".csv":
        return CSVSource(p)
    raise ValueError(f"Unsupported catalog source: {p}")
