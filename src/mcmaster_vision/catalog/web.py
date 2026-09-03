"""Import parts from McMaster-Carr product pages (and any product-style web page).

Given part numbers or product URLs, the importer fetches each page, extracts the
part number, name, category breadcrumb, spec table and product image(s), stores
the images under ``<image_dir>/<part_number>/`` and returns :class:`Part`
records ready for :func:`ingest`. It is *polite by design*: one request at a
time, a configurable delay, honouring ``robots.txt``, and an on-disk cache so a
page is never fetched twice. It is meant for the parts you care about (an
order history, a BOM, a shelf), not for crawling the whole 700k catalog.

McMaster-Carr renders most of its catalog client-side, so the parser tries, in
order: JSON-LD ``Product`` blocks, Open Graph / meta tags, ``<img>`` tags that
look like product images, and the page title. Adjust :class:`McMasterParser`
if the site's markup changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from mcmaster_vision.catalog.sources import CatalogSource
from mcmaster_vision.pipeline.ocr import PART_NUMBER_RE
from mcmaster_vision.schemas import Part

log = logging.getLogger(__name__)

MCMASTER_BASE = "https://www.mcmaster.com"
USER_AGENT = "mcmaster-vision/0.1 (+catalog importer; polite, cached, rate-limited)"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


# ---------------------------------------------------------------------------
# HTML parsing (stdlib only)
# ---------------------------------------------------------------------------
@dataclass
class PageData:
    url: str
    part_number: str | None = None
    name: str | None = None
    description: str = ""
    category_path: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    image_urls: list[str] = field(default_factory=list)


class _Extractor(HTMLParser):
    """Collects title, meta tags, JSON-LD, img tags, breadcrumb links, and table rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta: dict[str, str] = {}
        self.jsonld: list[str] = []
        self.images: list[dict[str, str]] = []
        self.links: list[tuple[str, str, str]] = []  # (href, class, text)
        self.rows: list[list[str]] = []
        self._stack: list[str] = []
        self._text_target: list[str] | None = None
        self._in_title = False
        self._in_jsonld = False
        self._link: tuple[str, str] | None = None
        self._link_text: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta" and (a.get("property") or a.get("name")):
            self.meta[(a.get("property") or a.get("name")).lower()] = a.get("content", "")
        elif tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
            self._in_jsonld = True
        elif tag == "img":
            self.images.append({k: v for k, v in a.items() if v})
        elif tag == "a" and a.get("href"):
            self._link = (a["href"], a.get("class", ""))
            self._link_text = []
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_jsonld = False
        elif tag == "a" and self._link is not None:
            self.links.append((self._link[0], self._link[1], " ".join(self._link_text).strip()))
            self._link = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._in_jsonld:
            self.jsonld.append(data)
        if self._link is not None:
            self._link_text.append(data.strip())
        if self._cell is not None:
            self._cell.append(data.strip())


def _clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


class McMasterParser:
    """Turn a product page's HTML into :class:`PageData`."""

    IMAGE_HINTS = ("imagecache", "/mv", "product", "/gfx/", "item")
    NOISE_IMAGE_HINTS = ("logo", "icon", "sprite", "pixel", "spacer", "arrow", "badge")

    def parse(self, url: str, html: str) -> PageData:
        ex = _Extractor()
        ex.feed(html)
        data = PageData(url=url)

        # 1. JSON-LD Product
        for blob in ex.jsonld:
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            for node in obj if isinstance(obj, list) else [obj]:
                if isinstance(node, dict) and str(node.get("@type", "")).lower() == "product":
                    data.name = data.name or _clean(node.get("name"))
                    data.description = data.description or _clean(node.get("description"))
                    data.part_number = (
                        data.part_number or _clean(node.get("sku") or node.get("mpn")) or None
                    )
                    imgs = node.get("image") or []
                    for im in imgs if isinstance(imgs, list) else [imgs]:
                        if isinstance(im, str):
                            data.image_urls.append(urljoin(url, im))
                    for prop in node.get("additionalProperty") or []:
                        if isinstance(prop, dict) and prop.get("name"):
                            data.attributes[_clean(prop["name"])] = _clean(
                                str(prop.get("value", ""))
                            )
                if (
                    isinstance(node, dict)
                    and str(node.get("@type", "")).lower() == "breadcrumblist"
                ):
                    items = sorted(
                        node.get("itemListElement") or [], key=lambda e: e.get("position", 0)
                    )
                    data.category_path = [
                        _clean(e.get("name")) for e in items if _clean(e.get("name"))
                    ]

        # 2. meta tags
        og_image = ex.meta.get("og:image")
        if og_image:
            data.image_urls.append(urljoin(url, og_image))
        data.name = data.name or _clean(ex.meta.get("og:title")) or None
        data.description = data.description or _clean(
            ex.meta.get("description") or ex.meta.get("og:description")
        )

        # 3. title / url -> part number
        title = _clean(ex.title)
        if not data.name and title:
            data.name = re.sub(r"\s*[|\-–]\s*McMaster-Carr.*$", "", title, flags=re.I)
        if not data.part_number:
            for candidate in (urlparse(url).path, title, data.name or ""):
                m = PART_NUMBER_RE.search(candidate.upper())
                if m:
                    data.part_number = m.group(1)
                    break

        # 4. product images from <img>
        for im in ex.images:
            src = im.get("src") or im.get("data-src") or im.get("data-original") or ""
            low = src.lower()
            if not src or any(n in low for n in self.NOISE_IMAGE_HINTS):
                continue
            alt = (im.get("alt") or "").lower()
            if any(h in low for h in self.IMAGE_HINTS) or (
                data.part_number and data.part_number.lower() in (low + alt)
            ):
                data.image_urls.append(urljoin(url, src))

        # 5. breadcrumb links when JSON-LD had none
        if not data.category_path:
            crumbs = [t for _, cls, t in ex.links if "breadcrumb" in cls.lower() and t]
            if crumbs:
                data.category_path = crumbs

        # 6. spec table rows (2 columns -> attribute)
        for row in ex.rows:
            if len(row) == 2 and row[0] and row[1] and len(row[0]) < 40:
                data.attributes.setdefault(_clean(row[0]).rstrip(":"), _clean(row[1]))

        # de-duplicate, keep order
        seen: set[str] = set()
        data.image_urls = [u for u in data.image_urls if not (u in seen or seen.add(u))]
        data.name = data.name or data.part_number
        return data


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
class RobotsPolicy:
    """Minimal robots.txt check for our user agent (Disallow prefixes under ``*``)."""

    def __init__(self, disallow: Iterable[str] = ()):
        self.disallow = [d for d in disallow if d]

    @classmethod
    def parse(cls, text: str, agent: str = "*") -> RobotsPolicy:
        rules: list[str] = []
        applies = False
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if key == "user-agent":
                applies = val == "*" or val.lower() in agent.lower()
            elif key == "disallow" and applies:
                rules.append(val)
        return cls(rules)

    def allowed(self, url: str) -> bool:
        path = urlparse(url).path or "/"
        return not any(path.startswith(rule) for rule in self.disallow)


class WebImporter:
    def __init__(
        self,
        image_dir: str | Path,
        *,
        cache_dir: str | Path | None = None,
        delay_s: float = 1.5,
        timeout_s: float = 30.0,
        max_images: int = 4,
        respect_robots: bool = True,
        parser: McMasterParser | None = None,
        client: httpx.Client | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.image_dir / ".cache"
        self.delay_s = delay_s
        self.max_images = max_images
        self.respect_robots = respect_robots
        self.parser = parser or McMasterParser()
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=timeout_s, follow_redirects=True
        )
        self._robots: dict[str, RobotsPolicy] = {}
        self._last_request = 0.0
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def url_for(part_number_or_url: str) -> str:
        s = part_number_or_url.strip()
        if s.lower().startswith("http"):
            return s
        return f"{MCMASTER_BASE}/{s.upper()}/"

    def _throttle(self) -> None:
        wait = self.delay_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _robots_for(self, url: str) -> RobotsPolicy:
        host = urlparse(url).netloc
        if host not in self._robots:
            policy = RobotsPolicy()
            if self.respect_robots:
                try:
                    r = self.client.get(f"{urlparse(url).scheme}://{host}/robots.txt")
                    if r.status_code == 200:
                        policy = RobotsPolicy.parse(r.text, USER_AGENT)
                except httpx.HTTPError as e:
                    log.debug("robots.txt unavailable for %s: %s", host, e)
            self._robots[host] = policy
        return self._robots[host]

    def _cached_get(self, url: str, binary: bool = False) -> bytes | None:
        key = hashlib.sha1(url.encode()).hexdigest()
        path = self.cache_dir / (key + (".bin" if binary else ".html"))
        if path.exists():
            return path.read_bytes()
        if not self._robots_for(url).allowed(url):
            log.warning("robots.txt disallows %s; skipping", url)
            return None
        self._throttle()
        try:
            r = self.client.get(url)
        except httpx.HTTPError as e:
            log.warning("fetch failed %s: %s", url, e)
            return None
        if r.status_code != 200:
            log.warning("fetch %s -> HTTP %s", url, r.status_code)
            return None
        path.write_bytes(r.content)
        return r.content

    # ------------------------------------------------------------- public
    def fetch_page(self, part_number_or_url: str) -> PageData | None:
        url = self.url_for(part_number_or_url)
        body = self._cached_get(url)
        if body is None:
            return None
        data = self.parser.parse(url, body.decode("utf-8", errors="replace"))
        if not data.part_number and not part_number_or_url.lower().startswith("http"):
            data.part_number = part_number_or_url.strip().upper()
        return data

    def download_images(self, data: PageData) -> list[str]:
        if not data.part_number:
            return []
        folder = self.image_dir / data.part_number
        folder.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, img_url in enumerate(data.image_urls[: self.max_images]):
            ext = next(
                (e for e in IMAGE_EXTS if urlparse(img_url).path.lower().endswith(e)), ".jpg"
            )
            out = folder / f"{data.part_number}_{i}{ext}"
            if not out.exists():
                blob = self._cached_get(img_url, binary=True)
                if blob is None or len(blob) < 512:
                    continue
                out.write_bytes(blob)
            paths.append(str(out.resolve()))
        return paths

    def import_one(self, part_number_or_url: str) -> Part | None:
        data = self.fetch_page(part_number_or_url)
        if data is None or not data.part_number:
            return None
        images = self.download_images(data)
        return Part(
            part_number=data.part_number,
            name=data.name or data.part_number,
            category_path=data.category_path,
            description=data.description,
            attributes=data.attributes,
            image_paths=images,
            family_id=None,
            url=data.url,
        )

    def import_many(self, items: Iterable[str]) -> Iterator[Part]:
        for item in items:
            item = item.strip()
            if not item or item.startswith("#"):
                continue
            part = self.import_one(item)
            if part is not None:
                yield part


class WebSource(CatalogSource):
    """CatalogSource adapter so ``ingest()`` can consume the importer directly."""

    def __init__(self, importer: WebImporter, items: Iterable[str]):
        self.importer = importer
        self.items = list(items)

    def __iter__(self) -> Iterator[Part]:
        return self.importer.import_many(self.items)

    def __len__(self) -> int:
        return len(self.items)


def read_items(path: str | Path) -> list[str]:
    """Part numbers / URLs, one per line (``#`` comments allowed)."""
    return [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
