"""Import parts from the text of printed catalog pages (an OCR dump of a scanned catalog).

A catalog page lists one family per section as a table: a header row naming the
columns (``Pipe Size  90° Elbows  45° Elbows  Tees ...``), material sub-headings
(``Type 304 Stainless Steel``), then one row per size with a part number and a
price under each column::

    1/8 ...... 4464K11 .... $4.90   4464K35 .... $6.94   4464K23 .... $6.48

Every part number on such a row becomes a Part whose attributes carry the pipe
size, material, column (the fitting type), price and page. Look-alikes share a
family id (section + column + material) so the app can ask the size question or
measure it. No images come from text; add them with ``mcv fetch-images`` or
``mcv import-web``. OCR text is noisy, so the parser is lenient: anything that
does not look like a size row with part numbers is ignored.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from mcmaster_vision.schemas import Part

PART_PRICE = re.compile(r"(\d{4,5}[A-Z]\d{1,4}[A-Z]?)\s*[.\s]*\$?\s*(\d{1,4}\.\d\d)")
SIZE_TOKEN = re.compile(
    r"^\s*(?P<size>(?:\d+\s*[-\s]?\s*)?\d+/\d+|\d+(?:\.\d+)?)\s*(?:\"|″|”)?\s*(?:[.\s]|$)"
)
MATERIAL = re.compile(
    r"^\s*(type\s+\d{3}L?(?:/\d{3}L?)?\s+[a-z ]*?(?:stainless steel|steel)|"
    r"(?:brass|bronze|aluminum|nylon|pvc|cpvc|black steel|galvanized steel|"
    r"cast iron|malleable iron|carbon steel|copper|titanium|polypropylene)[a-z ()\-/]*)\s*(?:\(cont\.?\))?\s*$",
    re.I,
)
HEADER = re.compile(r"^\s*(pipe\s+size|pipe\s*/\s*thread\s+size|size|thread\s+size)\b", re.I)
FOOTER = re.compile(r"mcmaster-?carr", re.I)
NOISE_HEADINGS = {"pipe", "size", "lg.", "lg", "(a)", "(b)", "qty.", "dia."}

# OCR of the printed catalog: vulgar-fraction glyphs, "S" for "$", I/l/O inside part numbers
_VULGAR = {
    "⅛": "1/8",
    "¼": "1/4",
    "⅜": "3/8",
    "½": "1/2",
    "⅝": "5/8",
    "¾": "3/4",
    "⅞": "7/8",
    "⅙": "1/6",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅕": "1/5",
    "⅟": "1/",
}
_PN_LIKE = re.compile(r"\b([\dOIl]{4,5})([A-Z])([\dOIl]{1,4})([A-Z]?)\b")
_DIGIT_FIX = str.maketrans({"O": "0", "I": "1", "l": "1"})


def normalise_ocr(line: str) -> str:
    """Undo the OCR errors that hide rows: ``⅛`` -> ``1/8``, ``1¼`` -> ``1-1/4``,
    ``S6.00`` -> ``$6.00``, ``4464KI3`` -> ``4464K13``."""
    for glyph, frac in _VULGAR.items():
        line = re.sub(rf"(\d)\s*{glyph}", rf"\1-{frac}", line)  # 1¼ -> 1-1/4
        line = line.replace(glyph, frac)
    line = re.sub(r"(?<![A-Za-z])S(?=\d{1,4}\.\d\d\b)", "$", line)
    line = _PN_LIKE.sub(
        lambda m: (
            m.group(1).translate(_DIGIT_FIX)
            + m.group(2)
            + m.group(3).translate(_DIGIT_FIX)
            + m.group(4)
        ),
        line,
    )
    return line


@dataclass
class PageContext:
    page: str = ""
    title: str = ""
    section: str = ""
    material: str = ""
    columns: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)


def split_pages(text: str) -> list[str]:
    """Form feeds separate pages in an OCR dump; otherwise split on the catalog footer."""
    if "\f" in text:
        return [p for p in text.split("\f") if p.strip()]
    parts = re.split(r"\n(?=.*McMASTER-CARR)", text, flags=re.I)
    return [p for p in parts if p.strip()]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip(" .:")


def _columns_from_header(line: str) -> list[str]:
    body = HEADER.sub("", line, count=1)
    names = [_clean(n) for n in re.split(r"\s{2,}|\t", body) if _clean(n)]
    return [n for n in names if n.lower() not in NOISE_HEADINGS]


def _norm_size(raw: str) -> str:
    raw = raw.replace("″", "").replace("”", "").replace('"', "").strip()
    raw = re.sub(r"\s*-\s*", "-", raw)
    raw = re.sub(r"(\d)\s+(\d+/\d+)", r"\1-\2", raw)
    return raw


def _title(lines: list[str]) -> str:
    for ln in lines[:8]:
        t = _clean(ln)
        if len(t) > 8 and re.search(r"[A-Za-z]{4}", t) and not FOOTER.search(t):
            return re.sub(
                r"^(?:cad\s*)?(?:for technical.*?mcmaster\.com\.?)?\s*", "", t, flags=re.I
            )
    return ""


def _is_heading(t: str) -> bool:
    """Table/section headings are short Title Case (or ALL CAPS) lines without a full stop;
    prose ("The 90° elbows are also known as...") is not."""
    if len(t) < 6 or len(t) > 90 or t.endswith((".", ":", ";")) or not t[0].isupper():
        return False
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", t) if len(w) > 3]
    if not words:
        return False
    capped = sum(1 for w in words if w[0].isupper())
    return capped / len(words) >= 0.6


def parse_page(text: str, page_label: str = "") -> Iterator[Part]:
    lines = text.splitlines()
    ctx = PageContext(page=page_label, title=_title(lines))
    for raw in lines:
        line = normalise_ocr(raw.rstrip())
        if not line.strip() or FOOTER.search(line):
            continue
        if HEADER.match(line):
            cols = _columns_from_header(line)
            if cols:
                ctx.columns = cols
            continue
        m_mat = MATERIAL.match(line)
        if m_mat and not PART_PRICE.search(line):
            ctx.material = _clean(m_mat.group(1)).title().replace("Type ", "Type ")
            continue
        hits = PART_PRICE.findall(line)
        m_size = SIZE_TOKEN.match(line)
        if hits and m_size:
            yield from _row_parts(ctx, _norm_size(m_size.group("size")), line, hits, m_size.end())
            continue
        # a heading (no part numbers, not a size row) names the next table's section
        t = _clean(line)
        if not hits and _is_heading(t):
            t = re.sub(r"\s*\(continued.*?\)\s*", "", t, flags=re.I)
            ctx.section = t.title() if t.isupper() else t
            ctx.columns = []


_LENGTH_COLUMN = re.compile(r"^\s*([\d\s/-]+(?:\"|″|”)?)\s*(?:lg\.?|lengths?)\s*$", re.I)


def _row_parts(
    ctx: PageContext, size: str, line: str, hits: list[tuple[str, str]], size_end: int
) -> Iterator[Part]:
    # extra tokens between the size and the first part number (a nipple length, a thread)
    first_pn = hits[0][0]
    between = _clean(line[size_end : line.index(first_pn)]).strip(" .")
    for i, (pn, price) in enumerate(hits):
        column = (
            ctx.columns[i] if i < len(ctx.columns) else (ctx.columns[-1] if ctx.columns else "")
        )
        if len(ctx.columns) == 1 and len(hits) > 1:
            column = ctx.columns[0]
        attrs: dict[str, str] = {"pipe_size": size, "price_usd": price}
        if ctx.material:
            attrs["material"] = ctx.material
        m_len = _LENGTH_COLUMN.match(column) if column else None
        if m_len:  # a "2\" Lengths" column: the length is the attribute, the section the type
            attrs["length"] = _norm_size(m_len.group(1)).replace("-", " ") + '"'
            column = ctx.section
        elif column:
            attrs["fitting_type"] = column
        if between:
            attrs["length" if re.search(r"\d", between) else "note"] = between
        if ctx.page:
            attrs["catalog_page"] = ctx.page
        name_bits = [ctx.material, column or ctx.section, f"{size} pipe size"]
        name = ", ".join(b for b in name_bits if b)
        cat = [c for c in (ctx.title, ctx.section) if c]
        yield Part(
            part_number=pn.upper(),
            name=name or pn,
            category_path=cat or ["Catalog"],
            description=f"{ctx.section} - {column}".strip(" -") if ctx.section else column,
            attributes=attrs,
            image_paths=[],
            family_id=":".join(x for x in (ctx.section, column, ctx.material) if x) or None,
        )


def parse_catalog_text(text: str, first_page: int = 1) -> Iterator[Part]:
    """Parts from a whole OCR dump; duplicates keep the first occurrence."""
    seen: set[str] = set()
    for i, page in enumerate(split_pages(text), first_page):
        for part in parse_page(page, page_label=str(i)):
            if part.part_number not in seen:
                seen.add(part.part_number)
                yield part


def read_pages(paths: Iterable[str | Path]) -> Iterator[Part]:
    """One or more text files (a whole dump, or one file per page)."""
    for path in paths:
        yield from parse_catalog_text(Path(path).read_text(encoding="utf-8", errors="replace"))
