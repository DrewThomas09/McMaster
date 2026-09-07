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

PART_PRICE = re.compile(r"(\d{4,5}[A-Z]\d{1,4}[A-Z]?)\s*[.\s]*\$?\s*(\d{1,4}\.\d\d)(?!\d)")
# a size never looks like money (a wrapped row's continuation starts with a price)
SIZE_TOKEN = re.compile(
    r"^\s*(?!\d+\.\d\d\b)(?P<size>(?:\d+\s*[-\s]?\s*)?\d+/\d+|\d+(?:\.\d+)?)\s*(?:\"|″|”)?\s*(?:[.\s]|$)"
)
# a second "size ...." cell on the same line: two-column pages OCR into interleaved rows
COLUMN_BREAK = re.compile(
    r"(?<=\d\.\d\d)\s+(?=(?:\d{1,2}\s*[-\s]\s*)?\d{1,2}(?:/\d+)?\s*(?:\"|″|”)?\s*\.{2,})"
)
PLACEHOLDER = re.compile(r"(?:—|–|-{2,}|n/a)", re.I)
MATERIAL = re.compile(
    r"^\s*(type\s+\d{3}L?(?:/\d{3}L?)?\s+[a-z ]*?(?:stainless steel|steel)|"
    r"(?:brass|bronze|aluminum|nylon|pvc|cpvc|black steel|galvanized steel|"
    r"cast iron|malleable iron|carbon steel|copper|titanium|polypropylene)"
    r"(?:\s+(?:alloy|forged|cast|plated|coated|\d{3,4}|grade\s*\w+))?)\s*(?:\(cont\.?\))?\s*$",
    re.I,
)
CONT = re.compile(r"\s*\((?:cont(?:inued)?\.?|continued.*?)\)\s*", re.I)
HEADER = re.compile(
    r"^\s*((?:fits\s+)?pipe\s+size(?:\s+range)?|pipe\s*/\s*thread\s+size|size|thread\s+size)\b",
    re.I,
)
FOOTER = re.compile(r"mcmaster-?carr", re.I)
NOISE_HEADINGS = {"pipe", "size", "lg.", "lg", "(a)", "(b)", "qty.", "dia."}
# header cells that describe the row rather than name a product column, in the order
# their values appear between the pipe size and the first part number
ROW_FIELDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(?:pipe\s+)?size\s*\(b\)|^\s*\(b\)", re.I), "pipe_size_b"),
    (re.compile(r"max\.?\s*psi", re.I), "max_psi"),
    (re.compile(r"wall\s*thick", re.I), "wall_thickness"),
    (re.compile(r"flange\s*od", re.I), "flange_od"),
    (re.compile(r"^\s*od\b", re.I), "od"),
    (re.compile(r"^\s*id\b", re.I), "id"),
    (re.compile(r"outlet\s+pipe\s+size", re.I), "outlet_pipe_size"),
    (re.compile(r"^\s*ht\.?\s*$", re.I), "height"),
    (re.compile(r"hex\s*key", re.I), "hex_key_size"),
    (re.compile(r"^\s*width", re.I), "width"),
    (re.compile(r"^\s*thick\.?\s*$", re.I), "thickness"),
    (re.compile(r"^\s*thread\s*size\s*(?:\(b\))?", re.I), "thread_b"),
    (re.compile(r"^\s*lg\.?\s*$|^\s*length", re.I), "length"),
    (re.compile(r"^\s*\(c\)\s*$|^\s*c\s*$", re.I), "dimension_c"),
    (re.compile(r"qty", re.I), "bolt_qty"),
    (re.compile(r"^\s*dia\.?\s*$", re.I), "bolt_dia"),
    (re.compile(r"fits\s+pipe\s+size|size\s+range", re.I), "fits_pipe_size"),
]
# what a value between the size and the part number looks like, when the header did not say
PSI = re.compile(r"^\d{1,2},\d{3}$|^\d{3,6}$")
WALL = re.compile(r"^0?\.\d{2,3}\s*(?:\"|″|”)?$")
THREAD = re.compile(
    r"^(?:M\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?|\d+(?:/\d+)?(?:\"|″|”)?\s*-\s*\d+|#\d+-\d+)$", re.I
)
FRACTION = re.compile(r"^(?:\d+\s*[-\s]\s*)?\d+/\d+\s*(?:\"|″|”)?$|^\d+(?:\.\d+)?\s*(?:\"|″|”)$")
_STD = r"(?:NPTF|NPT|NPS[MHLC]?|BSPP|BSPT|BSP|METRIC|UN/UNF|UNF|UNEF|UN|GHT|NH/NST|SAE|JIC|ORB)"
THREAD_PAIR = re.compile(rf"\b({_STD}(?:\s+\([A-Z]\))?)\s*[x×]\s*({_STD}(?:\s+\([A-Z]\))?)\b", re.I)

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
    fields: list[str] = field(default_factory=list)  # row fields named by the header
    connection: str = ""  # "Butt weld" / "NPT" from the section's prose
    counters: dict[str, int] = field(default_factory=dict)


def split_pages(text: str) -> list[str]:
    """Form feeds separate pages in an OCR dump; otherwise split on the catalog footer."""
    if "\f" in text:
        return [p for p in text.split("\f") if p.strip()]
    parts = re.split(r"\n(?=.*McMASTER-CARR)", text, flags=re.I)
    return [p for p in parts if p.strip()]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip(" .:")


def _cells(line: str) -> list[str]:
    return [_clean(c) for c in re.split(r"\s{2,}|\t", line) if _clean(c)]


def _looks_like_header(line: str) -> bool:
    """Three or more wide-spaced cells of words, no part numbers, not a size row: a table
    header even when its first cell is not "Pipe Size" ("Thread   90° Elbows   Tees")."""
    cells = _cells(line)
    if len(cells) < 3 or PART_PRICE.search(line) or SIZE_TOKEN.match(line):
        return False
    return sum(1 for c in cells if re.search(r"[A-Za-z]{2}", c)) >= len(cells) - 1


def _columns_from_header(line: str) -> tuple[list[str], list[str]]:
    """(product columns, row fields): the header cells that describe the row (max psi,
    wall thickness, a second pipe size, flange OD...) are not product columns."""
    body = HEADER.sub("", line, count=1) if HEADER.match(line) else line
    names = _cells(body)
    if not HEADER.match(line) and names:
        names = names[1:]  # the first cell names the row key ("Thread", "Size (A)")
    cols: list[str] = []
    fields: list[str] = []
    for n in names:
        low = n.lower()
        key = next((k for rx, k in ROW_FIELDS if rx.search(n)), None)
        if key:
            fields.append(key)
        elif low not in NOISE_HEADINGS:
            cols.append(n)
    return cols, fields


def _norm_size(raw: str) -> str:
    raw = raw.replace("″", "").replace("”", "").replace('"', "").strip()
    raw = re.sub(r"\s*-\s*", "-", raw)
    raw = re.sub(r"(\d)\s+(\d+/\d+)", r"\1-\2", raw)
    return raw


def _title(lines: list[str]) -> str:
    for ln in lines[:8]:
        t = _clean(ln)
        if (
            len(t) > 8
            and re.search(r"[A-Za-z]{4}", t)
            and not FOOTER.search(t)
            and not _looks_like_header(ln)
        ):
            return re.sub(
                r"^(?:cad\s*)?(?:for technical.*?mcmaster\.com\.?)?\s*", "", t, flags=re.I
            )
    return ""


def _is_heading(t: str) -> bool:
    """Table/section headings are short Title Case (or ALL CAPS) lines without a full stop;
    prose ("The 90° elbows are also known as...") is not."""
    if len(t) < 6 or len(t) > 90 or t.endswith((".", ":", ";")) or not t[0].isupper():
        return False
    if len([c for c in re.split(r"\s{2,}|\t", t) if c.strip()]) >= 3:
        return False  # three or more cells: a table header, not a section
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
        if HEADER.match(line) or _looks_like_header(line):
            cols, fields = _columns_from_header(line)
            if cols or fields:
                ctx.columns = cols
                ctx.fields = fields
            continue
        m_conn = re.search(r"connections?:\s*([A-Za-z/ ()-]+?)(?:[.,;]|$)", line, re.I)
        if m_conn and not PART_PRICE.search(line):
            ctx.connection = _clean(m_conn.group(1))
            continue
        m_mat = MATERIAL.match(line)
        if m_mat and not PART_PRICE.search(line):
            ctx.material = _clean(m_mat.group(1)).title().replace("Type ", "Type ")
            continue
        hits = PART_PRICE.findall(line)
        m_size = SIZE_TOKEN.match(line)
        if hits and m_size:
            for seg in COLUMN_BREAK.split(line):  # interleaved two-column rows
                seg_hits = PART_PRICE.findall(seg)
                seg_size = SIZE_TOKEN.match(seg)
                if seg_hits and seg_size:
                    yield from _row_parts(
                        ctx, _norm_size(seg_size.group("size")), seg, seg_hits, seg_size.end()
                    )
            continue
        # a heading (no part numbers, not a size row) names the next table's section
        t = _clean(line)
        if not hits and _is_heading(t):
            cont = bool(CONT.search(t))
            t = CONT.sub("", t).strip()
            ctx.section = t.title() if t.isupper() else t
            if not cont:  # a continued table keeps the header from the previous page
                ctx.columns = []
                ctx.fields = []


_LENGTH_COLUMN = re.compile(
    r"^\s*([\d\s/-]+)\s*(?:\"|″|”)?\s*(?P<unit>ft\.?)?\s*(?:lg\.?|lengths?)?\s*$", re.I
)


def _row_values(between: str, fields: list[str]) -> dict[str, str]:
    """Attributes from what sits between the pipe size and the first part number: the
    header's row fields in order when it named them, else by what each token looks like
    (5,000 -> max psi, 0.083" -> wall thickness, M10 x 1.0 -> a thread, 3/4" -> a length)."""
    tokens = [t.strip(" .") for t in re.split(r"\s*\.{2,}\s*|\s{2,}", between)]
    tokens = [t for t in tokens if t]
    out: dict[str, str] = {}
    named = list(fields)
    shape = {
        "pipe_size_b": lambda t: bool(FRACTION.match(t)) or t.isdigit(),
        "outlet_pipe_size": lambda t: bool(FRACTION.match(t)) or t.isdigit(),
        "max_psi": lambda t: bool(PSI.match(t)),
        "wall_thickness": lambda t: bool(WALL.match(t)),
        "thread_b": lambda t: bool(THREAD.match(t)),
    }
    j = 0  # next header-named field to fill
    for tok in tokens:
        key = None
        # the next named field whose shape accepts the token (a missing cell skips it)
        while j < len(named):
            cand = named[j]
            j += 1
            if cand not in shape or shape[cand](tok):
                key = cand
                break
        if key is None:
            if PSI.match(tok):
                key = "max_psi"
            elif WALL.match(tok):
                key = "wall_thickness"
            elif THREAD.match(tok):
                key = "thread_b"
            elif FRACTION.match(tok) or tok.isdigit():
                key = key or ("length" if "length" not in out else "dimension")
            else:
                key = key or "note"
        if key == "max_psi":
            tok = tok.replace(",", "")
        elif key in ("pipe_size_b", "outlet_pipe_size", "fits_pipe_size"):
            tok = _norm_size(tok)
        elif key == "thread_b":
            tok = re.sub(r"\s*[x×]\s*", " x ", tok)
        out.setdefault(key, tok)
    return out


def _row_parts(
    ctx: PageContext, size: str, line: str, hits: list[tuple[str, str]], size_end: int
) -> Iterator[Part]:
    # extra tokens between the size and the first part number (a second pipe size, max
    # psi, wall thickness, a nipple length, a thread)
    first_pn = hits[0][0]
    between = line[size_end : line.index(first_pn)].strip(" .")
    m_range = re.match(r"to\s+((?:\d+\s*[-\s]\s*)?\d+(?:/\d+)?)\s*(?:\"|″|”)?\s*[.\s]*", between)
    fits_range = None
    if m_range:  # "1/4 to 36": the range of pipe sizes the part fits (an outlet)
        fits_range = f"{size} to {_norm_size(m_range.group(1))}"
        between = between[m_range.end() :].strip(" .")
    row_values = _row_values(between, ctx.fields) if between else {}
    if fits_range:
        row_values["fits_pipe_size"] = fits_range
        # the thread the photo shows is the outlet's; the range is what it welds onto
        size = row_values.get("outlet_pipe_size", size)
    # a max-psi cell in front of every column ("5,000 51205K162 $21.99  6,000 51205K113 ...")
    psi_by_hit: list[str | None] = []
    cursor_p = size_end
    for pn, _price in hits:
        start = line.index(pn, cursor_p)
        m_psi = re.search(
            r"(\d{1,2},\d{3}|\d{3,6})(?!\S)\s*[.\s]*$", line[cursor_p:start].rstrip(" .")
        )
        psi_by_hit.append(m_psi.group(1).replace(",", "") if m_psi else None)
        cursor_p = start + len(pn)
    # column index by position: a placeholder ("—", "n/a") between two cells is a skipped
    # column, so later part numbers must not shift left
    col_idx = []
    cursor = size_end
    col = 0
    for pn, _price in hits:
        start = line.index(pn, cursor)
        col += len(PLACEHOLDER.findall(line[cursor:start]))
        col_idx.append(col)
        col += 1
        cursor = start + len(pn)
    for i, (pn, price) in enumerate(hits):
        ci = col_idx[i]
        column = (
            ctx.columns[ci] if ci < len(ctx.columns) else (ctx.columns[-1] if ctx.columns else "")
        )
        if len(ctx.columns) == 1 and len(hits) > 1:
            column = ctx.columns[0]
        attrs: dict[str, str] = {"pipe_size": size, "price_usd": price}
        if ctx.material:
            attrs["material"] = ctx.material
        m_len = _LENGTH_COLUMN.match(column) if column else None
        if m_len and (m_len.group("unit") or re.search(r"lg|length", column, re.I)):
            # a "2\" Lengths" / "3 ft." column: the length is the attribute
            raw_len = _norm_size(m_len.group(1)).replace("-", " ")
            attrs["length"] = raw_len + (" ft" if m_len.group("unit") else '"')
            column = ctx.section
        elif column:
            attrs["fitting_type"] = column
        attrs.update(row_values)
        if psi_by_hit[i] and (not ctx.fields or "max_psi" in ctx.fields):
            attrs["max_psi"] = psi_by_hit[i]
        if ctx.connection:
            attrs["connection"] = ctx.connection
        m_pair = THREAD_PAIR.search(column or "") or THREAD_PAIR.search(ctx.section or "")
        if m_pair:  # "BSPP (A) x NPT (B)": each end's thread standard
            attrs["thread_a"] = re.sub(r"\s*\([A-Z]\)", "", m_pair.group(1)).upper()
            attrs["thread_b_type"] = re.sub(r"\s*\([A-Z]\)", "", m_pair.group(2)).upper()
        if ctx.page:
            attrs["catalog_page"] = ctx.page
        size_text = f"{size} x {attrs['pipe_size_b']}" if "pipe_size_b" in attrs else size
        name_bits = [ctx.material, column or ctx.section, f"{size_text} pipe size"]
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


def parse_catalog_text(
    text: str, first_page: int = 1, seen: set[str] | None = None
) -> Iterator[Part]:
    """Parts from a whole OCR dump; duplicates (``seen`` may span files) keep the first."""
    seen = set() if seen is None else seen
    for i, page in enumerate(split_pages(text), first_page):
        for part in parse_page(page, page_label=str(i)):
            if part.part_number not in seen:
                seen.add(part.part_number)
                yield part


def read_pages(paths: Iterable[str | Path], first_page: int = 1) -> Iterator[Part]:
    """One or more text files (a whole dump, or one file per page): page numbers run on
    across files and a part number is imported once."""
    seen: set[str] = set()
    page = first_page
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        yield from parse_catalog_text(text, first_page=page, seen=seen)
        page += len(split_pages(text))
