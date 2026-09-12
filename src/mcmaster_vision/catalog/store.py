"""SQLite-backed catalog store.

SQLite is deliberately chosen for the skeleton: a 700k-row parts table with a JSON
attributes column is a few hundred MB and needs no server. Swap for Postgres by
re-implementing this class; nothing else in the system touches SQL.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from mcmaster_vision.catalog.taxonomy import Taxonomy
from mcmaster_vision.schemas import Part

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parts (
    part_number   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    category_path TEXT NOT NULL,   -- JSON list
    description   TEXT NOT NULL DEFAULT '',
    attributes    TEXT NOT NULL DEFAULT '{}',  -- JSON object
    image_paths   TEXT NOT NULL DEFAULT '[]',  -- JSON list
    family_id     TEXT,
    url           TEXT
);
CREATE INDEX IF NOT EXISTS idx_parts_family ON parts(family_id);
CREATE INDEX IF NOT EXISTS idx_parts_category ON parts(category_path);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS parts_fts USING fts5(
    part_number, name, category, description, attributes
);
"""


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _fts_tokens(text: str) -> list[str]:
    """The tokens FTS5's unicode61 tokenizer would produce (alphanumeric runs)."""
    return re.findall(r"[0-9A-Za-z]+", text)


EXACT_VALUE_BONUS = 0.25  # a spec value typed verbatim: that many times stronger a match


def _promote_exact_values(hits: list[tuple[Part, float]], query: str) -> list[tuple[Part, float]]:
    """A spec value typed as it is written (``1/2"``, ``Galvanized Steel``, ``M6``) is an
    exact match, not two more words for bm25: ``1-1/2"`` mentions ``1`` twice and would
    otherwise outscore ``1/2"``. A hit whose attribute value appears in the query as a
    whole whitespace-delimited token gets its text score strengthened by
    ``EXACT_VALUE_BONUS`` per such value, so the exact variants form the leading tier and
    the facet chips can narrow twice. Part-number matches keep their pinned score."""
    if not hits or not query:
        return hits
    q = query.lower()
    out = []
    for part, score in hits:
        n = 0
        if score > CatalogStore.PINNED_SCORE:
            for v in (part.attributes or {}).values():
                text = str(v).strip().lower()
                if text and re.search(rf"(?<!\S){re.escape(text)}(?!\S)", q):
                    n += 1
        if n:
            score = score * (1 + EXACT_VALUE_BONUS * n) if score < 0 else -EXACT_VALUE_BONUS * n
        out.append((part, score))
    out.sort(key=lambda x: x[1])
    return out


class CatalogStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._fts = self._try_enable_fts()

    # ------------------------------------------------------------------ setup
    def _try_enable_fts(self) -> bool:
        try:
            self._conn.executescript(_FTS_SCHEMA)
            return True
        except sqlite3.OperationalError:
            return False

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CatalogStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- writes
    def upsert(self, parts: Iterable[Part], batch_size: int = 5000, *, merge: bool = False) -> int:
        """Insert or replace parts. With ``merge=True`` an existing row's image paths are
        kept (union) and its name / category / description survive when the new record
        has none, so enriching a part never erases what a previous import found."""
        n = 0
        batch: list[Part] = []
        for part in parts:
            batch.append(self._merged(part) if merge else part)
            if len(batch) >= batch_size:
                n += self._write_batch(batch)
                batch = []
        if batch:
            n += self._write_batch(batch)
        return n

    def _merged(self, part: Part) -> Part:
        old = self.get(part.part_number)
        if old is None:
            return part
        images = list(dict.fromkeys([*old.image_paths, *part.image_paths]))
        return part.model_copy(
            update={
                "image_paths": images,
                "name": part.name if part.name and part.name != part.part_number else old.name,
                "category_path": part.category_path or old.category_path,
                "description": part.description or old.description,
                "attributes": {**old.attributes, **part.attributes},
                "family_id": part.family_id or old.family_id,
                "url": part.url or old.url,
            }
        )

    def _write_batch(self, batch: list[Part]) -> int:
        rows = [
            (
                p.part_number,
                p.name,
                json.dumps(p.category_path),
                p.description,
                json.dumps(p.attributes, sort_keys=True, default=str),
                json.dumps(p.image_paths),
                p.family_id,
                p.url,
            )
            for p in batch
        ]
        with self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO parts VALUES (?,?,?,?,?,?,?,?)", rows)
            if self._fts:
                # delete through the FTS index: a plain WHERE on an FTS5 table is a full
                # scan per row, which made every ingest quadratic in catalog size
                for p in batch:
                    self._fts_delete(p.part_number)
                self._conn.executemany(
                    "INSERT INTO parts_fts(part_number, name, category, description, attributes) "
                    "VALUES (?,?,?,?,?)",
                    [
                        (
                            p.part_number,
                            p.name,
                            " ".join(p.category_path),
                            p.description,
                            " ".join(f"{k} {v}" for k, v in p.attributes.items()),
                        )
                        for p in batch
                    ],
                )
        from datetime import datetime, timezone

        self._conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('updated_at', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        self._conn.commit()
        return len(batch)

    def _fts_delete(self, part_number: str) -> None:
        tokens = _fts_tokens(part_number)
        if tokens:
            phrase = 'part_number:"' + " ".join(tokens).replace('"', '""') + '"'
            self._conn.execute(
                "DELETE FROM parts_fts WHERE rowid IN (SELECT rowid FROM parts_fts "
                "WHERE parts_fts MATCH ? AND part_number = ?)",
                (phrase, part_number),
            )
        else:  # no indexable token (punctuation only): the slow path is the only one
            self._conn.execute("DELETE FROM parts_fts WHERE part_number = ?", (part_number,))

    def set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    # ----------------------------------------------------------------- reads
    @staticmethod
    def _row_to_part(row: sqlite3.Row) -> Part:
        return Part(
            part_number=row["part_number"],
            name=row["name"],
            category_path=json.loads(row["category_path"]),
            description=row["description"],
            attributes=json.loads(row["attributes"]),
            image_paths=json.loads(row["image_paths"]),
            family_id=row["family_id"],
            url=row["url"],
        )

    def get(self, part_number: str) -> Part | None:
        row = self._conn.execute(
            "SELECT * FROM parts WHERE part_number=?", (part_number,)
        ).fetchone()
        return self._row_to_part(row) if row else None

    def get_many(self, part_numbers: Iterable[str]) -> dict[str, Part]:
        pns = list(dict.fromkeys(part_numbers))
        out: dict[str, Part] = {}
        for i in range(0, len(pns), 900):  # SQLite variable limit
            chunk = pns[i : i + 900]
            q = f"SELECT * FROM parts WHERE part_number IN ({','.join('?' * len(chunk))})"
            for row in self._conn.execute(q, chunk):
                out[row["part_number"]] = self._row_to_part(row)
        return out

    def iter_parts(self, with_images_only: bool = False) -> Iterator[Part]:
        q = "SELECT * FROM parts"
        if with_images_only:
            q += " WHERE image_paths != '[]'"
        q += " ORDER BY part_number"
        for row in self._conn.execute(q):
            yield self._row_to_part(row)

    def family(self, family_id: str) -> list[Part]:
        rows = self._conn.execute(
            "SELECT * FROM parts WHERE family_id=? ORDER BY part_number", (family_id,)
        ).fetchall()
        return [self._row_to_part(r) for r in rows]

    PINNED_SCORE = -1e9  # a part-number match: ahead of any keyword match

    def search_text(self, query: str, limit: int = 20) -> list[Part]:
        """Keyword / part-number search (FTS5 when available, LIKE otherwise).

        Exact and prefix part-number matches always come first.
        """
        return [p for p, _ in self.search_text_scored(query, limit)]

    def search_text_scored(self, query: str, limit: int = 20) -> list[tuple[Part, float]]:
        """``search_text`` with each hit's text score: the FTS5 bm25 rank (more negative
        is a stronger match), ``PINNED_SCORE`` for part-number matches, 0 for the LIKE
        fallback, where every hit is as good as any other."""
        query = query.strip()
        if not query:
            return []
        ordered: list[str] = []
        scores: dict[str, float] = {}
        for row in self._conn.execute(
            "SELECT part_number FROM parts WHERE part_number LIKE ? ESCAPE '\\' "
            "ORDER BY part_number LIMIT ?",
            (f"{_like_escape(query.upper())}%", limit),
        ):
            ordered.append(row["part_number"])
            scores[row["part_number"]] = self.PINNED_SCORE
        if len(ordered) < limit:
            safe = " ".join(f'"{tok}"' for tok in query.replace('"', " ").split())
            if self._fts and safe:
                try:
                    rows = self._conn.execute(
                        "SELECT part_number, rank AS score FROM parts_fts WHERE parts_fts MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (safe, limit),
                    ).fetchall()
                except sqlite3.OperationalError:  # FTS syntax we did not anticipate
                    rows = []
            elif self._fts:
                rows = []
            else:
                like = f"%{_like_escape(query)}%"
                rows = self._conn.execute(
                    "SELECT part_number, 0.0 AS score FROM parts "
                    "WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' LIMIT ?",
                    (like, like, limit),
                ).fetchall()
            for r in rows:
                if r["part_number"] not in scores:
                    ordered.append(r["part_number"])
                    scores[r["part_number"]] = float(r["score"] or 0.0)
        parts = self.get_many(ordered[:limit])
        hits = [(parts[pn], scores[pn]) for pn in ordered[:limit] if pn in parts]
        return _promote_exact_values(hits, query)

    def by_category(
        self,
        prefix: list[str],
        limit: int = 60,
        offset: int = 0,
        material: str | None = None,
    ) -> list[Part]:
        """Parts whose category path starts with ``prefix`` (JSON list prefix match),
        optionally only those whose ``material`` attribute is ``material``."""
        mat_sql = ""
        args: list = []
        if material:
            mat_sql = (
                " AND COALESCE(json_extract(attributes, '$.material'), "
                "json_extract(attributes, '$.Material')) = ?"
            )
            args.append(material)
        if not prefix:
            rows = self._conn.execute(
                f"SELECT * FROM parts WHERE 1{mat_sql} ORDER BY part_number LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
            return [self._row_to_part(r) for r in rows]
        head = json.dumps(prefix)[:-1]  # '["A", "B"' matches '["A", "B"]' and '["A", "B", ...'
        # an exact, case-sensitive prefix test as a range the category index can serve
        # (LIKE would treat % and _ in a category name as wildcards and fold ASCII case,
        # disagreeing with the taxonomy counts): every path that starts with '["A", "B", '
        # sorts between that string and the same string with the next byte appended
        deeper = head + ", "
        rows = self._conn.execute(
            "SELECT * FROM parts WHERE (category_path = ? OR "
            "(category_path >= ? AND category_path < ?))"
            f"{mat_sql} ORDER BY part_number LIMIT ? OFFSET ?",
            (head + "]", deeper, deeper + "\x7f", *args, limit, offset),
        ).fetchall()
        return [self._row_to_part(r) for r in rows]

    def count(self, with_images_only: bool = False) -> int:
        q = "SELECT COUNT(*) FROM parts" + (
            " WHERE image_paths != '[]'" if with_images_only else ""
        )
        return int(self._conn.execute(q).fetchone()[0])

    def taxonomy(self) -> Taxonomy:
        tax = Taxonomy()
        for row in self._conn.execute("SELECT category_path FROM parts"):
            tax.add(json.loads(row["category_path"]))
        return tax
