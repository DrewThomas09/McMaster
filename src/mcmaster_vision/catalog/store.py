"""SQLite-backed catalog store.

SQLite is deliberately chosen for the skeleton: a 700k-row parts table with a JSON
attributes column is a few hundred MB and needs no server. Swap for Postgres by
re-implementing this class; nothing else in the system touches SQL.
"""

from __future__ import annotations

import json
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
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS parts_fts USING fts5(
    part_number, name, category, description, attributes
);
"""


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
    def upsert(self, parts: Iterable[Part], batch_size: int = 5000) -> int:
        n = 0
        batch: list[Part] = []
        for part in parts:
            batch.append(part)
            if len(batch) >= batch_size:
                n += self._write_batch(batch)
                batch = []
        if batch:
            n += self._write_batch(batch)
        return n

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
                self._conn.executemany(
                    "DELETE FROM parts_fts WHERE part_number = ?", [(p.part_number,) for p in batch]
                )
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
        return len(batch)

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

    def search_text(self, query: str, limit: int = 20) -> list[Part]:
        """Keyword / part-number search (FTS5 when available, LIKE otherwise).

        Exact and prefix part-number matches always come first.
        """
        query = query.strip()
        if not query:
            return []
        ordered: list[str] = []
        for row in self._conn.execute(
            "SELECT part_number FROM parts WHERE part_number LIKE ? ORDER BY part_number LIMIT ?",
            (f"{query.upper()}%", limit),
        ):
            ordered.append(row["part_number"])
        if len(ordered) < limit:
            if self._fts:
                safe = " ".join(f'"{tok}"' for tok in query.replace('"', " ").split())
                rows = self._conn.execute(
                    "SELECT part_number FROM parts_fts WHERE parts_fts MATCH ? ORDER BY rank LIMIT ?",
                    (safe, limit),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = self._conn.execute(
                    "SELECT part_number FROM parts WHERE name LIKE ? OR description LIKE ? LIMIT ?",
                    (like, like, limit),
                ).fetchall()
            for r in rows:
                if r["part_number"] not in ordered:
                    ordered.append(r["part_number"])
        parts = self.get_many(ordered[:limit])
        return [parts[pn] for pn in ordered[:limit] if pn in parts]

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
