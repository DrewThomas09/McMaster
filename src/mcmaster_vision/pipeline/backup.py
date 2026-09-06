"""Backup and restore of everything the system learns at run time.

A deployment accumulates state that is expensive or impossible to recreate: the
ingested catalog, the vector index, fitted calibration, confirmed photos, the
request log and the manifest. ``mcv backup`` bundles all of it into one
``tar.gz`` (with a ``BACKUP.json`` inventory); ``mcv restore`` puts it back.
Caches (thumbnails, recent query photos) are not included: they are rebuilt.
"""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mcmaster_vision import __version__
from mcmaster_vision.config import Settings

INVENTORY = "BACKUP.json"


def _components(settings: Settings) -> dict[str, Path]:
    """Name -> path of every piece of durable state (some may not exist yet)."""
    return {
        "catalog": settings.catalog_db,
        "index": settings.index_path,
        "calibration": settings.model_dir / "calibration.json",
        "queries": settings.queries_dir,
        "logs": settings.data_dir / "logs",
        "manifest": settings.data_dir / "manifest.json",
    }


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _checkpoint_sqlite(db: Path) -> None:
    """Fold the WAL into the main file so the copy is self-contained."""
    try:
        with sqlite3.connect(str(db)) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass


def create_backup(
    settings: Settings, out: str | Path | None = None, *, include_checkpoint: bool = True
) -> Path:
    """Write ``<out>`` (default ``data/backups/mcv-<timestamp>.tar.gz``); returns its path."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out is None:
        out = settings.data_dir / "backups" / f"mcv-{stamp}.tar.gz"
    out = Path(out)
    if out.is_dir() or not out.name.endswith((".tar.gz", ".tgz")):  # a directory
        out = out / f"mcv-{stamp}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    comps = _components(settings)
    if include_checkpoint and settings.backbone_checkpoint:
        ckpt = Path(settings.backbone_checkpoint)
        if ckpt.exists():
            comps["checkpoint"] = ckpt
    inventory: dict = {
        "version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "components": {},
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for name, path in comps.items():
            if not path.exists():
                continue
            if name == "catalog":
                _checkpoint_sqlite(path)
            tar.add(str(path), arcname=name if path.is_dir() else f"{name}/{path.name}")
            inventory["components"][name] = {
                "source": str(path),
                "bytes": _dir_size(path),
                "kind": "dir" if path.is_dir() else "file",
            }
        blob = json.dumps(inventory, indent=2).encode("utf-8")
        info = tarfile.TarInfo(INVENTORY)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    tmp.replace(out)
    return out


def read_inventory(archive: str | Path) -> dict:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            fh = tar.extractfile(INVENTORY)
        except KeyError as e:
            raise ValueError(f"{archive} is not a McMaster-Vision backup (no {INVENTORY})") from e
        if fh is None:
            raise ValueError(f"{archive}: bad {INVENTORY}")
        return json.loads(fh.read().decode("utf-8"))


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
        return
    dest_r = dest.resolve()
    for m in tar.getmembers():  # pragma: no cover - Python < 3.12
        target = (dest / m.name).resolve()
        if dest_r not in target.parents and target != dest_r:
            raise ValueError(f"unsafe path in archive: {m.name}")
    tar.extractall(dest)


def restore_backup(
    settings: Settings, archive: str | Path, *, components: list[str] | None = None
) -> dict:
    """Put the archived state back where ``settings`` expects it. Existing components are
    replaced atomically (extract to a temp dir, then swap); everything else is untouched.
    Returns the inventory of what was restored."""
    inventory = read_inventory(archive)
    wanted = set(components or inventory["components"].keys())
    unknown = wanted - set(inventory["components"])
    if unknown:
        raise ValueError(f"backup has no component(s): {', '.join(sorted(unknown))}")
    comps = _components(settings)
    restored: dict = {}
    with tempfile.TemporaryDirectory(prefix="mcv-restore-") as tmp:
        tmpd = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, tmpd)
        for name in sorted(wanted):
            meta = inventory["components"][name]
            src_root = tmpd / name
            if name == "checkpoint":
                dest = Path(settings.backbone_checkpoint or settings.model_dir / "checkpoint.pt")
            else:
                dest = comps[name]
            src = src_root if meta["kind"] == "dir" else src_root / Path(meta["source"]).name
            if not src.exists():
                raise ValueError(f"backup is missing {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                old = dest.with_name(dest.name + ".old")
                if old.exists():
                    shutil.rmtree(old) if old.is_dir() else old.unlink()
                dest.rename(old)
            else:
                old = None
            if name == "catalog":  # drop stale WAL/SHM next to the restored database
                for side in (
                    dest.with_name(dest.name + "-wal"),
                    dest.with_name(dest.name + "-shm"),
                ):
                    side.unlink(missing_ok=True)
            shutil.move(str(src), str(dest))
            if name == "index":  # a fresh mtime so a running API notices the swap
                meta = dest / "meta.json"
                if meta.exists():
                    meta.touch()
            if old is not None:
                shutil.rmtree(old) if old.is_dir() else old.unlink()
            restored[name] = str(dest)
    return {
        "version": inventory.get("version"),
        "created_at": inventory.get("created_at"),
        "restored": restored,
    }


def storage_status(settings: Settings) -> dict:
    """Sizes and ages of every durable component plus the newest backup (for
    ``GET /status``, the dashboard and ``mcv doctor``)."""
    comps = {}
    total = 0
    for name, path in _components(settings).items():
        if path.exists():
            size = _dir_size(path)
            total += size
            mtime = (
                max((f.stat().st_mtime for f in path.rglob("*") if f.is_file()), default=0)
                if path.is_dir()
                else path.stat().st_mtime
            )
            comps[name] = {
                "path": str(path),
                "bytes": size,
                "updated_at": datetime.fromtimestamp(mtime, timezone.utc).isoformat(
                    timespec="seconds"
                )
                if mtime
                else None,
            }
        else:
            comps[name] = {"path": str(path), "bytes": 0, "updated_at": None}
    root = settings.data_dir / "backups"
    backups = (
        sorted(root.glob("*.tar.gz"), key=lambda f: f.stat().st_mtime) if root.exists() else []
    )
    last = backups[-1] if backups else None
    newest_change = max((c["updated_at"] or "" for c in comps.values()), default="")
    last_at = (
        datetime.fromtimestamp(last.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        if last
        else None
    )
    return {
        "components": comps,
        "bytes_total": total,
        "backups": len(backups),
        "last_backup": {"path": str(last), "bytes": last.stat().st_size, "created_at": last_at}
        if last
        else None,
        # something changed since the newest backup (or there is none)
        "backup_stale": bool(newest_change and (last_at is None or newest_change > last_at)),
    }
