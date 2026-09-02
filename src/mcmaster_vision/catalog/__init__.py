"""Catalog layer: taxonomy, data sources, persistent store, and ingestion."""

from mcmaster_vision.catalog.ingest import ingest
from mcmaster_vision.catalog.sources import (
    CatalogSource,
    CSVSource,
    DirectorySource,
    JSONLSource,
    McMasterApiSource,
    open_source,
)
from mcmaster_vision.catalog.store import CatalogStore
from mcmaster_vision.catalog.taxonomy import TOP_LEVEL_CATEGORIES, Taxonomy

__all__ = [
    "CatalogSource",
    "CSVSource",
    "CatalogStore",
    "DirectorySource",
    "JSONLSource",
    "McMasterApiSource",
    "TOP_LEVEL_CATEGORIES",
    "Taxonomy",
    "ingest",
    "open_source",
]
