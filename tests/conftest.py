from __future__ import annotations

import pytest

from mcmaster_vision.catalog import CatalogStore, ingest
from mcmaster_vision.data import SyntheticCatalog
from mcmaster_vision.index import build_index
from mcmaster_vision.models import HashBackbone, PartEmbedder
from mcmaster_vision.pipeline import Identifier


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("demo")


@pytest.fixture(scope="session")
def jsonl_path(demo_dir):
    path = demo_dir / "parts.jsonl"
    SyntheticCatalog(n_parts=40, images_per_part=2, seed=3).write_jsonl(demo_dir / "images", path)
    return path


@pytest.fixture(scope="session")
def store(demo_dir, jsonl_path):
    st = CatalogStore(demo_dir / "catalog.sqlite")
    ingest(jsonl_path, st)
    yield st
    st.close()


@pytest.fixture(scope="session")
def embedder():
    return PartEmbedder(HashBackbone())


@pytest.fixture(scope="session")
def index(store, embedder, demo_dir):
    return build_index(store, embedder, "numpy", out_path=demo_dir / "index")


@pytest.fixture(scope="session")
def identifier(store, index, embedder):
    return Identifier(store, index, embedder, top_k=20)
