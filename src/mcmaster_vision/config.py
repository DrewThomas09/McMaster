"""Runtime configuration.

Values are resolved in this order (highest priority first):
  1. explicit keyword arguments to ``Settings(...)``
  2. environment variables prefixed with ``MCV_`` (and a ``.env`` file)
  3. a YAML file passed to ``load_settings(path)``
  4. the defaults below
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BackboneName = Literal["hash", "openclip", "dinov2"]
IndexBackend = Literal["numpy", "faiss"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCV_", env_file=".env", extra="ignore")

    # Paths
    data_dir: Path = Path("./data")
    catalog_db: Path = Path("./data/catalog/catalog.sqlite")
    index_dir: Path = Path("./data/index")
    model_dir: Path = Path("./data/models")

    # Embedding backbone
    backbone: BackboneName = "hash"
    backbone_model: str = "ViT-B-16"
    backbone_pretrained: str = "laion2b_s34b_b88k"
    backbone_checkpoint: Path | None = None
    device: str = "auto"
    image_size: int = 224

    # Vector index
    index_backend: IndexBackend = "numpy"
    index_top_k: int = Field(default=50, ge=1, le=1000)

    # Reranking
    rerank_llm_enabled: bool = False
    rerank_llm_model: str = "claude-opus-5"
    rerank_llm_candidates: int = Field(default=8, ge=1, le=20)

    # OCR
    ocr_enabled: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_mb: int = 20

    @property
    def index_path(self) -> Path:
        return self.index_dir / "parts"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.catalog_db.parent, self.index_dir, self.model_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | Path | None = None, **overrides: Any) -> Settings:
    """Build settings from an optional YAML file plus env vars and overrides."""
    file_values: dict[str, Any] = {}
    if config_path:
        with open(config_path, encoding="utf-8") as fh:
            file_values = yaml.safe_load(fh) or {}
    file_values = {k: v for k, v in file_values.items() if k in Settings.model_fields}
    # Env vars beat the YAML file: construct from YAML first, then let pydantic-settings
    # layer env on top by passing YAML values as *defaults* via init kwargs only for keys
    # that are not set in the environment.
    import os

    env_keys = {k[len("MCV_") :].lower() for k in os.environ if k.upper().startswith("MCV_")}
    init_values = {k: v for k, v in file_values.items() if k not in env_keys}
    init_values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**init_values)
