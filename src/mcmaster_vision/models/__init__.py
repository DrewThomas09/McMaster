"""Embedding models: backbones, projection heads, and metric-learning losses."""

from mcmaster_vision.models.backbone import Backbone, HashBackbone, load_backbone
from mcmaster_vision.models.embedder import PartEmbedder

__all__ = ["Backbone", "HashBackbone", "PartEmbedder", "load_backbone"]
