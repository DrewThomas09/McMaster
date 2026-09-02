"""Inference pipeline: preprocess -> embed -> retrieve -> rerank -> calibrate."""

from mcmaster_vision.pipeline.identify import Identifier, load_identifier

__all__ = ["Identifier", "load_identifier"]
