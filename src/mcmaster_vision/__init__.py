"""McMaster-Vision: identify a McMaster-Carr part number from a photograph.

The system is a retrieval pipeline rather than a 700,000-way classifier:

    photo -> preprocess -> embed -> vector search (top-K) -> rerank -> calibrated answer

See ARCHITECTURE.md for the full design.
"""

__version__ = "0.3.0"
