"""
pNSFWMedia - NSFW Detection Model

A two-stage NSFW image classifier inspired by Twitter's pNSFWMedia.
"""

from .extract_embeddings import CLIPEmbeddingExtractor
from .extract_embeddings_nudenet import NudeNetEmbeddingExtractor
from .inference import NSFWClassifier

__version__ = "1.0.0"
__all__ = ["CLIPEmbeddingExtractor", "NudeNetEmbeddingExtractor", "NSFWClassifier"]
