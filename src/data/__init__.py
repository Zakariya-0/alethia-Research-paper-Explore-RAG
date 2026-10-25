"""
Module data : gestion de la collecte, du découpage (chunking) et de l'indexation.
"""

from .collector import ArxivCollector, PaperMetadata
from .chunker import TextChunker, PaperChunk, clean_scientific_text

__all__ = [
    "ArxivCollector",
    "PaperMetadata",
    "TextChunker",
    "PaperChunk",
    "clean_scientific_text",
]
