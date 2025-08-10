"""
Document Processing Module

This module handles the preprocessing of extracted text including:
- Text chunking and splitting
- Embedding generation
- Metadata extraction
- Logging and monitoring

The module provides a pipeline for preparing documents for vector storage
and retrieval.
"""

from .chunks.chunk_splitter import ChunkSplitter, chunk_text
from .embedding.embed_generator import EmbeddingGenerator, create_enhanced_embeddings
from .metadata.metadata_extractor import MetadataExtractor

__all__ = [
    'ChunkSplitter',
    'chunk_text',
    'EmbeddingGenerator',
    'create_enhanced_embeddings',
    'MetadataExtractor'
]
