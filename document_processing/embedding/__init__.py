"""
Embedding Generation Module

Handles the creation and management of text embeddings for
vector search and similarity matching.
"""

from .embed_generator import EmbeddingGenerator, create_enhanced_embeddings

__all__ = ['EmbeddingGenerator', 'create_enhanced_embeddings']
