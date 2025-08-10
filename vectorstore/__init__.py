"""
Vector Store Module

Handles vector database operations including:
- FAISS index creation and management
- Vector storage and retrieval
- Index persistence and loading
- Vector similarity operations
"""

from .vector_store import (
    VectorStore,
    create_faiss_index,
    save_faiss_index,
    load_faiss_index
)

__all__ = [
    'VectorStore',
    'create_faiss_index',
    'save_faiss_index',
    'load_faiss_index'
]
