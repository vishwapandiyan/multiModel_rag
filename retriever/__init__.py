"""
Retriever Module

Handles document retrieval techniques including:
- Semantic search
- Relevance checking
- Question processing
- Multi-query expansion
"""

from .retriever import (
    DocumentRetriever,
    Retriever,
    faiss_semantic_search,
    enhanced_semantic_search,
    check_relevance_enhanced,
    process_question_with_relevance_check
)

__all__ = [
    'DocumentRetriever',
    'Retriever',
    'faiss_semantic_search',
    'enhanced_semantic_search',
    'check_relevance_enhanced',
    'process_question_with_relevance_check'
]
