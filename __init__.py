"""
HackRx Final Pipeline

A comprehensive, modular document processing and challenge-solving system.
"""

__version__ = "1.0.0"
__author__ = "HackRx Team"

# Import main components for easy access
from .main import (
    initialize_components,
    process_document_with_pipeline,
    app
)

# Import core modules
from . import text_extraction
from . import document_processing
from . import vectorstore
from . import retriever
from . import agent

__all__ = [
    'initialize_components',
    'process_document_with_pipeline', 
    'app',
    'text_extraction',
    'document_processing',
    'vectorstore',
    'retriever',
    'agent'
]
