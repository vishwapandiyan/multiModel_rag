"""
Text Extraction Module

This module handles text extraction from various document formats including:
- PDF files (with OCR support)
- Word documents
- Excel spreadsheets
- PowerPoint presentations
- Images
- HTML files
- ZIP archives
- Binary files

The module provides a unified interface for extracting text content
regardless of the source format.
"""

from .extractor import (
    TextExtractor,
    extract_text_from_pdf,
    extract_text_from_word,
    extract_text_from_excel,
    extract_text_from_pptx,
    extract_text_from_image,
    extract_text_from_html,
    extract_text_from_zip,
    extract_text_from_binary,
    detect_file_type_from_content
)

__all__ = [
    'TextExtractor',
    'extract_text_from_pdf',
    'extract_text_from_word',
    'extract_text_from_excel',
    'extract_text_from_pptx',
    'extract_text_from_image',
    'extract_text_from_html',
    'extract_text_from_zip',
    'extract_text_from_binary',
    'detect_file_type_from_content'
]
