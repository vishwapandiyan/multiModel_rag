"""
Chunk Splitting Module

Handles intelligent text chunking and splitting for optimal
document processing and retrieval.
"""

import re
import logging
from typing import List, Dict, Any, Optional
try:
    from text_extraction.extractor import TextExtractor
except ImportError:
    from ...text_extraction.extractor import TextExtractor

logger = logging.getLogger(__name__)

class ChunkSplitter:
    """Handles intelligent text chunking and splitting"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 300):
        """
        Initialize the chunk splitter
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split into chunks
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
            
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                search_end = min(end + 100, len(text))
                
                # Find the last sentence boundary
                sentence_end = self._find_last_sentence_boundary(
                    text[search_start:search_end]
                )
                
                if sentence_end > 0:
                    end = search_start + sentence_end
                else:
                    # If no sentence boundary found, try to break at word boundary
                    word_break = self._find_last_word_boundary(
                        text[start:end]
                    )
                    if word_break > 0:
                        end = start + word_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Debug: Log chunk types and first few characters
        for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
            chunk_type = type(chunk)
            chunk_preview = str(chunk)[:100] if chunk else "None"
            logger.debug(f"Chunk {i}: type={chunk_type}, preview='{chunk_preview}'")
        
        return chunks
    
    def _find_last_sentence_boundary(self, text: str) -> int:
        """Find the last sentence boundary in the given text"""
        # Common sentence endings
        sentence_endings = ['.', '!', '?', '\n\n']
        
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos > 0:
                return pos + len(ending)
        
        return -1
    
    def _find_last_word_boundary(self, text: str) -> int:
        """Find the last word boundary in the given text"""
        # Look for whitespace or punctuation
        for i in range(len(text) - 1, 0, -1):
            if text[i].isspace() or text[i] in ',;:':
                return i
        
        return -1
    
    def intelligent_chunk_excel_text(self, text: str) -> List[str]:
        """
        Intelligent chunking for Excel text with sheet awareness
        
        Args:
            text: Text extracted from Excel file
            
        Returns:
            List of intelligently chunked text segments
        """
        try:
            if not text:
                return []
            
            # Split by sheet markers
            sheet_pattern = r'Sheet:\s*([^\n]+)'
            sheet_splits = re.split(sheet_pattern, text)
            
            chunks = []
            current_chunk = ""
            
            for i, split in enumerate(sheet_splits):
                if i == 0:  # First part (before first sheet marker)
                    current_chunk = split.strip()
                elif i % 2 == 1:  # Sheet name
                    if current_chunk:
                        chunks.extend(self.chunk_text(current_chunk))
                    current_chunk = f"Sheet: {split}\n"
                else:  # Sheet content
                    current_chunk += split
            
            # Add the last chunk
            if current_chunk:
                chunks.extend(self.chunk_text(current_chunk))
            
            logger.info(f"Intelligently chunked Excel text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in intelligent Excel chunking: {e}")
            return self.chunk_text(text)
    
    def intelligent_chunk_pptx_text(self, text: str) -> List[str]:
        """
        Intelligent chunking for PowerPoint text with slide awareness
        
        Args:
            text: Text extracted from PowerPoint file
            
        Returns:
            List of intelligently chunked text segments
        """
        try:
            if not text:
                return []
            
            # Split by slide markers
            slide_pattern = r'Slide\s+\d+:'
            slide_splits = re.split(slide_pattern, text)
            
            chunks = []
            current_chunk = ""
            
            for i, split in enumerate(slide_splits):
                if i == 0:  # First part (before first slide marker)
                    current_chunk = split.strip()
                else:  # Slide content
                    if current_chunk:
                        chunks.extend(self.chunk_text(current_chunk))
                    current_chunk = f"Slide {i}: {split}"
            
            # Add the last chunk
            if current_chunk:
                chunks.extend(self.chunk_text(current_chunk))
            
            logger.info(f"Intelligently chunked PowerPoint text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in intelligent PowerPoint chunking: {e}")
            return self.chunk_text(text)
    
    def sample_random_pages(self, text: str, num_pages: int = 3) -> str:
        """
        Sample random pages from text for analysis
        
        Args:
            text: Full text content
            num_pages: Number of pages to sample
            
        Returns:
            Sampled text content
        """
        try:
            if not text:
                return ""
            
            # Split into pages (assuming page breaks are marked with \n\n or similar)
            pages = re.split(r'\n\s*\n', text)
            
            if len(pages) <= num_pages:
                return text
        
            # Sample random pages
            import random
            sampled_indices = random.sample(range(len(pages)), min(num_pages, len(pages)))
            sampled_pages = [pages[i] for i in sorted(sampled_indices)]
            
            sampled_text = '\n\n'.join(sampled_pages)
            logger.info(f"Sampled {len(sampled_pages)} pages from text")
            
            return sampled_text
            
        except Exception as e:
            logger.error(f"Error sampling random pages: {e}")
            return text[:1000] if text else ""  # Fallback to first 1000 characters
    
    def process_chunks_by_route(self, chunks: List[str], route: str, url_hash: str) -> List[str]:
        """
        Process chunks based on document processing route
        
        Args:
            chunks: List of text chunks
            route: Processing route identifier
            url_hash: Hash of the document URL
            
        Returns:
            Processed chunks
        """
        try:
            if route == "excel":
                # For Excel files, use intelligent chunking
                full_text = '\n'.join(chunks)
                return self.intelligent_chunk_excel_text(full_text)
            elif route == "pptx":
                # For PowerPoint files, use intelligent chunking
                full_text = '\n'.join(chunks)
                return self.intelligent_chunk_pptx_text(full_text)
            elif route == "pdf":
                # For PDFs, ensure proper chunking
                return [chunk for chunk in chunks if chunk.strip()]
            else:
                # Default processing
                return chunks
                
        except Exception as e:
            logger.error(f"Error processing chunks by route: {e}")
            return chunks

# Utility functions for backward compatibility
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 300) -> List[str]:
    """Utility function to chunk text"""
    splitter = ChunkSplitter(chunk_size, chunk_overlap)
    return splitter.chunk_text(text, chunk_size, chunk_overlap)

def sample_random_pages(text: str, num_pages: int = 3) -> str:
    """Utility function to sample random pages"""
    splitter = ChunkSplitter()
    return splitter.sample_random_pages(text, num_pages)
