"""
Metadata Extraction Module

Handles the extraction and management of document metadata
for improved search and retrieval capabilities.
"""

import os
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import mimetypes
from pathlib import Path

try:
    from redis_integration.metadata_cache import MetadataCache
    REDIS_METADATA_CACHE_AVAILABLE = True
except ImportError:
    REDIS_METADATA_CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Handles document metadata extraction and management"""
    
    def __init__(self, base_path: str = "document_processing/metadata"):
        """
        Initialize the metadata extractor
        
        Args:
            base_path: Base path for storing metadata files
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        # Initialize Redis cache if available
        self.cache = None
        if REDIS_METADATA_CACHE_AVAILABLE:
            try:
                self.cache = MetadataCache()
                logger.info("Redis metadata cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis metadata cache: {e}")
    
    def extract_basic_metadata(self, file_path: str, file_content: str = None) -> Dict[str, Any]:
        """
        Extract basic metadata from a file
        
        Args:
            file_path: Path to the file
            file_content: Optional file content for additional analysis
            
        Returns:
            Dictionary containing basic metadata
        """
        try:
            file_path = Path(file_path)
            
            # Basic file metadata
            metadata = {
                'filename': file_path.name,
                'file_extension': file_path.suffix.lower(),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'file_type': self._detect_file_type(file_path),
                'created_time': self._get_file_time(file_path, 'created'),
                'modified_time': self._get_file_time(file_path, 'modified'),
                'accessed_time': self._get_file_time(file_path, 'accessed'),
                'content_hash': self._generate_content_hash(file_content) if file_content else None,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            # Add content-based metadata if available
            if file_content:
                metadata.update(self._extract_content_metadata(file_content))
            
            logger.info(f"Extracted metadata for {file_path.name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting basic metadata from {file_path}: {e}")
            return {}
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type based on extension and content"""
        try:
            # Try to detect from extension first
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                return mime_type
            
            # Fallback to extension-based detection
            extension = file_path.suffix.lower()
            type_mapping = {
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.txt': 'text/plain',
                '.html': 'text/html',
                '.zip': 'application/zip'
            }
            
            return type_mapping.get(extension, 'application/octet-stream')
            
        except Exception as e:
            logger.error(f"Error detecting file type: {e}")
            return 'application/octet-stream'
    
    def _get_file_time(self, file_path: Path, time_type: str) -> Optional[str]:
        """Get file time information"""
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            if time_type == 'created':
                timestamp = stat.st_ctime
            elif time_type == 'modified':
                timestamp = stat.st_mtime
            elif time_type == 'accessed':
                timestamp = stat.st_atime
            else:
                return None
            
            return datetime.fromtimestamp(timestamp).isoformat()
            
        except Exception as e:
            logger.error(f"Error getting file time {time_type}: {e}")
            return None
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash of content for change detection"""
        try:
            if not content:
                return ""
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Error generating content hash: {e}")
            return ""
    
    def _extract_content_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content analysis"""
        try:
            metadata = {
                'content_length': len(content),
                'word_count': len(content.split()),
                'line_count': len(content.splitlines()),
                'has_tables': '|' in content or '\t' in content,
                'has_images': '[image]' in content.lower() or 'img' in content.lower(),
                'language': self._detect_language(content),
                'complexity_score': self._calculate_complexity_score(content)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting content metadata: {e}")
            return {}
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection based on common words"""
        try:
            # Simple English detection based on common words
            english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            content_lower = content.lower()
            
            english_count = sum(1 for word in english_words if word in content_lower)
            if english_count > 3:
                return 'en'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'unknown'
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate a simple complexity score"""
        try:
            if not content:
                return 0.0
        
            # Simple complexity based on sentence length and vocabulary
            sentences = content.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            
            # Normalize to 0-1 scale
            complexity = min(avg_sentence_length / 20.0, 1.0)
            
            return round(complexity, 2)
            
        except Exception as e:
            logger.error(f"Error calculating complexity score: {e}")
            return 0.0
    
    def save_metadata(self, url_hash: str, metadata: Dict[str, Any]) -> bool:
        """
        Save metadata to file and cache
        
        Args:
            url_hash: Hash of the document URL
            metadata: Metadata dictionary to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Save to cache first
            if self.cache:
                self.cache.set_metadata(url_hash, metadata)
            
            # Save to file as backup
            metadata_file = os.path.join(self.base_path, f"{url_hash}_metadata.json")
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved metadata for {url_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metadata for {url_hash}: {e}")
            return False
    
    def load_metadata(self, url_hash: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata from cache or file
        
        Args:
            url_hash: Hash of the document URL
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            # Try cache first
            if self.cache:
                cached_metadata = self.cache.get_metadata(url_hash)
                if cached_metadata:
                    logger.debug(f"Retrieved metadata from cache for {url_hash}")
                    return cached_metadata
            
            # Fallback to file
            metadata_file = os.path.join(self.base_path, f"{url_hash}_metadata.json")
            
            if not os.path.exists(metadata_file):
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Cache the file-loaded metadata for future use
            if self.cache and metadata:
                self.cache.set_metadata(url_hash, metadata)
            
            logger.info(f"Loaded metadata for {url_hash}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata for {url_hash}: {e}")
            return None
    
    def update_metadata(self, url_hash: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Update existing metadata
        
        Args:
            url_hash: Hash of the document URL
            new_metadata: New metadata to merge
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing metadata
            existing_metadata = self.load_metadata(url_hash) or {}
            
            # Merge with new metadata
            existing_metadata.update(new_metadata)
            existing_metadata['last_updated'] = datetime.now().isoformat()
            
            # Save updated metadata
            return self.save_metadata(url_hash, existing_metadata)
            
        except Exception as e:
            logger.error(f"Error updating metadata for {url_hash}: {e}")
            return False
    
    def list_all_metadata(self) -> List[str]:
        """List all available metadata files"""
        try:
            metadata_files = []
            for file in os.listdir(self.base_path):
                if file.endswith('_metadata.json'):
                    url_hash = file.replace('_metadata.json', '')
                    metadata_files.append(url_hash)
            
            return metadata_files
            
        except Exception as e:
            logger.error(f"Error listing metadata files: {e}")
            return []
    
    def delete_metadata(self, url_hash: str) -> bool:
        """
        Delete metadata file
        
        Args:
            url_hash: Hash of the document URL
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metadata_file = os.path.join(self.base_path, f"{url_hash}_metadata.json")
            
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"Deleted metadata for {url_hash}")
            return True
            
            return False
    
        except Exception as e:
            logger.error(f"Error deleting metadata for {url_hash}: {e}")
            return False
