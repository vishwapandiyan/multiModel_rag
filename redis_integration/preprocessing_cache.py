"""
Preprocessing Pipeline Cache

Provides Redis-based caching for preprocessing artifacts like chunked text,
extracted text, and processing results.
"""

import hashlib
import logging
from typing import List, Optional, Dict, Any
from .redis_client import get_redis_client

logger = logging.getLogger(__name__)

class PreprocessingCache:
    """Cache for preprocessing pipeline artifacts"""
    
    def __init__(self, default_ttl: int = 86400):  # 24 hours default
        self.redis_client = get_redis_client()
        self.default_ttl = default_ttl
        self.text_prefix = "extracted_text"
        self.chunks_prefix = "text_chunks"
        self.processing_prefix = "processing_result"
    
    def _generate_url_hash(self, url: str) -> str:
        """Generate hash for URL"""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def get_extracted_text(self, url: str) -> Optional[str]:
        """Get cached extracted text for URL"""
        if not self.redis_client.is_available():
            return None
        
        try:
            url_hash = self._generate_url_hash(url)
            cache_key = self.redis_client.generate_key(self.text_prefix, url_hash)
            
            cached_data = self.redis_client.get_json(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for extracted text: {url_hash}")
                return cached_data.get('text')
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached extracted text: {e}")
            return None
    
    def set_extracted_text(self, url: str, text: str, ttl: Optional[int] = None) -> bool:
        """Cache extracted text for URL"""
        if not self.redis_client.is_available():
            return False
        
        try:
            url_hash = self._generate_url_hash(url)
            cache_key = self.redis_client.generate_key(self.text_prefix, url_hash)
            
            cache_data = {
                'text': text,
                'url': url,
                'url_hash': url_hash,
                'text_length': len(text),
                'cached_at': self.redis_client.client.time()[0] if self.redis_client.is_available() else None
            }
            
            ttl = ttl or self.default_ttl
            success = self.redis_client.set_json(cache_key, cache_data, ex=ttl)
            
            if success:
                logger.debug(f"Cached extracted text: {url_hash} ({len(text)} chars)")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching extracted text: {e}")
            return False
    
    def get_text_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> Optional[List[str]]:
        """Get cached text chunks"""
        if not self.redis_client.is_available():
            return None
        
        try:
            # Create cache key based on text content and chunking parameters
            text_hash = self._generate_text_hash(text)
            cache_key = self.redis_client.generate_key(
                self.chunks_prefix, text_hash, str(chunk_size), str(chunk_overlap)
            )
            
            cached_data = self.redis_client.get_json(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for text chunks: {text_hash}")
                return cached_data.get('chunks')
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached text chunks: {e}")
            return None
    
    def set_text_chunks(self, text: str, chunks: List[str], chunk_size: int, 
                       chunk_overlap: int, ttl: Optional[int] = None) -> bool:
        """Cache text chunks"""
        if not self.redis_client.is_available():
            return False
        
        try:
            text_hash = self._generate_text_hash(text)
            cache_key = self.redis_client.generate_key(
                self.chunks_prefix, text_hash, str(chunk_size), str(chunk_overlap)
            )
            
            cache_data = {
                'chunks': chunks,
                'text_hash': text_hash,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'num_chunks': len(chunks),
                'cached_at': self.redis_client.client.time()[0] if self.redis_client.is_available() else None
            }
            
            ttl = ttl or self.default_ttl
            success = self.redis_client.set_json(cache_key, cache_data, ex=ttl)
            
            if success:
                logger.debug(f"Cached text chunks: {text_hash} ({len(chunks)} chunks)")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching text chunks: {e}")
            return False
    
    def get_processing_result(self, url: str, questions: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached processing result"""
        if not self.redis_client.is_available():
            return None
        
        try:
            # Create cache key based on URL and questions
            url_hash = self._generate_url_hash(url)
            questions_hash = hashlib.sha256(
                "|".join(sorted(questions)).encode('utf-8')
            ).hexdigest()[:16]
            
            cache_key = self.redis_client.generate_key(
                self.processing_prefix, url_hash, questions_hash
            )
            
            cached_result = self.redis_client.get_json(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for processing result: {url_hash}:{questions_hash}")
                return cached_result
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached processing result: {e}")
            return None
    
    def set_processing_result(self, url: str, questions: List[str], result: Dict[str, Any],
                            ttl: Optional[int] = None) -> bool:
        """Cache processing result"""
        if not self.redis_client.is_available():
            return False
        
        try:
            url_hash = self._generate_url_hash(url)
            questions_hash = hashlib.sha256(
                "|".join(sorted(questions)).encode('utf-8')
            ).hexdigest()[:16]
            
            cache_key = self.redis_client.generate_key(
                self.processing_prefix, url_hash, questions_hash
            )
            
            # Add cache metadata
            cache_data = result.copy()
            cache_data['cache_metadata'] = {
                'url_hash': url_hash,
                'questions_hash': questions_hash,
                'num_questions': len(questions),
                'cached_at': self.redis_client.client.time()[0] if self.redis_client.is_available() else None
            }
            
            ttl = ttl or self.default_ttl
            success = self.redis_client.set_json(cache_key, cache_data, ex=ttl)
            
            if success:
                logger.debug(f"Cached processing result: {url_hash}:{questions_hash}")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching processing result: {e}")
            return False
    
    def get_puzzle_detection_result(self, document_text: str, questions: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Get cached puzzle detection result"""
        if not self.redis_client.is_available():
            return None
        
        try:
            text_hash = self._generate_text_hash(document_text)
            questions_hash = "none"
            if questions:
                questions_hash = hashlib.sha256(
                    "|".join(sorted(questions)).encode('utf-8')
                ).hexdigest()[:16]
            
            cache_key = self.redis_client.generate_key(
                "puzzle_detection", text_hash, questions_hash
            )
            
            cached_result = self.redis_client.get_json(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for puzzle detection: {text_hash}")
                return cached_result
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached puzzle detection result: {e}")
            return None
    
    def set_puzzle_detection_result(self, document_text: str, result: Dict[str, Any],
                                  questions: Optional[List[str]] = None, ttl: Optional[int] = None) -> bool:
        """Cache puzzle detection result"""
        if not self.redis_client.is_available():
            return False
        
        try:
            text_hash = self._generate_text_hash(document_text)
            questions_hash = "none"
            if questions:
                questions_hash = hashlib.sha256(
                    "|".join(sorted(questions)).encode('utf-8')
                ).hexdigest()[:16]
            
            cache_key = self.redis_client.generate_key(
                "puzzle_detection", text_hash, questions_hash
            )
            
            ttl = ttl or self.default_ttl
            success = self.redis_client.set_json(cache_key, result, ex=ttl)
            
            if success:
                logger.debug(f"Cached puzzle detection result: {text_hash}")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching puzzle detection result: {e}")
            return False
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """Clear preprocessing cache"""
        if not self.redis_client.is_available():
            return False
        
        try:
            # This is a simplified implementation
            # In production, you'd want to use Redis SCAN to find and delete keys
            logger.info(f"Preprocessing cache clearing not fully implemented. Manual Redis cleanup needed.")
            return True
        except Exception as e:
            logger.warning(f"Error clearing preprocessing cache: {e}")
            return False