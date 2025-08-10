"""Embedding Cache Layer"""

import hashlib
import logging
import numpy as np
from typing import List, Optional
from .redis_client import get_redis_client

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Cache for text embeddings"""
    
    def __init__(self, default_ttl: int = 86400):  # 24 hours default
        self.redis_client = get_redis_client()
        self.default_ttl = default_ttl
        self.prefix = "embedding"
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _generate_texts_hash(self, texts: List[str]) -> str:
        """Generate hash for list of texts"""
        combined_text = "\n".join(sorted(texts))  # Sort for consistency
        return hashlib.sha256(combined_text.encode('utf-8')).hexdigest()[:16]
    
    def get_single_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding for single text"""
        if not self.redis_client.is_available():
            return None
        
        try:
            text_hash = self._generate_text_hash(text)
            cache_key = self.redis_client.generate_key(
                self.prefix, "single", model_name, text_hash
            )
            
            cached_embedding = self.redis_client.get_pickle(cache_key)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for single embedding: {text_hash}")
                return cached_embedding
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached embedding: {e}")
            return None
    
    def set_single_embedding(self, text: str, model_name: str, 
                           embedding: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache embedding for single text"""
        if not self.redis_client.is_available():
            return False
        
        try:
            text_hash = self._generate_text_hash(text)
            cache_key = self.redis_client.generate_key(
                self.prefix, "single", model_name, text_hash
            )
            
            ttl = ttl or self.default_ttl
            success = self.redis_client.set_pickle(cache_key, embedding, ex=ttl)
            
            if success:
                logger.debug(f"Cached single embedding: {text_hash}")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching single embedding: {e}")
            return False
    
    def get_batch_embeddings(self, texts: List[str], model_name: str) -> Optional[np.ndarray]:
        """Get cached embeddings for batch of texts"""
        if not self.redis_client.is_available():
            return None
        
        try:
            texts_hash = self._generate_texts_hash(texts)
            cache_key = self.redis_client.generate_key(
                self.prefix, "batch", model_name, texts_hash
            )
            
            cached_embeddings = self.redis_client.get_pickle(cache_key)
            if cached_embeddings is not None:
                logger.debug(f"Cache hit for batch embeddings: {texts_hash}")
                return cached_embeddings
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached batch embeddings: {e}")
            return None
    
    def set_batch_embeddings(self, texts: List[str], model_name: str, 
                           embeddings: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache embeddings for batch of texts"""
        if not self.redis_client.is_available():
            return False
        
        try:
            texts_hash = self._generate_texts_hash(texts)
            cache_key = self.redis_client.generate_key(
                self.prefix, "batch", model_name, texts_hash
            )
            
            ttl = ttl or self.default_ttl
            success = self.redis_client.set_pickle(cache_key, embeddings, ex=ttl)
            
            if success:
                logger.debug(f"Cached batch embeddings: {texts_hash} ({len(texts)} texts)")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching batch embeddings: {e}")
            return False
    
    def get_partial_batch(self, texts: List[str], model_name: str) -> tuple[List[int], List[np.ndarray]]:
        """Get cached embeddings for individual texts in batch"""
        cached_indices = []
        cached_embeddings = []
        
        if not self.redis_client.is_available():
            return cached_indices, cached_embeddings
        
        for i, text in enumerate(texts):
            embedding = self.get_single_embedding(text, model_name)
            if embedding is not None:
                cached_indices.append(i)
                cached_embeddings.append(embedding)
        
        return cached_indices, cached_embeddings
    
    def cache_individual_embeddings(self, texts: List[str], model_name: str, 
                                  embeddings: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache individual embeddings from a batch"""
        if not self.redis_client.is_available():
            return False
        
        success_count = 0
        for text, embedding in zip(texts, embeddings):
            if self.set_single_embedding(text, model_name, embedding, ttl):
                success_count += 1
        
        logger.debug(f"Cached {success_count}/{len(texts)} individual embeddings")
        return success_count > 0
    
    def clear_cache(self, model_name: Optional[str] = None) -> bool:
        """Clear embedding cache"""
        if not self.redis_client.is_available():
            return False
        
        try:
            # This is a simplified implementation
            # In production, you'd want to use Redis SCAN to find and delete keys
            logger.info(f"Cache clearing not fully implemented. Manual Redis cleanup needed.")
            return True
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
            return False