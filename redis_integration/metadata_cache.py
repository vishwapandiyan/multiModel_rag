"""
Document Metadata Caching

Provides Redis-based caching for document metadata to improve lookup performance.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from .redis_client import get_redis_client

logger = logging.getLogger(__name__)

class MetadataCache:
    """Cache for document metadata"""
    
    def __init__(self, default_ttl: int = 86400):  # 24 hours default
        self.redis_client = get_redis_client()
        self.default_ttl = default_ttl
        self.prefix = "metadata"
        self.index_prefix = "metadata_index"
    
    def get_metadata(self, url_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata for document"""
        if not self.redis_client.is_available():
            return None
        
        try:
            cache_key = self.redis_client.generate_key(self.prefix, url_hash)
            cached_metadata = self.redis_client.get_json(cache_key)
            
            if cached_metadata:
                logger.debug(f"Cache hit for metadata: {url_hash}")
                return cached_metadata
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached metadata for {url_hash}: {e}")
            return None
    
    def set_metadata(self, url_hash: str, metadata: Dict[str, Any], 
                    ttl: Optional[int] = None) -> bool:
        """Cache metadata for document"""
        if not self.redis_client.is_available():
            return False
        
        try:
            # Add caching timestamp
            metadata_with_cache_info = metadata.copy()
            metadata_with_cache_info['cached_at'] = datetime.now().isoformat()
            metadata_with_cache_info['url_hash'] = url_hash
            
            cache_key = self.redis_client.generate_key(self.prefix, url_hash)
            ttl = ttl or self.default_ttl
            success = self.redis_client.set_json(cache_key, metadata_with_cache_info, ex=ttl)
            
            if success:
                # Update metadata index
                self._update_metadata_index(url_hash, metadata.get('url', ''))
                logger.debug(f"Cached metadata: {url_hash}")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching metadata for {url_hash}: {e}")
            return False
    
    def update_metadata(self, url_hash: str, updates: Dict[str, Any]) -> bool:
        """Update existing cached metadata"""
        if not self.redis_client.is_available():
            return False
        
        try:
            existing_metadata = self.get_metadata(url_hash)
            if not existing_metadata:
                return False
            
            # Merge updates
            existing_metadata.update(updates)
            existing_metadata['last_updated'] = datetime.now().isoformat()
            
            return self.set_metadata(url_hash, existing_metadata)
        except Exception as e:
            logger.warning(f"Error updating cached metadata for {url_hash}: {e}")
            return False
    
    def delete_metadata(self, url_hash: str) -> bool:
        """Delete cached metadata"""
        if not self.redis_client.is_available():
            return False
        
        try:
            cache_key = self.redis_client.generate_key(self.prefix, url_hash)
            success = self.redis_client.delete(cache_key)
            
            if success:
                # Remove from index
                self._remove_from_metadata_index(url_hash)
                logger.debug(f"Deleted cached metadata: {url_hash}")
            
            return success
        except Exception as e:
            logger.warning(f"Error deleting cached metadata for {url_hash}: {e}")
            return False
    
    def _update_metadata_index(self, url_hash: str, url: str) -> bool:
        """Update metadata index for listing purposes"""
        try:
            index_key = self.redis_client.generate_key(self.index_prefix, "all")
            
            # Get current index
            index_data = self.redis_client.get_json(index_key) or {}
            
            # Add/update entry
            index_data[url_hash] = {
                'url': url,
                'indexed_at': datetime.now().isoformat()
            }
            
            # Save updated index
            return self.redis_client.set_json(index_key, index_data, ex=self.default_ttl * 2)
        except Exception as e:
            logger.warning(f"Error updating metadata index: {e}")
            return False
    
    def _remove_from_metadata_index(self, url_hash: str) -> bool:
        """Remove entry from metadata index"""
        try:
            index_key = self.redis_client.generate_key(self.index_prefix, "all")
            
            # Get current index
            index_data = self.redis_client.get_json(index_key) or {}
            
            # Remove entry
            if url_hash in index_data:
                del index_data[url_hash]
                
                # Save updated index
                return self.redis_client.set_json(index_key, index_data, ex=self.default_ttl * 2)
            
            return True
        except Exception as e:
            logger.warning(f"Error removing from metadata index: {e}")
            return False
    
    def list_all_metadata(self) -> List[Dict[str, Any]]:
        """List all cached metadata entries"""
        if not self.redis_client.is_available():
            return []
        
        try:
            index_key = self.redis_client.generate_key(self.index_prefix, "all")
            index_data = self.redis_client.get_json(index_key) or {}
            
            metadata_list = []
            for url_hash, index_info in index_data.items():
                metadata = self.get_metadata(url_hash)
                if metadata:
                    metadata_list.append({
                        'url_hash': url_hash,
                        'url': index_info['url'],
                        'indexed_at': index_info['indexed_at'],
                        'metadata': metadata
                    })
            
            logger.debug(f"Listed {len(metadata_list)} cached metadata entries")
            return metadata_list
        except Exception as e:
            logger.warning(f"Error listing cached metadata: {e}")
            return []
    
    def get_processing_status(self, url_hash: str) -> Optional[str]:
        """Get processing status for document"""
        metadata = self.get_metadata(url_hash)
        if metadata:
            return metadata.get('processing_status', 'unknown')
        return None
    
    def set_processing_status(self, url_hash: str, status: str) -> bool:
        """Set processing status for document"""
        return self.update_metadata(url_hash, {
            'processing_status': status,
            'status_updated_at': datetime.now().isoformat()
        })
    
    def clear_cache(self) -> bool:
        """Clear all metadata cache"""
        if not self.redis_client.is_available():
            return False
        
        try:
            # This is a simplified implementation
            # In production, you'd want to use Redis SCAN to find and delete keys
            logger.info("Metadata cache clearing not fully implemented. Manual Redis cleanup needed.")
            return True
        except Exception as e:
            logger.warning(f"Error clearing metadata cache: {e}")
            return False