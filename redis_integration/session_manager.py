"""
Session Management & Request Caching

Provides Redis-based session management and request caching capabilities.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .redis_client import get_redis_client

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages user sessions and request caching"""
    
    def __init__(self, session_ttl: int = 3600, request_cache_ttl: int = 1800):
        self.redis_client = get_redis_client()
        self.session_ttl = session_ttl  # 1 hour default
        self.request_cache_ttl = request_cache_ttl  # 30 minutes default
        self.session_prefix = "session"
        self.request_prefix = "request"
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'request_count': 0,
            'processed_documents': [],
            'context': {}
        }
        
        if self.redis_client.is_available():
            cache_key = self.redis_client.generate_key(self.session_prefix, session_id)
            self.redis_client.set_json(cache_key, session_data, ex=self.session_ttl)
            logger.debug(f"Created session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if not self.redis_client.is_available():
            return None
        
        try:
            cache_key = self.redis_client.generate_key(self.session_prefix, session_id)
            session_data = self.redis_client.get_json(cache_key)
            
            if session_data:
                # Update last accessed time
                session_data['last_accessed'] = datetime.now().isoformat()
                self.redis_client.set_json(cache_key, session_data, ex=self.session_ttl)
                logger.debug(f"Retrieved session: {session_id}")
            
            return session_data
        except Exception as e:
            logger.warning(f"Error getting session {session_id}: {e}")
            return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        if not self.redis_client.is_available():
            return False
        
        try:
            session_data = self.get_session(session_id)
            if not session_data:
                return False
            
            # Merge updates
            session_data.update(updates)
            session_data['last_accessed'] = datetime.now().isoformat()
            
            cache_key = self.redis_client.generate_key(self.session_prefix, session_id)
            success = self.redis_client.set_json(cache_key, session_data, ex=self.session_ttl)
            
            if success:
                logger.debug(f"Updated session: {session_id}")
            
            return success
        except Exception as e:
            logger.warning(f"Error updating session {session_id}: {e}")
            return False
    
    def add_processed_document(self, session_id: str, url_hash: str, url: str) -> bool:
        """Add processed document to session"""
        session_data = self.get_session(session_id)
        if not session_data:
            return False
        
        processed_doc = {
            'url_hash': url_hash,
            'url': url,
            'processed_at': datetime.now().isoformat()
        }
        
        if 'processed_documents' not in session_data:
            session_data['processed_documents'] = []
        
        # Avoid duplicates
        existing_hashes = [doc['url_hash'] for doc in session_data['processed_documents']]
        if url_hash not in existing_hashes:
            session_data['processed_documents'].append(processed_doc)
        
        return self.update_session(session_id, session_data)
    
    def increment_request_count(self, session_id: str) -> bool:
        """Increment request count for session"""
        session_data = self.get_session(session_id)
        if not session_data:
            return False
        
        session_data['request_count'] = session_data.get('request_count', 0) + 1
        return self.update_session(session_id, {'request_count': session_data['request_count']})
    
    def cache_request_result(self, request_key: str, result: Dict[str, Any], 
                           ttl: Optional[int] = None) -> bool:
        """Cache request result"""
        if not self.redis_client.is_available():
            return False
        
        try:
            cache_key = self.redis_client.generate_key(self.request_prefix, request_key)
            cache_data = {
                'result': result,
                'cached_at': datetime.now().isoformat()
            }
            
            ttl = ttl or self.request_cache_ttl
            success = self.redis_client.set_json(cache_key, cache_data, ex=ttl)
            
            if success:
                logger.debug(f"Cached request result: {request_key}")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching request result: {e}")
            return False
    
    def get_cached_request_result(self, request_key: str) -> Optional[Dict[str, Any]]:
        """Get cached request result"""
        if not self.redis_client.is_available():
            return None
        
        try:
            cache_key = self.redis_client.generate_key(self.request_prefix, request_key)
            cache_data = self.redis_client.get_json(cache_key)
            
            if cache_data:
                logger.debug(f"Cache hit for request: {request_key}")
                return cache_data.get('result')
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached request result: {e}")
            return None
    
    def generate_request_key(self, url: str, questions: List[str], 
                           processing_mode: str = "standard") -> str:
        """Generate consistent request cache key"""
        import hashlib
        
        # Create deterministic key from request parameters
        key_data = {
            'url': url,
            'questions': sorted(questions),  # Sort for consistency
            'mode': processing_mode
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"req:{key_hash}"
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if not self.redis_client.is_available():
            return False
        
        try:
            cache_key = self.redis_client.generate_key(self.session_prefix, session_id)
            success = self.redis_client.delete(cache_key)
            
            if success:
                logger.debug(f"Deleted session: {session_id}")
            
            return success
        except Exception as e:
            logger.warning(f"Error deleting session {session_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Cleanup expired sessions (Redis TTL handles this automatically)"""
        # Redis TTL automatically handles cleanup
        # This method is for manual cleanup if needed
        logger.info("Session cleanup handled automatically by Redis TTL")
        return 0