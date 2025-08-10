"""Redis Client Configuration and Management"""

import os
import json
import pickle
import logging
import hashlib
from typing import Any, Optional, Union
from datetime import timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis client wrapper with fallback capabilities"""
    
    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 6379, 
                 db: int = 0,
                 decode_responses: bool = False,
                 socket_timeout: int = 5):
        self.host = host
        self.port = port
        self.db = db
        self.decode_responses = decode_responses
        self.socket_timeout = socket_timeout
        self.client = None
        self.available = False
        
        if REDIS_AVAILABLE:
            self._connect()
        else:
            logger.warning("Redis not available. Caching will be disabled.")
    
    def _connect(self):
        """Establish Redis connection"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_timeout
            )
            # Test connection
            self.client.ping()
            self.available = True
            logger.info(f"Redis connected successfully at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Operating without cache.")
            self.client = None
            self.available = False
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.available:
            return False
        
        try:
            self.client.ping()
            return True
        except:
            self.available = False
            return False
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value from Redis"""
        if not self.is_available():
            return None
        
        try:
            return self.client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET failed for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Union[str, bytes], ex: Optional[int] = None) -> bool:
        """Set value in Redis"""
        if not self.is_available():
            return False
        
        try:
            return self.client.set(key, value, ex=ex)
        except Exception as e:
            logger.warning(f"Redis SET failed for key {key}: {e}")
            return False
    
    def delete(self, *keys: str) -> bool:
        """Delete keys from Redis"""
        if not self.is_available():
            return False
        
        try:
            self.client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE failed: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.is_available():
            return False
        
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.warning(f"Redis EXISTS failed for key {key}: {e}")
            return False
    
    def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set JSON value in Redis"""
        try:
            json_value = json.dumps(value)
            return self.set(key, json_value, ex=ex)
        except Exception as e:
            logger.warning(f"Redis SET_JSON failed for key {key}: {e}")
            return False
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from Redis"""
        try:
            value = self.get(key)
            if value is None:
                return None
            return json.loads(value.decode('utf-8') if isinstance(value, bytes) else value)
        except Exception as e:
            logger.warning(f"Redis GET_JSON failed for key {key}: {e}")
            return None
    
    def set_pickle(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set pickled value in Redis"""
        try:
            pickled_value = pickle.dumps(value)
            return self.set(key, pickled_value, ex=ex)
        except Exception as e:
            logger.warning(f"Redis SET_PICKLE failed for key {key}: {e}")
            return False
    
    def get_pickle(self, key: str) -> Optional[Any]:
        """Get pickled value from Redis"""
        try:
            value = self.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            logger.warning(f"Redis GET_PICKLE failed for key {key}: {e}")
            return None
    
    def generate_key(self, prefix: str, *args) -> str:
        """Generate a consistent cache key"""
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = ":".join(key_parts)
        # Hash long keys to avoid Redis key length limits
        if len(key_string) > 200:
            hash_key = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:hash:{hash_key}"
        return key_string

# Global Redis client instance
_redis_client = None

def get_redis_client() -> RedisClient:
    """Get global Redis client instance"""
    global _redis_client
    if _redis_client is None:
        # Get Redis configuration from environment
        host = os.getenv('REDIS_HOST', 'localhost')
        port = int(os.getenv('REDIS_PORT', 6379))
        db = int(os.getenv('REDIS_DB', 0))
        
        _redis_client = RedisClient(host=host, port=port, db=db)
    
    return _redis_client