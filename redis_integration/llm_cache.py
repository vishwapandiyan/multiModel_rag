"""LLM Response Caching"""

import hashlib
import logging
from typing import Optional, Dict, Any
from .redis_client import get_redis_client

logger = logging.getLogger(__name__)

class LLMCache:
    """Cache for LLM responses"""
    
    def __init__(self, default_ttl: int = 3600):  # 1 hour default
        self.redis_client = get_redis_client()
        self.default_ttl = default_ttl
        self.prefix = "llm_response"
    
    def _generate_prompt_hash(self, prompt: str, model: str, **kwargs) -> str:
        """Generate hash for prompt + model + parameters"""
        # Include relevant parameters in hash
        params_str = f"model:{model}"
        for key, value in sorted(kwargs.items()):
            if key in ['temperature', 'max_tokens', 'top_p', 'reasoning_effort']:
                params_str += f":{key}:{value}"
        
        combined = f"{prompt}|{params_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def get_response(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """Get cached LLM response"""
        if not self.redis_client.is_available():
            return None
        
        try:
            prompt_hash = self._generate_prompt_hash(prompt, model, **kwargs)
            cache_key = self.redis_client.generate_key(
                self.prefix, model, prompt_hash
            )
            
            cached_response = self.redis_client.get_json(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for LLM response: {prompt_hash}")
                return cached_response.get('content')
            
            return None
        except Exception as e:
            logger.warning(f"Error getting cached LLM response: {e}")
            return None
    
    def set_response(self, prompt: str, model: str, response: str, 
                    ttl: Optional[int] = None, **kwargs) -> bool:
        """Cache LLM response"""
        if not self.redis_client.is_available():
            return False
        
        try:
            prompt_hash = self._generate_prompt_hash(prompt, model, **kwargs)
            cache_key = self.redis_client.generate_key(
                self.prefix, model, prompt_hash
            )
            
            cache_data = {
                'content': response,
                'model': model,
                'prompt_hash': prompt_hash,
                'cached_at': self.redis_client.client.time()[0] if self.redis_client.is_available() else None
            }
            
            ttl = ttl or self.default_ttl
            success = self.redis_client.set_json(cache_key, cache_data, ex=ttl)
            
            if success:
                logger.debug(f"Cached LLM response: {prompt_hash}")
            
            return success
        except Exception as e:
            logger.warning(f"Error caching LLM response: {e}")
            return False
    
    def get_groq_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 1024, reasoning_effort: str = "medium") -> Optional[str]:
        """Get cached Groq response with specific parameters"""
        return self.get_response(
            prompt, 
            "groq", 
            temperature=temperature, 
            max_tokens=max_tokens, 
            reasoning_effort=reasoning_effort
        )
    
    def set_groq_response(self, prompt: str, response: str, temperature: float = 0.7, 
                         max_tokens: int = 1024, reasoning_effort: str = "medium",
                         ttl: Optional[int] = None) -> bool:
        """Cache Groq response with specific parameters"""
        return self.set_response(
            prompt, 
            "groq", 
            response, 
            ttl=ttl,
            temperature=temperature, 
            max_tokens=max_tokens, 
            reasoning_effort=reasoning_effort
        )
    
    def get_nvidia_response(self, prompt: str) -> Optional[str]:
        """Get cached NVIDIA LLM response"""
        return self.get_response(prompt, "nvidia")
    
    def set_nvidia_response(self, prompt: str, response: str, ttl: Optional[int] = None) -> bool:
        """Cache NVIDIA LLM response"""
        return self.set_response(prompt, "nvidia", response, ttl=ttl)
    
    def get_enhanced_answer_response(self, question: str, context: str) -> Optional[str]:
        """Get cached response for enhanced answer generation (only for factual questions)"""
        # Only cache factual, document-based questions, not creative or reasoning questions
        if self._is_creative_question(question):
            return None
            
        # Create a combined prompt hash for question + context
        combined_prompt = f"Question: {question}\nContext: {context}"
        return self.get_response(combined_prompt, "enhanced_answer")
    
    def set_enhanced_answer_response(self, question: str, context: str, 
                                   response: str, ttl: Optional[int] = None) -> bool:
        """Cache response for enhanced answer generation (only for factual questions)"""
        # Only cache factual, document-based questions, not creative or reasoning questions
        if self._is_creative_question(question):
            return False
            
        combined_prompt = f"Question: {question}\nContext: {context}"
        return self.set_response(combined_prompt, "enhanced_answer", response, ttl=ttl)
    
    def _is_creative_question(self, question: str) -> bool:
        """Determine if a question requires creative/reasoning response (should not be cached)"""
        creative_keywords = [
            'what do you think', 'in your opinion', 'analyze', 'reasoning', 'explain why',
            'how would you', 'what should', 'recommend', 'suggest', 'creative', 'innovative',
            'brainstorm', 'imagine', 'suppose', 'hypothetical', 'what if', 'predict',
            'strategy', 'approach', 'solution', 'solve', 'plan', 'design'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in creative_keywords)
    
    def clear_cache(self, model: Optional[str] = None) -> bool:
        """Clear LLM response cache"""
        if not self.redis_client.is_available():
            return False
        
        try:
            # This is a simplified implementation
            # In production, you'd want to use Redis SCAN to find and delete keys
            logger.info(f"Cache clearing not fully implemented. Manual Redis cleanup needed.")
            return True
        except Exception as e:
            logger.warning(f"Error clearing LLM cache: {e}")
            return False