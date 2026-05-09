"""Redis caching layer for RAG system.

Industry best practice: Cache repeated queries to reduce latency and costs.
Reference: Redis caching patterns for ML systems.
"""

from typing import Optional, Any
import json
import hashlib
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis")

from src.utils.config_loader import get_config


class QueryCache:
    """Cache for RAG query results using Redis.
    
    Best practices:
    - Cache expensive operations (embeddings, LLM calls)
    - Set appropriate TTL (time-to-live)
    - Use query hash as key
    - Handle cache misses gracefully
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,
        enabled: bool = True,
    ):
        """Initialize query cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Time-to-live in seconds (default 1 hour)
            enabled: Whether caching is enabled
        """
        self.enabled = enabled and REDIS_AVAILABLE
        self.ttl = ttl
        self.redis_client = None
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True,
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Redis cache connected: {host}:{port}")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self.enabled = False
        else:
            logger.warning("Query cache disabled (Redis not available or disabled)")

    def get(self, query: str, config: str) -> Optional[dict]:
        """Get cached result for query.
        
        Args:
            query: User query
            config: RAG configuration name
            
        Returns:
            Cached result dict or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            key = self._generate_key(query, config)
            cached = self.redis_client.get(key)
            
            if cached:
                logger.debug(f"Cache HIT: {key[:32]}...")
                return json.loads(cached)
            else:
                logger.debug(f"Cache MISS: {key[:32]}...")
                return None
                
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    def set(self, query: str, config: str, result: dict) -> bool:
        """Cache result for query.
        
        Args:
            query: User query
            config: RAG configuration name
            result: Result dictionary to cache
            
        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False
        
        try:
            key = self._generate_key(query, config)
            serialized = json.dumps(result, default=str)
            
            self.redis_client.setex(key, self.ttl, serialized)
            logger.debug(f"Cache SET: {key[:32]}... (TTL={self.ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False

    def invalidate(self, query: str, config: str) -> bool:
        """Invalidate cache for specific query.
        
        Args:
            query: User query
            config: RAG configuration name
            
        Returns:
            True if invalidated successfully
        """
        if not self.enabled:
            return False
        
        try:
            key = self._generate_key(query, config)
            deleted = self.redis_client.delete(key)
            logger.debug(f"Cache INVALIDATE: {key[:32]}...")
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Cache invalidate failed: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all cached queries.
        
        Returns:
            True if cleared successfully
        """
        if not self.enabled:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.info("Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        if not self.enabled:
            return {"enabled": False}
        
        try:
            info = self.redis_client.info("stats")
            return {
                "enabled": True,
                "total_keys": self.redis_client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}

    def _generate_key(self, query: str, config: str) -> str:
        """Generate cache key from query and config.
        
        Args:
            query: User query
            config: RAG configuration name
            
        Returns:
            Cache key string
        """
        # Create hash of query + config
        content = f"{query}:{config}"
        hash_obj = hashlib.sha256(content.encode())
        key = f"rag:query:{hash_obj.hexdigest()}"
        return key

    def _calculate_hit_rate(self, info: dict) -> float:
        """Calculate cache hit rate.
        
        Args:
            info: Redis info dict
            
        Returns:
            Hit rate (0-1)
        """
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return hits / total


# Global cache instance
_cache_instance = None


def get_cache(enabled: bool = True) -> QueryCache:
    """Get global cache instance (singleton).
    
    Args:
        enabled: Whether caching is enabled
        
    Returns:
        QueryCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache(enabled=enabled)
    return _cache_instance


class EmbeddingCache:
    """Specialized cache for embeddings.
    
    Embeddings are expensive to compute, so we cache them separately.
    """

    def __init__(self, cache: QueryCache):
        """Initialize embedding cache.
        
        Args:
            cache: Underlying cache instance
        """
        self.cache = cache

    def get_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding for text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector or None
        """
        if not self.cache.enabled:
            return None
        
        try:
            key = self._generate_embedding_key(text)
            cached = self.cache.redis_client.get(key)
            
            if cached:
                logger.debug("Embedding cache HIT")
                return json.loads(cached)
            else:
                logger.debug("Embedding cache MISS")
                return None
                
        except Exception as e:
            logger.error(f"Embedding cache get failed: {e}")
            return None

    def set_embedding(self, text: str, embedding: list) -> bool:
        """Cache embedding for text.
        
        Args:
            text: Text
            embedding: Embedding vector
            
        Returns:
            True if cached successfully
        """
        if not self.cache.enabled:
            return False
        
        try:
            key = self._generate_embedding_key(text)
            serialized = json.dumps(embedding)
            
            # Embeddings can be cached longer (24 hours)
            self.cache.redis_client.setex(key, 86400, serialized)
            logger.debug("Embedding cached")
            return True
            
        except Exception as e:
            logger.error(f"Embedding cache set failed: {e}")
            return False

    def _generate_embedding_key(self, text: str) -> str:
        """Generate cache key for embedding.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key
        """
        hash_obj = hashlib.sha256(text.encode())
        return f"rag:embedding:{hash_obj.hexdigest()}"
