import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import structlog
import redis.asyncio as redis
import hashlib
import time
from datetime import datetime, timedelta

from app.core.config import settings

logger = structlog.get_logger()

class CacheService:
    def __init__(self):
        self.redis_client = None
        self.cache_ttl = getattr(settings, 'cache_ttl', 3600)  # 1 hour default
        self.vector_cache_ttl = 7200  # 2 hours for vectors
        self.rerank_cache_ttl = 1800  # 30 minutes for reranking
        self.answer_cache_ttl = 600  # 10 minutes for answers
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_vector_hits(self, query: str, collection: str) -> Optional[List[Tuple[Any, float]]]:
        """Get cached vector search results"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key("vector", query, collection)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.debug(f"Vector cache hit for query: {query[:50]}...")
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting vector cache: {e}")
            return None
    
    async def set_vector_hits(self, query: str, collection: str, 
                            results: List[Tuple[Any, float]], ttl: Optional[int] = None) -> bool:
        """Cache vector search results"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key("vector", query, collection)
            ttl = ttl or self.vector_cache_ttl
            
            # Serialize results (simplified for now)
            serializable_results = []
            for doc, score in results:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    serializable_results.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    })
            
            await self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(serializable_results, default=str)
            )
            
            logger.debug(f"Vector cache set for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error setting vector cache: {e}")
            return False
    
    async def get_rerank_score(self, query: str, doc_id: str) -> Optional[float]:
        """Get cached reranking score"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key("rerank", query, doc_id)
            cached_score = await self.redis_client.get(cache_key)
            
            if cached_score:
                logger.debug(f"Rerank cache hit for query: {query[:50]}...")
                return float(cached_score)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting rerank cache: {e}")
            return None
    
    async def set_rerank_score(self, query: str, doc_id: str, score: float, 
                             ttl: Optional[int] = None) -> bool:
        """Cache reranking score"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key("rerank", query, doc_id)
            ttl = ttl or self.rerank_cache_ttl
            
            await self.redis_client.setex(cache_key, ttl, str(score))
            
            logger.debug(f"Rerank cache set for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error setting rerank cache: {e}")
            return False
    
    async def get_answer(self, query: str, collection: str, filters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get cached answer"""
        if not self.redis_client:
            return None
        
        try:
            # Include filters in cache key
            filter_str = json.dumps(filters or {}, sort_keys=True)
            cache_key = self._generate_cache_key("answer", query, collection, filter_str)
            
            cached_answer = await self.redis_client.get(cache_key)
            
            if cached_answer:
                logger.debug(f"Answer cache hit for query: {query[:50]}...")
                return json.loads(cached_answer)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting answer cache: {e}")
            return None
    
    async def set_answer(self, query: str, collection: str, answer: Dict[str, Any], 
                        filters: Dict[str, Any] = None, ttl: Optional[int] = None) -> bool:
        """Cache answer"""
        if not self.redis_client:
            return False
        
        try:
            filter_str = json.dumps(filters or {}, sort_keys=True)
            cache_key = self._generate_cache_key("answer", query, collection, filter_str)
            ttl = ttl or self.answer_cache_ttl
            
            await self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(answer, default=str)
            )
            
            logger.debug(f"Answer cache set for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error setting answer cache: {e}")
            return False
    
    async def invalidate_collection_cache(self, collection: str) -> bool:
        """Invalidate all cache entries for a collection"""
        if not self.redis_client:
            return False
        
        try:
            # Get all keys for the collection
            pattern = f"*:{collection}:*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for collection: {collection}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating collection cache: {e}")
            return False
    
    async def invalidate_document_cache(self, collection: str, source: str) -> bool:
        """Invalidate cache entries for a specific document"""
        if not self.redis_client:
            return False
        
        try:
            # This would need to be more sophisticated in practice
            # For now, invalidate all cache entries for the collection
            return await self.invalidate_collection_cache(collection)
            
        except Exception as e:
            logger.error(f"Error invalidating document cache: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        try:
            info = await self.redis_client.info()
            
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        
        if hits + misses == 0:
            return 0.0
        
        return hits / (hits + misses)
    
    async def clear_all_cache(self) -> bool:
        """Clear all cache entries"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.flushdb()
            logger.info("Cleared all cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def get_cache_size(self) -> int:
        """Get number of cache entries"""
        if not self.redis_client:
            return 0
        
        try:
            return await self.redis_client.dbsize()
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return 0

class CacheMetrics:
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.hit_counts = {
            "vector": 0,
            "rerank": 0,
            "answer": 0
        }
        self.miss_counts = {
            "vector": 0,
            "rerank": 0,
            "answer": 0
        }
    
    def record_hit(self, cache_type: str):
        """Record cache hit"""
        self.hit_counts[cache_type] = self.hit_counts.get(cache_type, 0) + 1
    
    def record_miss(self, cache_type: str):
        """Record cache miss"""
        self.miss_counts[cache_type] = self.miss_counts.get(cache_type, 0) + 1
    
    def get_hit_rate(self, cache_type: str) -> float:
        """Get hit rate for cache type"""
        hits = self.hit_counts.get(cache_type, 0)
        misses = self.miss_counts.get(cache_type, 0)
        
        if hits + misses == 0:
            return 0.0
        
        return hits / (hits + misses)
    
    def get_total_hit_rate(self) -> float:
        """Get overall hit rate"""
        total_hits = sum(self.hit_counts.values())
        total_misses = sum(self.miss_counts.values())
        
        if total_hits + total_misses == 0:
            return 0.0
        
        return total_hits / (total_hits + total_misses)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        return {
            "hit_counts": self.hit_counts,
            "miss_counts": self.miss_counts,
            "hit_rates": {
                cache_type: self.get_hit_rate(cache_type)
                for cache_type in self.hit_counts.keys()
            },
            "total_hit_rate": self.get_total_hit_rate()
        }
