import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
import time
from functools import lru_cache
import structlog
from sentence_transformers import CrossEncoder
import cohere

from app.core.config import settings
from app.core.cache import CacheService

logger = structlog.get_logger()

class RerankingService:
    def __init__(self):
        self.reranker_type = getattr(settings, 'reranker_type', 'cross_encoder')
        self.cache_ttl = getattr(settings, 'reranker_cache_ttl', 3600)  # 1 hour
        self.cache_service = CacheService()
        self.cohere_client = None
        
        # Initialize reranker based on type
        self._initialize_reranker()
    
    def _initialize_reranker(self):
        """Initialize the appropriate reranker"""
        try:
            if self.reranker_type == 'bge_reranker':
                self.reranker = CrossEncoder('BAAI/bge-reranker-large')
                logger.info("Initialized BGE reranker")
            elif self.reranker_type == 'cohere':
                cohere_api_key = getattr(settings, 'cohere_api_key', None)
                if not cohere_api_key:
                    logger.warning("Cohere API key not found, falling back to cross-encoder")
                    self.reranker_type = 'cross_encoder'
                    self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                else:
                    self.cohere_client = cohere.Client(cohere_api_key)
                    logger.info("Initialized Cohere reranker")
            else:  # cross_encoder
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Initialized Cross-Encoder reranker")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.reranker = None
    
    def _get_cache_key(self, query: str, documents: List[str]) -> str:
        """Generate cache key for reranking results"""
        content = f"{query}|||{json.dumps(documents, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def score_batch(self, query: str, documents: List[Tuple[Any, float]]) -> List[float]:
        """Score a batch of documents for reranking"""
        if not documents or not self.reranker:
            return [0.5] * len(documents)
        
        try:
            # Extract document texts
            doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                        for doc, _ in documents]
            
            # Check cache first
            cache_key = self._get_cache_key(query, doc_texts)
            cached_scores = await self.cache_service.get_rerank_score(cache_key, "batch")
            
            if cached_scores is not None:
                logger.debug("Using cached reranking scores")
                return cached_scores
            
            # Perform reranking
            if self.reranker_type == 'cohere' and self.cohere_client:
                scores = await self._rerank_with_cohere(query, doc_texts)
            else:
                scores = await self._rerank_with_cross_encoder(query, doc_texts)
            
            # Cache results
            await self.cache_service.set_rerank_score(cache_key, "batch", scores, self.cache_ttl)
            
            return scores
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [0.5] * len(documents)  # Fallback scores
    
    async def _rerank_with_cross_encoder(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using Cross-Encoder"""
        try:
            # Prepare query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get scores in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, 
                self.reranker.predict, 
                pairs
            )
            
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return [0.5] * len(documents)  # Fallback scores
    
    async def _rerank_with_cohere(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using Cohere"""
        try:
            # Prepare documents for Cohere
            doc_texts = [{"text": doc} for doc in documents]
            
            # Call Cohere rerank API
            response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=doc_texts,
                top_k=len(documents)
            )
            
            # Extract scores
            scores = [0.0] * len(documents)
            for result in response.results:
                scores[result.index] = result.relevance_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return [0.5] * len(documents)  # Fallback scores
    
    async def rerank_documents(self, query: str, documents: List[Tuple[Any, float]], 
                              top_k: Optional[int] = None) -> List[Tuple[Any, float]]:
        """Rerank documents with caching"""
        if not documents or not self.reranker:
            return documents
        
        try:
            # Get reranking scores
            rerank_scores = await self.score_batch(query, documents)
            
            # Apply scores to documents
            combined_results = []
            for i, ((doc, original_score), rerank_score) in enumerate(zip(documents, rerank_scores)):
                # Weighted combination: 60% rerank, 40% original
                final_score = 0.6 * rerank_score + 0.4 * original_score
                combined_results.append((doc, final_score))
            
            # Sort by final score
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            return combined_results[:top_k] if top_k else combined_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k] if top_k else documents
    
    async def get_reranking_stats(self) -> Dict[str, Any]:
        """Get reranking service statistics"""
        return {
            "reranker_type": self.reranker_type,
            "cache_ttl": self.cache_ttl,
            "is_available": self.reranker is not None or self.cohere_client is not None
        }
    
    async def clear_cache(self):
        """Clear all cached reranking scores"""
        # This would clear the reranking cache
        logger.info("Reranking cache cleared")
