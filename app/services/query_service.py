import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import structlog
from openai import AsyncOpenAI

from app.core.embedding_service import EmbeddingService
from app.core.vector_store import VectorStoreManager
from app.core.hybrid_search import HybridSearchEngine
from app.core.reranking_service import RerankingService
from app.core.query_planner import QueryPlanner
from app.core.prompts import PromptTemplates
from app.core.config import settings

logger = structlog.get_logger()

class QueryService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()
        self.hybrid_search = HybridSearchEngine()
        self.reranking_service = RerankingService()
        self.query_planner = QueryPlanner()
        self.prompt_templates = PromptTemplates()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    async def answer_question(self, question: str, collection_name: str, 
                            top_k: int = 10, use_hybrid: bool = True, 
                            use_reranking: bool = True, use_query_expansion: bool = True,
                            use_planning: bool = True) -> Dict[str, Any]:
        """Answer question with hybrid search and reranking by default"""
        start_time = time.time()
        latency_breakdown = {}
        
        try:
            logger.info(f"Processing question: '{question[:100]}...' in collection '{collection_name}'")
            
            # Query planning
            if use_planning:
                plan_start = time.time()
                plan = self.query_planner.get_optimal_params(question)
                latency_breakdown["planning"] = time.time() - plan_start
                
                # Override parameters with planned values
                top_k = plan.get("top_k", top_k)
                use_hybrid = plan.get("use_hybrid", use_hybrid)
                use_reranking = plan.get("use_reranking", use_reranking)
                use_query_expansion = plan.get("use_expansion", use_query_expansion)
            else:
                plan = {"confidence": 0.5}
            
            # Generate query embedding
            embed_start = time.time()
            query_embedding = await self.embedding_service.embed_query(question)
            latency_breakdown["embedding"] = time.time() - embed_start
            
            # Retrieve documents
            if use_hybrid:
                retrieval_start = time.time()
                documents = await self._hybrid_retrieval(
                    question, query_embedding, collection_name, top_k, plan
                )
                latency_breakdown["retrieval"] = time.time() - retrieval_start
            else:
                retrieval_start = time.time()
                documents = await self.vector_store.similarity_search(
                    collection_name, query_embedding, top_k
                )
                latency_breakdown["retrieval"] = time.time() - retrieval_start
            
            if not documents:
                return {
                    "answer": "No relevant documents found for your question.",
                    "sources": [],
                    "contexts": [],
                    "confidence_score": 0.0,
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "tokens_used": 0,
                    "latency_breakdown": latency_breakdown,
                    "search_strategy": "hybrid" if use_hybrid else "vector_only"
                }
            
            # Rerank documents
            if use_reranking and len(documents) > 1:
                rerank_start = time.time()
                documents = await self.reranking_service.rerank_documents(
                    question, documents, plan.get("rerank_top_k", 5)
                )
                latency_breakdown["reranking"] = time.time() - rerank_start
            
            # Generate answer
            generation_start = time.time()
            answer, sources, contexts, tokens_used = await self._generate_answer(
                question, documents
            )
            latency_breakdown["generation"] = time.time() - generation_start
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(documents, plan.get("confidence", 0.5))
            
            processing_time = time.time() - start_time
            latency_breakdown["total"] = processing_time
            
            result = {
                "answer": answer,
                "sources": sources,
                "contexts": contexts,
                "confidence_score": confidence_score,
                "processing_time_seconds": round(processing_time, 2),
                "tokens_used": tokens_used,
                "latency_breakdown": latency_breakdown,
                "search_strategy": "hybrid_reranked" if use_hybrid and use_reranking else "vector_only",
                "query_plan": plan
            }
            
            logger.info(f"Query processed in {processing_time:.2f}s, confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "contexts": [],
                "confidence_score": 0.0,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "tokens_used": 0,
                "latency_breakdown": latency_breakdown,
                "search_strategy": "error"
            }
    
    async def _hybrid_retrieval(self, question: str, query_embedding: List[float], 
                               collection_name: str, top_k: int, plan: Dict[str, Any]) -> List[Tuple[Any, float]]:
        """Perform hybrid retrieval with BM25 and vector search"""
        try:
            # Get vector results
            vector_results = await self.vector_store.similarity_search(
                collection_name, query_embedding, top_k * 2
            )
            
            # Get BM25 results
            bm25_results = await self.hybrid_search.bm25_search(
                collection_name, question, top_k * 2
            )
            
            # Blend results
            blended_results = await self.hybrid_search.hybrid_search(
                collection_name, question, vector_results, top_k,
                semantic_weight=plan.get("vector_weight", 0.7),
                keyword_weight=plan.get("bm25_weight", 0.3)
            )
            
            return blended_results
            
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector: {e}")
            return await self.vector_store.similarity_search(
                collection_name, query_embedding, top_k
            )
    
    async def _generate_answer(self, question: str, documents: List[Tuple[Any, float]]) -> Tuple[str, List[Dict], List[str], int]:
        """Generate answer with strict grounding"""
        try:
            # Prepare context and sources
            context_parts = []
            sources = []
            contexts = []
            
            for i, (doc, score) in enumerate(documents):
                # Add to context
                context_parts.append(f"Source {i+1}:\n{doc.page_content}")
                contexts.append(doc.page_content)
                
                # Prepare source metadata
                source_info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "relevance_score": round(score, 3),
                    "section_title": doc.metadata.get("section_title", "Unknown"),
                    "page_num": doc.metadata.get("page_num", 0),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                    "file_type": doc.metadata.get("file_type", "Unknown"),
                    "doc_title": doc.metadata.get("doc_title", "Unknown")
                }
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Create prompts
            system_prompt = self.prompt_templates.get_system_prompt(
                "strict_grounding",
                max_tokens=settings.max_tokens,
                require_citations=True
            )
            
            user_prompt = self.prompt_templates.create_user_prompt(
                question, context, sources
            )
            
            # Generate answer
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=settings.max_tokens,
                temperature=0.1,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return answer, sources, contexts, tokens_used
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}", [], [], 0
    
    def _calculate_confidence(self, documents: List[Tuple[Any, float]], plan_confidence: float) -> float:
        """Calculate confidence score"""
        if not documents:
            return 0.0
        
        # Base confidence from top document score
        top_score = documents[0][1] if documents else 0.0
        
        # Diversity bonus
        unique_sources = len(set(doc.metadata.get("source", "") for doc, _ in documents))
        diversity_bonus = min(0.2, unique_sources * 0.05)
        
        # Plan confidence influence
        plan_influence = plan_confidence * 0.1
        
        # Calculate final confidence
        final_confidence = top_score + diversity_bonus + plan_influence
        
        return min(1.0, final_confidence)
    
    async def answer_question_streaming(self, question: str, collection_name: str, 
                                      top_k: int = 10, use_hybrid: bool = True, 
                                      use_reranking: bool = True) -> AsyncGenerator[str, None]:
        """Stream answer as it's generated"""
        try:
            # Get documents (same as regular query)
            query_embedding = await self.embedding_service.embed_query(question)
            
            if use_hybrid:
                documents = await self._hybrid_retrieval(
                    question, query_embedding, collection_name, top_k, {}
                )
            else:
                documents = await self.vector_store.similarity_search(
                    collection_name, query_embedding, top_k
                )
            
            if not documents:
                yield "No relevant documents found for your question."
                return
            
            # Rerank if requested
            if use_reranking and len(documents) > 1:
                documents = await self.reranking_service.rerank_documents(
                    question, documents, 5
                )
            
            # Prepare context
            context_parts = []
            for i, (doc, score) in enumerate(documents):
                context_parts.append(f"Source {i+1}:\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Create streaming prompt
            user_prompt = self.prompt_templates.create_streaming_prompt(
                question, context, []
            )
            
            # Stream response
            async for chunk in self._stream_generation(user_prompt):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"Error: {str(e)}"
    
    async def _stream_generation(self, user_prompt: str) -> AsyncGenerator[str, None]:
        """Stream generation from OpenAI"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide accurate, cited responses."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=settings.max_tokens,
                temperature=0.1,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"
