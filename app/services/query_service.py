import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import structlog
from openai import AsyncOpenAI

from app.core.embedding_service import EmbeddingService
from app.core.vector_store import VectorStoreManager
from app.core.config import settings

logger = structlog.get_logger()

class QueryService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    async def answer_question(self, question: str, collection_name: str, 
                            top_k: int = 5, rerank: bool = True, 
                            include_metadata: bool = True) -> Dict[str, Any]:
        """Main query processing pipeline"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: '{question[:100]}...' in collection '{collection_name}'")
            
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(question)
            
            # Retrieve similar documents
            similar_docs = await self.vector_store.similarity_search(
                collection_name, query_embedding, top_k
            )
            
            if not similar_docs:
                return {
                    "answer": "No relevant documents found for your question.",
                    "sources": [],
                    "confidence_score": 0.0,
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "tokens_used": 0
                }
            
            # Rerank if requested
            if rerank and len(similar_docs) > 1:
                similar_docs = await self._rerank_documents(question, similar_docs)
            
            # Generate answer using LLM
            answer, sources, tokens_used = await self._generate_answer(
                question, similar_docs, include_metadata
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(similar_docs)
            
            processing_time = time.time() - start_time
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence_score": confidence_score,
                "processing_time_seconds": round(processing_time, 2),
                "tokens_used": tokens_used
            }
            
            logger.info(f"Query processed in {processing_time:.2f}s, confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "confidence_score": 0.0,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "tokens_used": 0
            }
    
    async def _rerank_documents(self, question: str, documents: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Rerank documents using a more sophisticated approach"""
        try:
            # Simple reranking based on text similarity and metadata quality
            reranked = []
            for doc, score in documents:
                # Boost score based on text length and metadata quality
                text_length = len(doc.page_content)
                metadata_score = self._calculate_metadata_score(doc.metadata)
                
                # Weighted reranking
                rerank_score = score * 0.7 + (text_length / 1000) * 0.2 + metadata_score * 0.1
                reranked.append((doc, rerank_score))
            
            # Sort by rerank score
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return documents
    
    def _calculate_metadata_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate a score based on metadata quality"""
        score = 0.0
        
        # Prefer documents with more metadata
        if metadata.get("file_type") in [".pdf", ".md"]:
            score += 0.1
        
        # Prefer longer documents
        text_length = metadata.get("text_length", 0)
        if text_length > 1000:
            score += 0.2
        elif text_length > 500:
            score += 0.1
        
        # Prefer documents with chunk index 0 (first chunk)
        if metadata.get("chunk_index", 0) == 0:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _generate_answer(self, question: str, documents: List[Tuple[Any, float]], 
                             include_metadata: bool) -> Tuple[str, List[Dict[str, Any]], int]:
        """Generate answer using OpenAI GPT"""
        try:
            # Prepare context from documents
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(documents):
                context_parts.append(f"Document {i+1}:\n{doc.page_content}")
                
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "relevance_score": round(score, 3),
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                }
                
                if include_metadata:
                    source_info.update({
                        "file_name": doc.metadata.get("file_name", "Unknown"),
                        "file_type": doc.metadata.get("file_type", "Unknown"),
                        "chunk_size": doc.metadata.get("chunk_size", 0)
                    })
                
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Based on the following documents, please answer the question. 
            If the answer cannot be found in the documents, say so clearly.
            Provide specific citations by referencing "Document X" in your answer.

            Question: {question}

            Documents:
            {context}

            Answer:"""
            
            # Generate answer
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents. Always cite your sources."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return answer, sources, tokens_used
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}", [], 0
    
    def _calculate_confidence(self, documents: List[Tuple[Any, float]]) -> float:
        """Calculate confidence score based on retrieved documents"""
        if not documents:
            return 0.0
        
        # Average similarity score
        avg_score = sum(score for _, score in documents) / len(documents)
        
        # Boost confidence if we have multiple high-quality documents
        if len(documents) >= 3:
            avg_score *= 1.1
        
        # Cap at 1.0
        return min(avg_score, 1.0)
