import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import structlog
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder

from app.core.embedding_service import EmbeddingService
from app.core.vector_store import VectorStoreManager
from app.core.config import settings

logger = structlog.get_logger()

class AdvancedQueryService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Initialize reranking model for better precision
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
            self.reranker = None
        
        # Query expansion and refinement
        self.query_expansion_keywords = {
            "what": ["definition", "meaning", "explanation", "concept"],
            "how": ["method", "process", "steps", "procedure", "implementation"],
            "why": ["reason", "cause", "purpose", "benefit", "advantage"],
            "when": ["time", "date", "schedule", "timeline", "duration"],
            "where": ["location", "place", "position", "site", "area"],
            "who": ["person", "individual", "expert", "author", "creator"]
        }
    
    async def answer_question_advanced(self, question: str, collection_name: str, 
                                     top_k: int = 10, use_reranking: bool = True,
                                     use_query_expansion: bool = True,
                                     use_hybrid_search: bool = True) -> Dict[str, Any]:
        """Advanced question answering with multiple precision techniques"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing advanced query: '{question[:100]}...' in collection '{collection_name}'")
            
            # Step 1: Query preprocessing and expansion
            processed_queries = await self._preprocess_query(question, use_query_expansion)
            
            # Step 2: Multi-query retrieval
            all_candidates = []
            for query in processed_queries:
                candidates = await self._retrieve_candidates(query, collection_name, top_k * 2)
                all_candidates.extend(candidates)
            
            # Remove duplicates while preserving order
            seen_docs = set()
            unique_candidates = []
            for doc, score in all_candidates:
                doc_id = doc.metadata.get('source', '') + str(doc.metadata.get('chunk_index', 0))
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    unique_candidates.append((doc, score))
            
            # Step 3: Hybrid search (semantic + keyword)
            if use_hybrid_search:
                hybrid_candidates = await self._hybrid_search(question, unique_candidates)
            else:
                hybrid_candidates = unique_candidates
            
            # Step 4: Reranking for precision
            if use_reranking and len(hybrid_candidates) > top_k:
                reranked_candidates = await self._rerank_candidates(question, hybrid_candidates)
            else:
                reranked_candidates = hybrid_candidates
            
            # Step 5: Select top results
            final_candidates = reranked_candidates[:top_k]
            
            if not final_candidates:
                return {
                    "answer": "No relevant documents found for your question.",
                    "sources": [],
                    "confidence_score": 0.0,
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "tokens_used": 0,
                    "search_strategy": "advanced"
                }
            
            # Step 6: Generate precise answer
            answer, sources, tokens_used = await self._generate_precise_answer(
                question, final_candidates
            )
            
            # Step 7: Calculate advanced confidence score
            confidence_score = self._calculate_advanced_confidence(final_candidates, question)
            
            processing_time = time.time() - start_time
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence_score": confidence_score,
                "processing_time_seconds": round(processing_time, 2),
                "tokens_used": tokens_used,
                "search_strategy": "advanced",
                "candidates_evaluated": len(unique_candidates),
                "reranking_used": use_reranking,
                "query_expansion_used": use_query_expansion
            }
            
            logger.info(f"Advanced query processed in {processing_time:.2f}s, confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Advanced query processing failed: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "confidence_score": 0.0,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "tokens_used": 0,
                "search_strategy": "advanced"
            }
    
    async def _preprocess_query(self, question: str, use_expansion: bool) -> List[str]:
        """Preprocess and expand the query"""
        queries = [question]
        
        if use_expansion:
            # Extract question type
            question_lower = question.lower()
            question_type = None
            for q_type, keywords in self.query_expansion_keywords.items():
                if question_lower.startswith(q_type):
                    question_type = q_type
                    break
            
            # Add expanded queries
            if question_type and question_type in self.query_expansion_keywords:
                for keyword in self.query_expansion_keywords[question_type][:2]:  # Top 2 keywords
                    expanded_query = f"{question} {keyword}"
                    queries.append(expanded_query)
            
            # Add synonym-based expansion
            synonyms = await self._get_synonyms(question)
            for synonym in synonyms[:2]:  # Top 2 synonyms
                synonym_query = f"{question} {synonym}"
                queries.append(synonym_query)
        
        return queries
    
    async def _get_synonyms(self, text: str) -> List[str]:
        """Get synonyms for query expansion"""
        try:
            # Simple synonym mapping (in production, use WordNet or similar)
            synonym_map = {
                "machine learning": ["ML", "artificial intelligence", "AI", "neural networks"],
                "data": ["information", "dataset", "records", "facts"],
                "algorithm": ["method", "technique", "approach", "procedure"],
                "model": ["system", "framework", "architecture", "structure"],
                "performance": ["efficiency", "speed", "accuracy", "effectiveness"],
                "analysis": ["examination", "evaluation", "study", "investigation"],
                "implementation": ["development", "creation", "building", "construction"],
                "optimization": ["improvement", "enhancement", "tuning", "refinement"]
            }
            
            synonyms = []
            text_lower = text.lower()
            for key, values in synonym_map.items():
                if key in text_lower:
                    synonyms.extend(values)
            
            return synonyms[:5]  # Return top 5 synonyms
        except Exception as e:
            logger.warning(f"Error getting synonyms: {e}")
            return []
    
    async def _retrieve_candidates(self, query: str, collection_name: str, top_k: int) -> List[Tuple[Any, float]]:
        """Retrieve candidates using semantic search"""
        try:
            query_embedding = await self.embedding_service.embed_query(query)
            candidates = await self.vector_store.similarity_search(
                collection_name, query_embedding, top_k
            )
            return candidates
        except Exception as e:
            logger.error(f"Error retrieving candidates: {e}")
            return []
    
    async def _hybrid_search(self, question: str, candidates: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Combine semantic and keyword-based search"""
        try:
            if not candidates:
                return candidates
            
            # Extract documents and scores
            docs = [doc for doc, score in candidates]
            semantic_scores = [score for doc, score in candidates]
            
            # Keyword-based scoring using TF-IDF
            texts = [doc.page_content for doc in docs]
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate keyword similarity
            question_vector = vectorizer.transform([question])
            keyword_scores = cosine_similarity(question_vector, tfidf_matrix)[0]
            
            # Combine scores (70% semantic, 30% keyword)
            combined_scores = []
            for i, (doc, semantic_score) in enumerate(zip(docs, semantic_scores)):
                keyword_score = keyword_scores[i] if i < len(keyword_scores) else 0
                combined_score = 0.7 * semantic_score + 0.3 * keyword_score
                combined_scores.append((doc, combined_score))
            
            # Sort by combined score
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            return combined_scores
            
        except Exception as e:
            logger.warning(f"Hybrid search failed, using semantic only: {e}")
            return candidates
    
    async def _rerank_candidates(self, question: str, candidates: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Rerank candidates using cross-encoder for better precision"""
        try:
            if not self.reranker or len(candidates) <= 1:
                return candidates
            
            # Prepare query-document pairs for reranking
            pairs = []
            for doc, score in candidates:
                pairs.append([question, doc.page_content])
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine original scores with rerank scores
            reranked_candidates = []
            for i, (doc, original_score) in enumerate(candidates):
                rerank_score = float(rerank_scores[i])
                # Weighted combination: 60% rerank, 40% original
                final_score = 0.6 * rerank_score + 0.4 * original_score
                reranked_candidates.append((doc, final_score))
            
            # Sort by final score
            reranked_candidates.sort(key=lambda x: x[1], reverse=True)
            return reranked_candidates
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return candidates
    
    async def _generate_precise_answer(self, question: str, candidates: List[Tuple[Any, float]]) -> Tuple[str, List[Dict], int]:
        """Generate precise answer with better context understanding"""
        try:
            # Prepare context with relevance scoring
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(candidates):
                # Add relevance indicator
                relevance = "high" if score > 0.8 else "medium" if score > 0.6 else "low"
                
                context_parts.append(f"Document {i+1} (Relevance: {relevance}):\n{doc.page_content}")
                
                source_info = {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "relevance_score": round(score, 3),
                    "relevance_level": relevance,
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                    "file_type": doc.metadata.get("file_type", "Unknown")
                }
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Create enhanced prompt for precise answering
            prompt = f"""You are an expert AI assistant with access to a comprehensive knowledge base. 
            Answer the following question using ONLY the provided documents. Be precise, accurate, and cite specific sources.

            Question: {question}

            Available Documents:
            {context}

            Instructions:
            1. Provide a comprehensive and accurate answer based on the documents
            2. Cite specific documents when making claims (e.g., "According to Document 1...")
            3. If information is incomplete, mention what's available and what's missing
            4. If the answer cannot be found in the documents, state this clearly
            5. Prioritize information from high-relevance documents
            6. Be concise but thorough

            Answer:"""
            
            # Generate answer with GPT-4 for better quality
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better precision
                messages=[
                    {"role": "system", "content": "You are a precise and accurate AI assistant that provides detailed, well-cited answers based on provided documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return answer, sources, tokens_used
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}", [], 0
    
    def _calculate_advanced_confidence(self, candidates: List[Tuple[Any, float]], question: str) -> float:
        """Calculate advanced confidence score based on multiple factors"""
        if not candidates:
            return 0.0
        
        # Base confidence from top candidate score
        top_score = candidates[0][1] if candidates else 0.0
        
        # Diversity bonus (if we have multiple high-quality sources)
        high_quality_count = sum(1 for _, score in candidates if score > 0.7)
        diversity_bonus = min(0.2, high_quality_count * 0.05)
        
        # Consistency bonus (if multiple sources agree)
        if len(candidates) >= 2:
            score_consistency = 1.0 - abs(candidates[0][1] - candidates[1][1])
            consistency_bonus = score_consistency * 0.1
        else:
            consistency_bonus = 0.0
        
        # Length bonus (longer, more detailed answers are often better)
        total_content_length = sum(len(doc.page_content) for doc, _ in candidates)
        length_bonus = min(0.1, total_content_length / 10000)  # Cap at 0.1
        
        # Calculate final confidence
        final_confidence = top_score + diversity_bonus + consistency_bonus + length_bonus
        
        return min(1.0, final_confidence)  # Cap at 1.0
