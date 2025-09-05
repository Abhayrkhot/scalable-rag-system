import re
import logging
from typing import Dict, Any, Tuple
import structlog

logger = structlog.get_logger()

class QueryPlanner:
    def __init__(self):
        # Query type patterns
        self.factual_patterns = [
            r'\b(what|who|when|where|which|how many|how much)\b',
            r'\b(define|definition|meaning|explain)\b',
            r'\b(compare|difference|similar|versus|vs)\b'
        ]
        
        self.procedural_patterns = [
            r'\b(how to|how do|steps|process|procedure|method)\b',
            r'\b(implement|create|build|develop|setup|configure)\b',
            r'\b(tutorial|guide|walkthrough|example)\b'
        ]
        
        self.conceptual_patterns = [
            r'\b(why|reason|cause|purpose|benefit|advantage)\b',
            r'\b(concept|theory|principle|idea|notion)\b',
            r'\b(understand|comprehend|learn|study)\b'
        ]
        
        self.search_patterns = [
            r'\b(find|search|look for|locate|discover)\b',
            r'\b(list|show|display|present)\b',
            r'\b(available|options|choices|alternatives)\b'
        ]
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query and return planning information"""
        query_lower = query.lower()
        
        # Determine query type
        query_type = self._classify_query_type(query_lower)
        
        # Calculate weights based on query type
        bm25_weight, vector_weight = self._calculate_weights(query_type, query_lower)
        
        # Determine if reranking is beneficial
        use_reranking = self._should_use_reranking(query_type, query_lower)
        
        # Determine if query expansion is beneficial
        use_expansion = self._should_use_expansion(query_type, query_lower)
        
        return {
            "query_type": query_type,
            "bm25_weight": bm25_weight,
            "vector_weight": vector_weight,
            "use_reranking": use_reranking,
            "use_expansion": use_expansion,
            "confidence": self._calculate_confidence(query_type, query_lower)
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        scores = {
            "factual": self._score_patterns(query, self.factual_patterns),
            "procedural": self._score_patterns(query, self.procedural_patterns),
            "conceptual": self._score_patterns(query, self.conceptual_patterns),
            "search": self._score_patterns(query, self.search_patterns)
        }
        
        # Return the type with highest score, default to factual
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "factual"
    
    def _score_patterns(self, query: str, patterns: list) -> int:
        """Score query against pattern list"""
        score = 0
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 1
        return score
    
    def _calculate_weights(self, query_type: str, query: str) -> Tuple[float, float]:
        """Calculate BM25 and vector weights based on query type"""
        base_weights = {
            "factual": (0.4, 0.6),      # More semantic for facts
            "procedural": (0.6, 0.4),   # More keyword for procedures
            "conceptual": (0.3, 0.7),   # More semantic for concepts
            "search": (0.7, 0.3)        # More keyword for search
        }
        
        bm25_base, vector_base = base_weights.get(query_type, (0.5, 0.5))
        
        # Adjust based on query characteristics
        query_length = len(query.split())
        
        # Longer queries benefit more from semantic search
        if query_length > 10:
            vector_base += 0.1
            bm25_base -= 0.1
        elif query_length < 5:
            bm25_base += 0.1
            vector_base -= 0.1
        
        # Technical terms suggest more keyword search
        technical_terms = ['api', 'function', 'method', 'class', 'variable', 'code', 'syntax']
        if any(term in query for term in technical_terms):
            bm25_base += 0.1
            vector_base -= 0.1
        
        # Ensure weights sum to 1.0
        total = bm25_base + vector_base
        return bm25_base / total, vector_base / total
    
    def _should_use_reranking(self, query_type: str, query: str) -> bool:
        """Determine if reranking should be used"""
        # Always use reranking for complex queries
        if len(query.split()) > 8:
            return True
        
        # Use reranking for conceptual and factual queries
        if query_type in ["conceptual", "factual"]:
            return True
        
        # Use reranking if query has multiple concepts
        concept_indicators = ['and', 'or', 'but', 'however', 'although', 'while']
        if sum(1 for indicator in concept_indicators if indicator in query) > 1:
            return True
        
        return False
    
    def _should_use_expansion(self, query_type: str, query: str) -> bool:
        """Determine if query expansion should be used"""
        # Use expansion for short queries
        if len(query.split()) < 4:
            return True
        
        # Use expansion for conceptual queries
        if query_type == "conceptual":
            return True
        
        # Use expansion if query lacks specificity
        specific_terms = ['specific', 'exact', 'precise', 'detailed', 'particular']
        if not any(term in query for term in specific_terms):
            return True
        
        return False
    
    def _calculate_confidence(self, query_type: str, query: str) -> float:
        """Calculate confidence in the query analysis"""
        # Base confidence
        confidence = 0.7
        
        # Increase confidence for longer queries
        query_length = len(query.split())
        if query_length > 5:
            confidence += 0.1
        if query_length > 10:
            confidence += 0.1
        
        # Increase confidence for clear query types
        if query_type in ["factual", "procedural"]:
            confidence += 0.1
        
        # Decrease confidence for ambiguous queries
        ambiguous_terms = ['maybe', 'perhaps', 'might', 'could', 'possibly']
        if any(term in query for term in ambiguous_terms):
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def get_optimal_params(self, query: str) -> Dict[str, Any]:
        """Get optimal parameters for query processing"""
        analysis = self.analyze_query(query)
        
        return {
            "bm25_weight": analysis["bm25_weight"],
            "vector_weight": analysis["vector_weight"],
            "use_reranking": analysis["use_reranking"],
            "use_expansion": analysis["use_expansion"],
            "top_k": self._get_optimal_top_k(analysis["query_type"]),
            "rerank_top_k": self._get_optimal_rerank_k(analysis["query_type"]),
            "confidence": analysis["confidence"]
        }
    
    def _get_optimal_top_k(self, query_type: str) -> int:
        """Get optimal top-k for initial retrieval"""
        k_mapping = {
            "factual": 8,
            "procedural": 12,
            "conceptual": 10,
            "search": 15
        }
        return k_mapping.get(query_type, 10)
    
    def _get_optimal_rerank_k(self, query_type: str) -> int:
        """Get optimal top-k for reranking"""
        k_mapping = {
            "factual": 5,
            "procedural": 8,
            "conceptual": 6,
            "search": 10
        }
        return k_mapping.get(query_type, 6)
