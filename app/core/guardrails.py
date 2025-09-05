import logging
from typing import List, Dict, Any, Optional, Tuple
import structlog
from pydantic import BaseModel, Field
import tiktoken

logger = structlog.get_logger()

class GuardrailConfig(BaseModel):
    max_tokens: int = Field(default=4000, description="Maximum tokens per response")
    max_context_tokens: int = Field(default=8000, description="Maximum context tokens")
    require_citations: bool = Field(default=True, description="Require citations in responses")
    forbid_unverifiable: bool = Field(default=True, description="Forbid unverifiable claims")
    min_confidence_threshold: float = Field(default=0.3, description="Minimum confidence threshold")
    max_sources: int = Field(default=10, description="Maximum number of sources to use")

class GuardrailService:
    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def validate_context_size(self, context: str) -> Tuple[bool, str]:
        """Validate context size against limits"""
        token_count = self.count_tokens(context)
        
        if token_count > self.config.max_context_tokens:
            return False, f"Context too large: {token_count} tokens (max: {self.config.max_context_tokens})"
        
        return True, ""
    
    def validate_response_size(self, response: str) -> Tuple[bool, str]:
        """Validate response size against limits"""
        token_count = self.count_tokens(response)
        
        if token_count > self.config.max_tokens:
            return False, f"Response too large: {token_count} tokens (max: {self.config.max_tokens})"
        
        return True, ""
    
    def create_guardrail_prompt(self, question: str, context: str, sources: List[Dict]) -> str:
        """Create a prompt with guardrails"""
        
        # Validate context size
        is_valid, error = self.validate_context_size(context)
        if not is_valid:
            logger.warning(f"Context size validation failed: {error}")
            # Truncate context if too large
            context = self._truncate_context(context)
        
        # Limit sources
        limited_sources = sources[:self.config.max_sources]
        
        prompt = f"""You are a helpful AI assistant with access to a knowledge base. Answer the following question using ONLY the provided context.

IMPORTANT RULES:
1. You MUST cite your sources using "According to Source X..." format
2. If you cannot find information in the provided context, say "I don't have enough information in the provided context to answer this question accurately"
3. Do NOT make claims that cannot be verified from the provided sources
4. If the context is insufficient, acknowledge this limitation
5. Be precise and factual in your response
6. Keep your response concise but comprehensive

Question: {question}

Context from {len(limited_sources)} sources:
{context}

Sources:
{self._format_sources(limited_sources)}

Answer (with citations):"""

        return prompt
    
    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit within token limits"""
        # Simple truncation - in production, you'd want smarter chunking
        max_chars = self.config.max_context_tokens * 4  # Rough estimate
        if len(context) > max_chars:
            context = context[:max_chars] + "..."
        return context
    
    def _format_sources(self, sources: List[Dict]) -> str:
        """Format sources for the prompt"""
        formatted = []
        for i, source in enumerate(sources, 1):
            source_info = f"Source {i}: {source.get('source', 'Unknown')}"
            if 'relevance_score' in source:
                source_info += f" (Relevance: {source['relevance_score']:.2f})"
            formatted.append(source_info)
        return "\n".join(formatted)
    
    def validate_response(self, response: str, sources: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate response against guardrails"""
        issues = []
        
        # Check response size
        is_valid, error = self.validate_response_size(response)
        if not is_valid:
            issues.append(error)
        
        # Check for citations if required
        if self.config.require_citations:
            if not self._has_citations(response):
                issues.append("Response must include citations")
        
        # Check for unverifiable claims
        if self.config.forbid_unverifiable:
            unverifiable_phrases = [
                "I believe", "I think", "In my opinion", "It seems like",
                "Probably", "Maybe", "Perhaps", "I assume"
            ]
            if any(phrase in response for phrase in unverifiable_phrases):
                issues.append("Response contains unverifiable claims")
        
        # Check minimum confidence
        if hasattr(self, '_last_confidence') and self._last_confidence < self.config.min_confidence_threshold:
            issues.append(f"Confidence too low: {self._last_confidence:.2f}")
        
        return len(issues) == 0, issues
    
    def _has_citations(self, response: str) -> bool:
        """Check if response contains citations"""
        citation_patterns = [
            "According to Source",
            "Source",
            "Document",
            "As mentioned in",
            "Based on the provided"
        ]
        return any(pattern in response for pattern in citation_patterns)
    
    def create_fallback_response(self, question: str, reason: str) -> str:
        """Create a fallback response when guardrails fail"""
        return f"""I apologize, but I cannot provide a complete answer to your question: "{question}"

Reason: {reason}

Please try:
1. Rephrasing your question
2. Providing more specific context
3. Asking a different question

If you continue to have issues, please contact support."""

class StreamingResponse:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.tokens_generated = 0
    
    def should_continue(self, new_tokens: int) -> bool:
        """Check if streaming should continue"""
        self.tokens_generated += new_tokens
        return self.tokens_generated < self.max_tokens
    
    def get_remaining_tokens(self) -> int:
        """Get remaining tokens before hitting limit"""
        return max(0, self.max_tokens - self.tokens_generated)

class ContentFilter:
    def __init__(self):
        self.blocked_patterns = [
            # Add patterns for inappropriate content
            r'\b(?:hack|exploit|vulnerability)\b',
            # Add more patterns as needed
        ]
    
    def is_safe(self, text: str) -> Tuple[bool, str]:
        """Check if content is safe"""
        import re
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Content blocked due to pattern: {pattern}"
        
        return True, ""
    
    def filter_response(self, response: str) -> str:
        """Filter response for safety"""
        is_safe, reason = self.is_safe(response)
        
        if not is_safe:
            logger.warning(f"Content filtered: {reason}")
            return "I cannot provide that information due to content policy restrictions."
        
        return response
