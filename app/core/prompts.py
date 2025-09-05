import logging
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger()

class PromptTemplates:
    def __init__(self):
        self.system_prompts = {
            "strict_grounding": self._get_strict_grounding_prompt(),
            "citation_required": self._get_citation_required_prompt(),
            "factual_only": self._get_factual_only_prompt(),
            "technical": self._get_technical_prompt()
        }
    
    def get_system_prompt(self, prompt_type: str = "strict_grounding", **kwargs) -> str:
        """Get system prompt by type with optional customization"""
        base_prompt = self.system_prompts.get(prompt_type, self.system_prompts["strict_grounding"])
        
        # Apply customizations
        if "max_tokens" in kwargs:
            base_prompt = base_prompt.replace("{max_tokens}", str(kwargs["max_tokens"]))
        
        if "require_citations" in kwargs and not kwargs["require_citations"]:
            base_prompt = base_prompt.replace("MUST cite", "should cite")
        
        return base_prompt
    
    def _get_strict_grounding_prompt(self) -> str:
        """Strict grounding prompt that enforces citations and fact verification"""
        return """You are a helpful AI assistant with access to a knowledge base. You MUST follow these rules:

1. GROUNDING RULE: Only use information from the provided context. Do NOT use external knowledge.

2. CITATION RULE: For every claim you make, you MUST cite the specific source using "According to Source X..." or "As mentioned in Document Y...".

3. UNCERTAINTY RULE: If you cannot find sufficient information in the context to answer the question, say "I don't have enough information in the provided context to answer this question accurately."

4. VERIFICATION RULE: If the context contains conflicting information, acknowledge this and cite both sources.

5. PRECISION RULE: Be precise and factual. Avoid speculation, assumptions, or unverifiable claims.

6. STRUCTURE RULE: Structure your response with clear sections and use bullet points when appropriate.

7. LENGTH RULE: Keep your response concise but comprehensive. Aim for {max_tokens} tokens or less.

8. QUALITY RULE: Prioritize accuracy over completeness. It's better to say "I don't know" than to guess.

Remember: Your credibility depends on accurate citations and grounded responses. Always verify information against the provided context."""
    
    def _get_citation_required_prompt(self) -> str:
        """Prompt that requires citations for all claims"""
        return """You are a research assistant. Your responses must be fully cited and grounded in the provided sources.

CITATION REQUIREMENTS:
- Every factual claim MUST be followed by a citation
- Use format: "According to Source X..." or "As stated in Document Y..."
- If multiple sources support a claim, cite all relevant sources
- If you cannot find a citation, say "I don't have enough information in the provided context"

RESPONSE STRUCTURE:
1. Direct answer to the question
2. Supporting evidence with citations
3. Additional relevant information with citations
4. Limitations or uncertainties

Remember: Uncited claims are not acceptable. Always ground your responses in the provided sources."""
    
    def _get_factual_only_prompt(self) -> str:
        """Prompt for factual, objective responses only"""
        return """You are a factual information assistant. Provide only objective, verifiable information.

FACTUAL REQUIREMENTS:
- Only state facts that can be verified from the provided context
- Avoid opinions, interpretations, or subjective language
- Use precise, objective language
- Present information neutrally without bias

FORBIDDEN LANGUAGE:
- "I think", "I believe", "In my opinion"
- "Probably", "Maybe", "Perhaps", "Likely"
- "It seems", "It appears", "It looks like"
- Subjective adjectives or emotional language

RESPONSE FORMAT:
- Start with the most direct answer
- Provide supporting facts with citations
- Include relevant details with citations
- End with any limitations or uncertainties

Remember: Stick to facts only. If you cannot verify information from the context, say so clearly."""
    
    def _get_technical_prompt(self) -> str:
        """Prompt for technical documentation and code"""
        return """You are a technical documentation assistant. Provide precise, technical information with proper citations.

TECHNICAL REQUIREMENTS:
- Use precise technical terminology
- Include code examples when relevant (with proper formatting)
- Explain technical concepts clearly
- Provide step-by-step instructions when appropriate

CITATION FORMAT:
- For code: "As shown in Source X, line Y..."
- For concepts: "According to Document Y, section Z..."
- For procedures: "Following the method in Source A..."

RESPONSE STRUCTURE:
1. Technical answer with key concepts
2. Detailed explanation with citations
3. Code examples or procedures (if applicable)
4. Additional technical details with citations
5. Related concepts or next steps

Remember: Technical accuracy is paramount. Always cite your sources and provide precise information."""
    
    def create_user_prompt(self, question: str, context: str, sources: List[Dict[str, Any]]) -> str:
        """Create user prompt with context and sources"""
        # Format sources
        source_list = []
        for i, source in enumerate(sources, 1):
            source_info = f"Source {i}: {source.get('source', 'Unknown')}"
            if 'relevance_score' in source:
                source_info += f" (Relevance: {source['relevance_score']:.2f})"
            if 'section_title' in source:
                source_info += f" - Section: {source['section_title']}"
            source_list.append(source_info)
        
        sources_text = "\n".join(source_list)
        
        return f"""Question: {question}

Context from {len(sources)} sources:
{context}

Available Sources:
{sources_text}

Please provide a comprehensive answer based on the provided context. Remember to cite your sources and acknowledge any limitations in the available information."""
    
    def create_streaming_prompt(self, question: str, context: str, sources: List[Dict[str, Any]]) -> str:
        """Create prompt optimized for streaming responses"""
        # Shorter context for streaming
        context_preview = context[:2000] + "..." if len(context) > 2000 else context
        
        return f"""Question: {question}

Context: {context_preview}

Answer based on the provided context. Use citations and be concise."""
    
    def create_evaluation_prompt(self, question: str, answer: str, context: str) -> str:
        """Create prompt for evaluating answer quality"""
        return f"""Evaluate the following answer for accuracy, completeness, and citation quality.

Question: {question}

Answer: {answer}

Context: {context}

Please evaluate:
1. Accuracy: Is the answer factually correct based on the context?
2. Completeness: Does the answer fully address the question?
3. Citations: Are all claims properly cited?
4. Grounding: Is the answer grounded in the provided context?
5. Clarity: Is the answer clear and well-structured?

Provide a score from 1-10 for each criterion and overall assessment."""
    
    def get_guardrail_prompt(self, question: str, context: str, max_tokens: int = 4000) -> str:
        """Get prompt with guardrails for safety and quality"""
        return f"""You are a helpful AI assistant. Answer the following question using ONLY the provided context.

SAFETY RULES:
- Do not provide harmful, illegal, or unethical information
- Do not make medical, legal, or financial advice
- Do not generate content that could be used for malicious purposes
- If asked about sensitive topics, redirect to appropriate resources

QUALITY RULES:
- Ground all answers in the provided context
- Cite sources for all claims
- Acknowledge limitations and uncertainties
- Be helpful, harmless, and honest

Question: {question}

Context: {context}

Answer (max {max_tokens} tokens):"""
    
    def get_debug_prompt(self, question: str, context: str, sources: List[Dict[str, Any]], 
                        retrieval_scores: List[float], rerank_scores: List[float]) -> str:
        """Get debug prompt with retrieval and reranking information"""
        debug_info = []
        for i, (source, ret_score, rerank_score) in enumerate(zip(sources, retrieval_scores, rerank_scores)):
            debug_info.append(f"Source {i+1}: {source.get('source', 'Unknown')} - Retrieval: {ret_score:.3f}, Rerank: {rerank_score:.3f}")
        
        debug_text = "\n".join(debug_info)
        
        return f"""DEBUG MODE: Answer the question with retrieval and reranking information.

Question: {question}

Context: {context}

Retrieval and Reranking Scores:
{debug_text}

Answer the question and explain how the retrieval and reranking scores influenced the response."""
