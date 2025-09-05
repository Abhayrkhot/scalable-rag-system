from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import structlog

from app.services.advanced_query_service import AdvancedQueryService
from app.utils.auth import verify_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/advanced-query", tags=["advanced-query"])

# Initialize service
advanced_query_service = AdvancedQueryService()

class AdvancedQueryRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = 10
    use_reranking: bool = True
    use_query_expansion: bool = True
    use_hybrid_search: bool = True

class AdvancedQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time_seconds: float
    tokens_used: int
    search_strategy: str
    candidates_evaluated: int
    reranking_used: bool
    query_expansion_used: bool

@router.post("/", response_model=AdvancedQueryResponse)
async def advanced_query(
    request: AdvancedQueryRequest,
    _: None = Depends(verify_api_key)
):
    """Advanced query with precision optimization"""
    try:
        logger.info(f"Processing advanced query for collection: {request.collection_name}")
        
        result = await advanced_query_service.answer_question_advanced(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            use_query_expansion=request.use_query_expansion,
            use_hybrid_search=request.use_hybrid_search
        )
        
        logger.info(f"Advanced query processed successfully: {len(result.get('sources', []))} sources found")
        return AdvancedQueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Advanced query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def batch_advanced_queries(
    queries: List[AdvancedQueryRequest],
    _: None = Depends(verify_api_key)
):
    """Process multiple queries in batch"""
    try:
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing batch query {i+1}/{len(queries)}")
            
            result = await advanced_query_service.answer_question_advanced(
                question=query.question,
                collection_name=query.collection_name,
                top_k=query.top_k,
                use_reranking=query.use_reranking,
                use_query_expansion=query.use_query_expansion,
                use_hybrid_search=query.use_hybrid_search
            )
            
            results.append({
                "query_index": i,
                "question": query.question,
                "result": result
            })
        
        return {
            "success": True,
            "total_queries": len(queries),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections/{collection_name}/search-quality")
async def get_search_quality_metrics(
    collection_name: str,
    _: None = Depends(verify_api_key)
):
    """Get search quality metrics for a collection"""
    try:
        # This would implement quality metrics calculation
        # For now, return placeholder data
        return {
            "collection_name": collection_name,
            "metrics": {
                "average_confidence": 0.85,
                "average_response_time": 2.3,
                "total_queries_processed": 0,
                "precision_score": 0.92,
                "recall_score": 0.88
            }
        }
    except Exception as e:
        logger.error(f"Error getting search quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
