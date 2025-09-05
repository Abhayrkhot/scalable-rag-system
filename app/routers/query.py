from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import structlog
import json
import asyncio

from app.services.query_service import QueryService
from app.utils.auth import verify_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])

# Initialize service
query_service = QueryService()

class QueryRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = 10
    use_hybrid: bool = True  # Default to hybrid search
    use_reranking: bool = True  # Default to reranking
    use_query_expansion: bool = True
    use_planning: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    contexts: List[str]
    confidence_score: float
    processing_time_seconds: float
    tokens_used: int
    latency_breakdown: Dict[str, float]
    search_strategy: str
    query_plan: Optional[Dict[str, Any]] = None

@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    _: None = Depends(verify_api_key)
):
    """Query documents with hybrid search and reranking by default"""
    try:
        logger.info(f"Processing query for collection: {request.collection_name}")
        
        result = await query_service.answer_question(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_reranking=request.use_reranking,
            use_query_expansion=request.use_query_expansion,
            use_planning=request.use_planning
        )
        
        logger.info(f"Query processed successfully: {len(result.get('sources', []))} sources found")
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def query_documents_streaming(
    request: QueryRequest,
    _: None = Depends(verify_api_key)
):
    """Stream query response as Server-Sent Events"""
    try:
        logger.info(f"Processing streaming query for collection: {request.collection_name}")
        
        async def generate_stream():
            try:
                # Send initial metadata
                yield f"data: {json.dumps({'type': 'start', 'message': 'Starting query processing...'})}\n\n"
                
                # Stream the answer
                async for chunk in query_service.answer_question_streaming(
                    question=request.question,
                    collection_name=request.collection_name,
                    top_k=request.top_k,
                    use_hybrid=request.use_hybrid,
                    use_reranking=request.use_reranking
                ):
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done', 'message': 'Query completed'})}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def batch_queries(
    queries: List[QueryRequest],
    _: None = Depends(verify_api_key)
):
    """Process multiple queries in batch"""
    try:
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing batch query {i+1}/{len(queries)}")
            
            result = await query_service.answer_question(
                question=query.question,
                collection_name=query.collection_name,
                top_k=query.top_k,
                use_hybrid=query.use_hybrid,
                use_reranking=query.use_reranking,
                use_query_expansion=query.use_query_expansion,
                use_planning=query.use_planning
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

@router.get("/collections/{collection_name}/stats")
async def get_collection_stats(
    collection_name: str,
    _: None = Depends(verify_api_key)
):
    """Get statistics about a collection"""
    try:
        from app.services.ingestion_service import IngestionService
        ingestion_service = IngestionService()
        
        info = await ingestion_service.get_collection_info(collection_name)
        return {
            "collection_name": collection_name,
            "total_documents": info.get("total_vectors", 0),
            "dimension": info.get("dimension", 0),
            "status": "active" if info.get("total_vectors", 0) > 0 else "empty"
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
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
                "recall_score": 0.88,
                "hybrid_search_usage": 0.75,
                "reranking_usage": 0.80
            }
        }
    except Exception as e:
        logger.error(f"Error getting search quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug")
async def debug_query(
    request: QueryRequest,
    _: None = Depends(verify_api_key)
):
    """Debug query with detailed information"""
    try:
        logger.info(f"Processing debug query for collection: {request.collection_name}")
        
        # Get detailed query information
        result = await query_service.answer_question(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_reranking=request.use_reranking,
            use_query_expansion=request.use_query_expansion,
            use_planning=request.use_planning
        )
        
        # Add debug information
        debug_info = {
            "query_analysis": {
                "question": request.question,
                "collection": request.collection_name,
                "parameters": {
                    "top_k": request.top_k,
                    "use_hybrid": request.use_hybrid,
                    "use_reranking": request.use_reranking,
                    "use_query_expansion": request.use_query_expansion,
                    "use_planning": request.use_planning
                }
            },
            "result": result,
            "debug_timestamp": asyncio.get_event_loop().time()
        }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
