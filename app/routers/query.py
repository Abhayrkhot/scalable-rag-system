from fastapi import APIRouter, HTTPException, Depends
import structlog

from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse
from app.services.query_service import QueryService
from app.utils.auth import verify_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])

# Initialize service
query_service = QueryService()

@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    _: None = Depends(verify_api_key)
):
    """Query documents and get AI-generated answers"""
    try:
        logger.info(f"Processing query for collection: {request.collection_name}")
        
        result = await query_service.answer_question(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            rerank=request.rerank,
            include_metadata=request.include_metadata
        )
        
        logger.info(f"Query processed successfully: {len(result.get('sources', []))} sources found")
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
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
