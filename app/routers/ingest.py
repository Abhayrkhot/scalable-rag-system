from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
import structlog

from app.models.schemas import IngestRequest, IngestResponse, ErrorResponse
from app.services.ingestion_service import IngestionService
from app.utils.auth import verify_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Initialize service
ingestion_service = IngestionService()

@router.post("/", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key)
):
    """Ingest documents into the vector database"""
    try:
        logger.info(f"Starting ingestion for collection: {request.collection_name}")
        
        result = await ingestion_service.ingest_documents(
            file_paths=request.file_paths,
            collection_name=request.collection_name,
            batch_size=request.batch_size,
            force_reindex=request.force_reindex
        )
        
        if result["success"]:
            logger.info(f"Ingestion completed successfully: {result}")
        else:
            logger.error(f"Ingestion failed: {result}")
        
        return IngestResponse(**result)
        
    except Exception as e:
        logger.error(f"Ingestion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections/{collection_name}")
async def get_collection_info(
    collection_name: str,
    _: None = Depends(verify_api_key)
):
    """Get information about a collection"""
    try:
        info = await ingestion_service.get_collection_info(collection_name)
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    _: None = Depends(verify_api_key)
):
    """Delete a collection"""
    try:
        success = await ingestion_service.delete_collection(collection_name)
        if success:
            return {"message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete collection")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
