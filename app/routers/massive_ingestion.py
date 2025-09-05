from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
import structlog

from app.models.schemas import IngestRequest, IngestResponse, ErrorResponse
from app.services.massive_ingestion_service import MassiveIngestionService
from app.utils.auth import verify_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/massive", tags=["massive-ingestion"])

# Initialize service
massive_service = MassiveIngestionService()

@router.post("/ingest-million", response_model=IngestResponse)
async def ingest_million_documents(
    collection_name: str,
    batch_size: int = 1000,
    max_workers: int = 10,
    background_tasks: BackgroundTasks = None,
    _: None = Depends(verify_api_key)
):
    """Ingest 1 million documents for large-scale processing"""
    try:
        logger.info(f"Starting massive ingestion for collection: {collection_name}")
        
        # Run in background for large-scale processing
        if background_tasks:
            background_tasks.add_task(
                massive_service.process_million_documents,
                collection_name=collection_name,
                batch_size=batch_size,
                max_workers=max_workers
            )
            return IngestResponse(
                success=True,
                collection_name=collection_name,
                documents_processed=0,
                chunks_created=0,
                processing_time_seconds=0,
                errors=[]
            )
        else:
            result = await massive_service.process_million_documents(
                collection_name=collection_name,
                batch_size=batch_size,
                max_workers=max_workers
            )
            return IngestResponse(**result)
        
    except Exception as e:
        logger.error(f"Massive ingestion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/{collection_name}")
async def get_processing_stats(
    collection_name: str,
    _: None = Depends(verify_api_key)
):
    """Get current processing statistics"""
    try:
        stats = await massive_service.get_processing_stats()
        return {
            "collection_name": collection_name,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-dataset")
async def generate_large_dataset(
    document_count: int = 1000000,
    _: None = Depends(verify_api_key)
):
    """Generate a large dataset for testing"""
    try:
        from app.utils.dataset_generator import LargeDatasetGenerator
        generator = LargeDatasetGenerator()
        
        result = await generator.generate_million_documents(document_count)
        
        return {
            "success": True,
            "dataset_info": result,
            "message": f"Generated {result['total_documents']:,} documents"
        }
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
