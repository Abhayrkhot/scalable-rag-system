from celery import current_task
from app.celery_app import celery_app
from app.services.ingestion_service import IngestionService
import structlog

logger = structlog.get_logger()

@celery_app.task(bind=True)
def ingest_documents_async(self, file_paths, collection_name, batch_size=100, force_reindex=False):
    """Async task for document ingestion"""
    try:
        ingestion_service = IngestionService()
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"status": "Starting ingestion", "progress": 0}
        )
        
        # Run ingestion
        result = await ingestion_service.ingest_documents(
            file_paths=file_paths,
            collection_name=collection_name,
            batch_size=batch_size,
            force_reindex=force_reindex
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Async ingestion failed: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise

@celery_app.task
def cleanup_old_tasks():
    """Cleanup old completed tasks"""
    try:
        # This would implement cleanup logic
        logger.info("Cleaning up old tasks")
        return "Cleanup completed"
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise
