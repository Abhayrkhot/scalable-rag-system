from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import List, Optional, Dict, Any
import structlog
import asyncio
from pathlib import Path

from app.services.ingestion_service import IngestionService
from app.core.index_management import IndexManager
from app.utils.auth import verify_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Initialize services
ingestion_service = IngestionService()
index_manager = IndexManager()

class IngestRequest:
    def __init__(self, paths: List[str], collection: str, chunk_size: int = 1000, overlap: int = 200):
        self.paths = paths
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap = overlap

@router.post("/")
async def ingest_files(
    collection: str = Form(...),
    chunk_size: int = Form(1000),
    overlap: int = Form(200),
    files: List[UploadFile] = File(...),
    _: None = Depends(verify_api_key)
):
    """Ingest files into the RAG system"""
    try:
        logger.info(f"Starting ingestion for collection: {collection}")
        
        # Save uploaded files temporarily
        temp_paths = []
        for file in files:
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            temp_paths.append(temp_path)
        
        # Process files
        result = await ingestion_service.ingest_documents(
            file_paths=temp_paths,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        
        # Cleanup temp files
        for temp_path in temp_paths:
            Path(temp_path).unlink(missing_ok=True)
        
        logger.info(f"Ingestion completed: {result['documents_processed']} documents processed")
        return result
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reindex_document")
async def reindex_document(
    collection: str = Form(...),
    source: str = Form(...),
    version: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    _: None = Depends(verify_api_key)
):
    """Reindex a specific document (atomic delete + ingest)"""
    try:
        logger.info(f"Reindexing document {source} in collection {collection}")
        
        # Save uploaded files temporarily
        temp_paths = []
        for file in files:
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            temp_paths.append(temp_path)
        
        # Process files to get documents and embeddings
        documents = []
        embeddings = []
        
        for temp_path in temp_paths:
            # Load and process document
            doc_result = await ingestion_service.load_and_process_document(
                file_path=temp_path,
                collection_name=collection
            )
            documents.extend(doc_result['documents'])
            embeddings.extend(doc_result['embeddings'])
        
        # Reindex using index manager
        result = await index_manager.reindex_document(
            collection_name=collection,
            source=source,
            new_documents=documents,
            new_embeddings=embeddings
        )
        
        # Cleanup temp files
        for temp_path in temp_paths:
            Path(temp_path).unlink(missing_ok=True)
        
        logger.info(f"Reindex completed: {result['indexed_documents']} documents indexed")
        return result
        
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collections/{collection_name}/sources/{source}")
async def delete_document_source(
    collection_name: str,
    source: str,
    version: Optional[str] = None,
    _: None = Depends(verify_api_key)
):
    """Delete all documents from a specific source"""
    try:
        logger.info(f"Deleting documents from source {source} in collection {collection_name}")
        
        result = await index_manager.delete_by_source(
            collection_name=collection_name,
            source=source,
            version=version
        )
        
        if result['success']:
            logger.info(f"Deleted {result['deleted_documents']} documents from source {source}")
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
    except Exception as e:
        logger.error(f"Delete source error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections/{collection_name}/info")
async def get_collection_info(
    collection_name: str,
    _: None = Depends(verify_api_key)
):
    """Get collection information"""
    try:
        info = await ingestion_service.get_collection_info(collection_name)
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections")
async def list_collections(
    _: None = Depends(verify_api_key)
):
    """List all collections"""
    try:
        collections = await ingestion_service.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))
