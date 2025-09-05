import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

from app.core.document_processor import DocumentProcessor
from app.core.embedding_service import EmbeddingService
from app.core.vector_store import VectorStoreManager
from app.core.config import settings

logger = structlog.get_logger()

class IngestionService:
    def __init__(self):
        self.document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()
    
    async def ingest_documents(self, file_paths: List[str], collection_name: str, 
                             batch_size: int = 100, force_reindex: bool = False) -> Dict[str, Any]:
        """Main ingestion pipeline for processing documents"""
        start_time = time.time()
        errors = []
        documents_processed = 0
        chunks_created = 0
        
        try:
            # Validate file paths
            valid_paths = await self._validate_file_paths(file_paths)
            if not valid_paths:
                return {
                    "success": False,
                    "collection_name": collection_name,
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "processing_time_seconds": 0,
                    "errors": ["No valid files found"]
                }
            
            logger.info(f"Starting ingestion of {len(valid_paths)} files into collection '{collection_name}'")
            
            # Process documents in batches
            all_documents = []
            for i in range(0, len(valid_paths), batch_size):
                batch_paths = valid_paths[i:i + batch_size]
                batch_docs = await self.document_processor.process_files(batch_paths, batch_size)
                all_documents.extend(batch_docs)
                documents_processed += len(batch_paths)
                
                logger.info(f"Processed batch {i//batch_size + 1}, total chunks: {len(all_documents)}")
            
            if not all_documents:
                return {
                    "success": False,
                    "collection_name": collection_name,
                    "documents_processed": documents_processed,
                    "chunks_created": 0,
                    "processing_time_seconds": time.time() - start_time,
                    "errors": ["No documents could be processed"]
                }
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(all_documents)} chunks")
            embedded_documents = await self.embedding_service.embed_documents_async(all_documents)
            chunks_created = len(embedded_documents)
            
            # Extract embeddings for vector store
            embeddings = [doc.metadata.get("embedding") for doc in embedded_documents]
            
            # Store in vector database
            logger.info(f"Storing {chunks_created} chunks in vector database")
            success = await self.vector_store.add_documents(
                collection_name, embedded_documents, embeddings
            )
            
            if not success:
                errors.append("Failed to store documents in vector database")
            
            processing_time = time.time() - start_time
            
            result = {
                "success": success and not errors,
                "collection_name": collection_name,
                "documents_processed": documents_processed,
                "chunks_created": chunks_created,
                "processing_time_seconds": round(processing_time, 2),
                "errors": errors
            }
            
            logger.info(f"Ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {
                "success": False,
                "collection_name": collection_name,
                "documents_processed": documents_processed,
                "chunks_created": chunks_created,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "errors": [str(e)]
            }
    
    async def _validate_file_paths(self, file_paths: List[str]) -> List[str]:
        """Validate and filter file paths"""
        valid_paths = []
        
        for path in file_paths:
            try:
                file_path = Path(path)
                if not file_path.exists():
                    logger.warning(f"File not found: {path}")
                    continue
                
                if not file_path.is_file():
                    logger.warning(f"Path is not a file: {path}")
                    continue
                
                # Check file size
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > settings.max_file_size_mb:
                    logger.warning(f"File too large ({file_size_mb:.2f}MB): {path}")
                    continue
                
                # Check file extension
                if file_path.suffix.lower() not in ['.pdf', '.md', '.markdown', '.txt', '.text']:
                    logger.warning(f"Unsupported file type: {path}")
                    continue
                
                valid_paths.append(str(file_path))
                
            except Exception as e:
                logger.warning(f"Error validating {path}: {e}")
                continue
        
        return valid_paths
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            info = await self.vector_store.get_collection_info(collection_name)
            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"name": collection_name, "total_vectors": 0, "dimension": 0}
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            return await self.vector_store.delete_collection(collection_name)
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
