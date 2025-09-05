import asyncio
import logging
from typing import List, Dict, Any, Optional
import structlog
from pathlib import Path

from app.core.document_processor import load_document, clean_text, chunk_document
from app.core.embedding_service import EmbeddingService
from app.core.vector_store import VectorStoreManager
from app.core.config import settings
from app.utils.monitoring import record_ingestion_metrics

logger = structlog.get_logger()

class IngestionService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()
    
    async def ingest_documents(self, file_paths: List[str], collection_name: str,
                              chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """Ingest documents into the RAG system"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Starting ingestion of {len(file_paths)} files into collection '{collection_name}'")
            
            all_documents = []
            all_embeddings = []
            processed_files = 0
            
            for file_path in file_paths:
                try:
                    # Load and process document
                    doc_result = await self.load_and_process_document(
                        file_path, collection_name, chunk_size, chunk_overlap
                    )
                    
                    all_documents.extend(doc_result['documents'])
                    all_embeddings.extend(doc_result['embeddings'])
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            if not all_documents:
                return {
                    "success": False,
                    "error": "No documents were successfully processed",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "processing_time_seconds": 0
                }
            
            # Store in vector database
            success = await self.vector_store.add_documents(
                collection_name, all_documents, all_embeddings
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Record metrics
            record_ingestion_metrics(
                collection_name, "mixed", len(all_documents), 
                len(all_documents), processing_time
            )
            
            result = {
                "success": success,
                "documents_processed": processed_files,
                "chunks_created": len(all_documents),
                "processing_time_seconds": round(processing_time, 2),
                "collection_name": collection_name
            }
            
            logger.info(f"Ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0,
                "chunks_created": 0,
                "processing_time_seconds": 0
            }
    
    async def load_and_process_document(self, file_path: str, collection_name: str,
                                      chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """Load and process a single document"""
        try:
            # Load document
            document = load_document(file_path)
            
            # Clean text
            cleaned_doc = clean_text(document)
            
            # Chunk document
            chunks = chunk_document(cleaned_doc, chunk_size, chunk_overlap)
            
            # Generate embeddings
            embeddings = await self.embedding_service.embed_documents_batch(chunks)
            
            return {
                "documents": chunks,
                "embeddings": embeddings
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information"""
        try:
            return await self.vector_store.get_collection_info(collection_name)
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections"""
        try:
            # This would list all collections from the vector store
            # For now, return placeholder
            return [
                {
                    "name": "default",
                    "document_count": 0,
                    "status": "active"
                }
            ]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
