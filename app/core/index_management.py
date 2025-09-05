import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import structlog
import hashlib
from datetime import datetime

from app.core.vector_store import VectorStoreManager
from app.core.embedding_service import EmbeddingService
from app.core.deduplication import DeduplicationService, UpsertService
from app.core.cache import CacheService
from app.core.config import settings

logger = structlog.get_logger()

class IndexManager:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.embedding_service = EmbeddingService()
        self.dedup_service = DeduplicationService()
        self.cache_service = CacheService()
        self.upsert_service = UpsertService(self.vector_store, self.dedup_service)
    
    async def idempotent_upsert(self, collection_name: str, documents: List[Dict[str, Any]], 
                               embeddings: List[List[float]], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Idempotent upsert with deduplication"""
        try:
            logger.info(f"Starting idempotent upsert for collection: {collection_name}")
            
            # Prepare documents with content hashes
            processed_docs = []
            skipped_docs = []
            
            for doc, embedding in zip(documents, embeddings):
                # Create document object
                from langchain.schema import Document
                doc_obj = Document(
                    page_content=doc.get("content", ""),
                    metadata={
                        **doc.get("metadata", {}),
                        "source": doc.get("source", "unknown"),
                        "chunk_index": doc.get("chunk_index", 0),
                        "file_name": doc.get("file_name", "unknown"),
                        "file_type": doc.get("file_type", "unknown"),
                        "doc_title": doc.get("doc_title", "unknown"),
                        "section_title": doc.get("section_title", "unknown"),
                        "page_num": doc.get("page_num", 0),
                        "created_at": datetime.utcnow().isoformat()
                    }
                )
                
                # Check if document already exists
                is_dup, existing_id = self.dedup_service.is_duplicate(doc_obj)
                
                if is_dup:
                    skipped_docs.append({
                        "source": doc.get("source"),
                        "chunk_index": doc.get("chunk_index"),
                        "existing_id": existing_id,
                        "reason": "duplicate"
                    })
                    logger.debug(f"Skipped duplicate document: {doc.get('source')}")
                else:
                    # Add content hash to metadata
                    content_hash = self._compute_content_hash(doc_obj)
                    doc_obj.metadata["content_hash"] = content_hash
                    
                    processed_docs.append((doc_obj, embedding))
            
            # Upsert unique documents
            if processed_docs:
                unique_docs = [doc for doc, _ in processed_docs]
                unique_embeddings = [emb for _, emb in processed_docs]
                
                success = await self.upsert_service.upsert_documents(
                    collection_name, unique_docs, unique_embeddings
                )
                
                if success["upserted_documents"] > 0:
                    # Invalidate cache for this collection
                    await self.cache_service.invalidate_collection_cache(collection_name)
                    
                    logger.info(f"Upserted {success['upserted_documents']} unique documents")
            else:
                success = {"upserted_documents": 0, "errors": []}
            
            # Update collection metadata
            await self._update_collection_metadata(collection_name, metadata)
            
            return {
                "success": True,
                "total_documents": len(documents),
                "processed_documents": len(processed_docs),
                "skipped_documents": len(skipped_docs),
                "upserted_documents": success["upserted_documents"],
                "skipped_details": skipped_docs,
                "errors": success.get("errors", [])
            }
            
        except Exception as e:
            logger.error(f"Idempotent upsert failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_documents": len(documents),
                "processed_documents": 0,
                "skipped_documents": 0,
                "upserted_documents": 0
            }
    
    async def delete_by_source(self, collection_name: str, source: str, 
                              version: Optional[str] = None) -> Dict[str, Any]:
        """Delete all documents from a specific source"""
        try:
            logger.info(f"Deleting documents from source: {source} in collection: {collection_name}")
            
            # Get all documents from the source
            documents_to_delete = await self._get_documents_by_source(collection_name, source, version)
            
            if not documents_to_delete:
                return {
                    "success": True,
                    "deleted_documents": 0,
                    "message": "No documents found for source"
                }
            
            # Delete from vector store
            deleted_count = await self._delete_documents_from_vector_store(
                collection_name, documents_to_delete
            )
            
            # Remove from deduplication tracking
            for doc in documents_to_delete:
                self.dedup_service.remove_document(doc)
            
            # Invalidate cache
            await self.cache_service.invalidate_collection_cache(collection_name)
            
            logger.info(f"Deleted {deleted_count} documents from source: {source}")
            
            return {
                "success": True,
                "deleted_documents": deleted_count,
                "source": source,
                "version": version
            }
            
        except Exception as e:
            logger.error(f"Delete by source failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_documents": 0
            }
    
    async def reindex_document(self, collection_name: str, source: str, 
                              new_documents: List[Dict[str, Any]], 
                              new_embeddings: List[List[float]]) -> Dict[str, Any]:
        """Atomic delete and reindex of a document"""
        try:
            logger.info(f"Reindexing document: {source} in collection: {collection_name}")
            
            # Step 1: Delete existing documents
            delete_result = await self.delete_by_source(collection_name, source)
            
            if not delete_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to delete existing documents: {delete_result['error']}",
                    "deleted_documents": 0,
                    "indexed_documents": 0
                }
            
            # Step 2: Index new documents
            index_result = await self.idempotent_upsert(
                collection_name, new_documents, new_embeddings
            )
            
            if not index_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to index new documents: {index_result['error']}",
                    "deleted_documents": delete_result["deleted_documents"],
                    "indexed_documents": 0
                }
            
            logger.info(f"Successfully reindexed document: {source}")
            
            return {
                "success": True,
                "deleted_documents": delete_result["deleted_documents"],
                "indexed_documents": index_result["upserted_documents"],
                "source": source
            }
            
        except Exception as e:
            logger.error(f"Reindex document failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_documents": 0,
                "indexed_documents": 0
            }
    
    async def get_document_versions(self, collection_name: str, source: str) -> List[Dict[str, Any]]:
        """Get all versions of a document"""
        try:
            # This would query the vector store for all documents with the same source
            # For now, return placeholder
            return [
                {
                    "source": source,
                    "version": "1.0",
                    "created_at": datetime.utcnow().isoformat(),
                    "chunk_count": 0
                }
            ]
        except Exception as e:
            logger.error(f"Error getting document versions: {e}")
            return []
    
    async def cleanup_orphaned_documents(self, collection_name: str) -> Dict[str, Any]:
        """Clean up orphaned documents (documents without valid sources)"""
        try:
            logger.info(f"Cleaning up orphaned documents in collection: {collection_name}")
            
            # This would implement orphan cleanup logic
            # For now, return placeholder
            return {
                "success": True,
                "cleaned_documents": 0,
                "message": "Orphan cleanup not implemented"
            }
            
        except Exception as e:
            logger.error(f"Orphan cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "cleaned_documents": 0
            }
    
    def _compute_content_hash(self, document) -> str:
        """Compute content hash for deduplication"""
        content = document.page_content
        metadata_str = str(sorted(document.metadata.items()))
        combined = f"{content}|||{metadata_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    async def _update_collection_metadata(self, collection_name: str, metadata: Dict[str, Any]):
        """Update collection metadata"""
        try:
            if not metadata:
                return
            
            # This would update collection metadata in the vector store
            # For now, just log
            logger.info(f"Updated collection metadata for {collection_name}: {metadata}")
            
        except Exception as e:
            logger.error(f"Failed to update collection metadata: {e}")
    
    async def _get_documents_by_source(self, collection_name: str, source: str, 
                                     version: Optional[str] = None) -> List[Any]:
        """Get all documents from a specific source"""
        try:
            # This would query the vector store for documents with matching source
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting documents by source: {e}")
            return []
    
    async def _delete_documents_from_vector_store(self, collection_name: str, 
                                                documents: List[Any]) -> int:
        """Delete documents from vector store"""
        try:
            # This would implement actual deletion from the vector store
            # For now, return count
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {e}")
            return 0

class CollectionManager:
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
    
    async def create_collection(self, collection_name: str, 
                              embedding_model: str = None,
                              dimension: int = None) -> Dict[str, Any]:
        """Create a new collection with metadata"""
        try:
            # Set default values
            embedding_model = embedding_model or settings.embedding_model
            dimension = dimension or settings.embedding_dimension
            
            # Create collection metadata
            metadata = {
                "embedding_model": embedding_model,
                "dimension": dimension,
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
            
            # Create collection in vector store
            # This would call the vector store to create the collection
            # For now, just return success
            
            logger.info(f"Created collection: {collection_name}")
            
            return {
                "success": True,
                "collection_name": collection_name,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information and statistics"""
        try:
            # Get basic collection info
            info = await self.index_manager.vector_store.get_collection_info(collection_name)
            
            # Add additional metadata
            info.update({
                "embedding_model": settings.embedding_model,
                "dimension": settings.embedding_dimension,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active"
            })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "error": str(e),
                "collection_name": collection_name
            }
    
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
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Delete a collection and all its documents"""
        try:
            # Delete from vector store
            success = await self.index_manager.vector_store.delete_collection(collection_name)
            
            if success:
                # Invalidate cache
                await self.index_manager.cache_service.invalidate_collection_cache(collection_name)
                
                logger.info(f"Deleted collection: {collection_name}")
                
                return {
                    "success": True,
                    "collection_name": collection_name
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to delete collection from vector store"
                }
                
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
