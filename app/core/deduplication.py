import hashlib
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import structlog
from langchain.schema import Document

logger = structlog.get_logger()

class DeduplicationService:
    def __init__(self):
        self.processed_hashes: Set[str] = set()
        self.hash_to_doc_id: Dict[str, str] = {}
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common punctuation that might vary
        import re
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def compute_content_hash(self, document: Document) -> str:
        """Compute SHA256 hash of normalized document content"""
        normalized_text = self.normalize_text(document.page_content)
        
        # Include metadata in hash for uniqueness
        metadata_str = str(sorted(document.metadata.items()))
        
        content = f"{normalized_text}|||{metadata_str}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, document: Document) -> Tuple[bool, Optional[str]]:
        """Check if document is a duplicate"""
        content_hash = self.compute_content_hash(document)
        
        if content_hash in self.processed_hashes:
            existing_doc_id = self.hash_to_doc_id.get(content_hash)
            return True, existing_doc_id
        
        return False, None
    
    def add_document(self, document: Document, doc_id: str) -> bool:
        """Add document to deduplication tracking"""
        content_hash = self.compute_content_hash(document)
        
        if content_hash in self.processed_hashes:
            return False  # Duplicate
        
        self.processed_hashes.add(content_hash)
        self.hash_to_doc_id[content_hash] = doc_id
        return True
    
    def remove_document(self, document: Document) -> bool:
        """Remove document from deduplication tracking"""
        content_hash = self.compute_content_hash(document)
        
        if content_hash in self.processed_hashes:
            self.processed_hashes.remove(content_hash)
            if content_hash in self.hash_to_doc_id:
                del self.hash_to_doc_id[content_hash]
            return True
        
        return False
    
    def deduplicate_documents(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """Separate unique and duplicate documents"""
        unique_docs = []
        duplicate_docs = []
        
        for doc in documents:
            is_dup, existing_id = self.is_duplicate(doc)
            
            if is_dup:
                duplicate_docs.append(doc)
                logger.debug(f"Found duplicate document, existing ID: {existing_id}")
            else:
                unique_docs.append(doc)
        
        logger.info(f"Deduplication: {len(unique_docs)} unique, {len(duplicate_docs)} duplicates")
        return unique_docs, duplicate_docs
    
    def get_duplicate_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return {
            "total_processed": len(self.processed_hashes),
            "unique_documents": len(self.processed_hashes),
            "duplicate_rate": 0.0  # Would need to track total processed vs unique
        }
    
    def clear_tracking(self):
        """Clear all deduplication tracking"""
        self.processed_hashes.clear()
        self.hash_to_doc_id.clear()
        logger.info("Deduplication tracking cleared")

class UpsertService:
    def __init__(self, vector_store, deduplication_service: DeduplicationService):
        self.vector_store = vector_store
        self.dedup_service = deduplication_service
    
    async def upsert_documents(self, collection_name: str, documents: List[Document], 
                              embeddings: List[List[float]]) -> Dict[str, Any]:
        """Upsert documents with deduplication"""
        results = {
            "total_documents": len(documents),
            "unique_documents": 0,
            "duplicate_documents": 0,
            "upserted_documents": 0,
            "errors": []
        }
        
        try:
            # Deduplicate documents
            unique_docs, duplicate_docs = self.dedup_service.deduplicate_documents(documents)
            results["unique_documents"] = len(unique_docs)
            results["duplicate_documents"] = len(duplicate_docs)
            
            if not unique_docs:
                logger.info("No unique documents to upsert")
                return results
            
            # Prepare unique documents for upsert
            unique_embeddings = []
            doc_ids = []
            
            for i, doc in enumerate(unique_docs):
                # Generate document ID
                doc_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_index', i)}"
                doc_ids.append(doc_id)
                
                # Add to deduplication tracking
                self.dedup_service.add_document(doc, doc_id)
                
                # Find corresponding embedding
                original_index = documents.index(doc)
                unique_embeddings.append(embeddings[original_index])
            
            # Upsert to vector store
            success = await self.vector_store.add_documents(
                collection_name, unique_docs, unique_embeddings
            )
            
            if success:
                results["upserted_documents"] = len(unique_docs)
                logger.info(f"Successfully upserted {len(unique_docs)} unique documents")
            else:
                results["errors"].append("Failed to upsert documents to vector store")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in upsert operation: {e}")
            results["errors"].append(str(e))
            return results
    
    async def delete_by_filter(self, collection_name: str, filters: Dict[str, Any]) -> bool:
        """Delete documents by filter criteria"""
        try:
            # This would need to be implemented in the vector store
            # For now, return success
            logger.info(f"Delete by filter not implemented for {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents by filter: {e}")
            return False
    
    async def reindex_document(self, collection_name: str, document: Document, 
                              embedding: List[float]) -> bool:
        """Reindex a single document"""
        try:
            # Remove from deduplication tracking
            self.dedup_service.remove_document(document)
            
            # Add updated document
            success = await self.upsert_documents(
                collection_name, [document], [embedding]
            )
            
            return success["upserted_documents"] > 0
            
        except Exception as e:
            logger.error(f"Error reindexing document: {e}")
            return False
