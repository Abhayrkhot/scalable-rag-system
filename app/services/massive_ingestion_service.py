import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog
from tqdm.asyncio import tqdm
import json

from app.core.document_processor import DocumentProcessor
from app.core.embedding_service import EmbeddingService
from app.core.vector_store import VectorStoreManager
from app.core.config import settings
from app.utils.dataset_generator import LargeDatasetGenerator

logger = structlog.get_logger()

class MassiveIngestionService:
    def __init__(self):
        self.document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()
        self.dataset_generator = LargeDatasetGenerator()
        
        # Performance tracking
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "errors": [],
            "processing_times": [],
            "start_time": None
        }
    
    async def process_million_documents(self, collection_name: str, 
                                      batch_size: int = 1000,
                                      max_workers: int = 10) -> Dict[str, Any]:
        """Process 1 million documents with optimized performance"""
        logger.info(f"Starting massive ingestion for collection: {collection_name}")
        self.stats["start_time"] = time.time()
        
        try:
            # Generate dataset if it doesn't exist
            dataset_path = Path("large_dataset")
            if not dataset_path.exists() or len(list(dataset_path.rglob("*.txt"))) < 1000000:
                logger.info("Generating 1 million document dataset...")
                await self.dataset_generator.generate_million_documents(1000000)
            
            # Get all document paths
            all_paths = await self._get_all_document_paths()
            logger.info(f"Found {len(all_paths):,} documents to process")
            
            # Process in batches with progress tracking
            total_batches = (len(all_paths) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(all_paths), batch_size):
                batch_paths = all_paths[batch_idx:batch_idx + batch_size]
                batch_num = batch_idx // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} documents)")
                
                # Process batch
                batch_result = await self._process_batch_optimized(
                    batch_paths, collection_name, batch_num
                )
                
                # Update stats
                self.stats["documents_processed"] += batch_result["documents_processed"]
                self.stats["chunks_created"] += batch_result["chunks_created"]
                self.stats["errors"].extend(batch_result["errors"])
                self.stats["processing_times"].append(batch_result["processing_time"])
                
                # Progress update
                progress = (batch_num / total_batches) * 100
                logger.info(f"Progress: {progress:.1f}% - Processed {self.stats['documents_processed']:,} documents, "
                           f"{self.stats['chunks_created']:,} chunks")
                
                # Memory management - clear cache every 10 batches
                if batch_num % 10 == 0:
                    await self._cleanup_memory()
            
            # Final statistics
            total_time = time.time() - self.stats["start_time"]
            avg_batch_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            
            result = {
                "success": True,
                "collection_name": collection_name,
                "total_documents_processed": self.stats["documents_processed"],
                "total_chunks_created": self.stats["chunks_created"],
                "total_processing_time_seconds": total_time,
                "average_batch_time_seconds": avg_batch_time,
                "documents_per_second": self.stats["documents_processed"] / total_time,
                "chunks_per_second": self.stats["chunks_created"] / total_time,
                "total_errors": len(self.stats["errors"]),
                "error_rate": len(self.stats["errors"]) / self.stats["documents_processed"] * 100
            }
            
            logger.info(f"Massive ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Massive ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": self.stats["documents_processed"],
                "chunks_created": self.stats["chunks_created"]
            }
    
    async def _get_all_document_paths(self) -> List[str]:
        """Get all document paths from the dataset"""
        dataset_path = Path("large_dataset")
        all_paths = []
        
        # Get all text files
        text_paths = list(dataset_path.glob("text/*.txt"))
        all_paths.extend([str(p) for p in text_paths])
        
        # Get all markdown files
        md_paths = list(dataset_path.glob("markdown/*.md"))
        all_paths.extend([str(p) for p in md_paths])
        
        # Get all PDF files
        pdf_paths = list(dataset_path.glob("pdfs/*.pdf"))
        all_paths.extend([str(p) for p in pdf_paths])
        
        return all_paths
    
    async def _process_batch_optimized(self, batch_paths: List[str], 
                                     collection_name: str, batch_num: int) -> Dict[str, Any]:
        """Process a batch of documents with optimizations"""
        start_time = time.time()
        
        try:
            # Process documents
            documents = await self.document_processor.process_files(batch_paths, len(batch_paths))
            
            if not documents:
                return {
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "processing_time": time.time() - start_time,
                    "errors": ["No documents processed in batch"]
                }
            
            # Generate embeddings in parallel
            embedded_documents = await self.embedding_service.embed_documents_async(documents)
            
            # Extract embeddings
            embeddings = [doc.metadata.get("embedding") for doc in embedded_documents]
            
            # Store in vector database
            success = await self.vector_store.add_documents(
                collection_name, embedded_documents, embeddings
            )
            
            if not success:
                raise Exception("Failed to store documents in vector database")
            
            processing_time = time.time() - start_time
            
            return {
                "documents_processed": len(batch_paths),
                "chunks_created": len(embedded_documents),
                "processing_time": processing_time,
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            return {
                "documents_processed": 0,
                "chunks_created": 0,
                "processing_time": time.time() - start_time,
                "errors": [str(e)]
            }
    
    async def _cleanup_memory(self):
        """Clean up memory to prevent OOM issues"""
        import gc
        gc.collect()
        logger.info("Memory cleanup completed")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        if self.stats["start_time"]:
            elapsed_time = time.time() - self.stats["start_time"]
            docs_per_second = self.stats["documents_processed"] / elapsed_time if elapsed_time > 0 else 0
            chunks_per_second = self.stats["chunks_created"] / elapsed_time if elapsed_time > 0 else 0
        else:
            elapsed_time = 0
            docs_per_second = 0
            chunks_per_second = 0
        
        return {
            "documents_processed": self.stats["documents_processed"],
            "chunks_created": self.stats["chunks_created"],
            "elapsed_time_seconds": elapsed_time,
            "documents_per_second": docs_per_second,
            "chunks_per_second": chunks_per_second,
            "total_errors": len(self.stats["errors"]),
            "is_processing": self.stats["start_time"] is not None
        }
