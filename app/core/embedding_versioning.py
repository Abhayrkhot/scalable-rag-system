import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import structlog
import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.embedding_service import EmbeddingService
from app.core.vector_store import VectorStoreManager

logger = structlog.get_logger()

class EmbeddingVersionManager:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()
        self.version_metadata = {}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about an embedding model"""
        try:
            # Get model dimensions
            if model_name.startswith("text-embedding"):
                # OpenAI models
                dimension_map = {
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072,
                    "text-embedding-ada-002": 1536
                }
                dimension = dimension_map.get(model_name, 1536)
            else:
                # Sentence transformers models
                dimension = 384  # Default for most sentence transformers
            
            # Generate model hash
            model_hash = self._compute_model_hash(model_name, dimension)
            
            return {
                "model_name": model_name,
                "dimension": dimension,
                "model_hash": model_hash,
                "created_at": datetime.utcnow().isoformat(),
                "is_openai": model_name.startswith("text-embedding"),
                "is_sentence_transformers": not model_name.startswith("text-embedding")
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "model_name": model_name,
                "dimension": 0,
                "model_hash": "",
                "error": str(e)
            }
    
    def _compute_model_hash(self, model_name: str, dimension: int) -> str:
        """Compute hash for model identification"""
        model_info = f"{model_name}:{dimension}:{settings.embedding_batch_size}"
        return hashlib.sha256(model_info.encode()).hexdigest()[:16]
    
    async def validate_collection_compatibility(self, collection_name: str, 
                                              new_model: str) -> Dict[str, Any]:
        """Validate if new model is compatible with existing collection"""
        try:
            # Get collection info
            collection_info = await self.vector_store.get_collection_info(collection_name)
            
            if not collection_info or collection_info.get("total_vectors", 0) == 0:
                return {
                    "compatible": True,
                    "reason": "Empty collection",
                    "action": "safe_to_proceed"
                }
            
            # Get current model info
            current_model_info = self.get_model_info(settings.embedding_model)
            new_model_info = self.get_model_info(new_model)
            
            # Check dimension compatibility
            if current_model_info["dimension"] != new_model_info["dimension"]:
                return {
                    "compatible": False,
                    "reason": f"Dimension mismatch: {current_model_info['dimension']} vs {new_model_info['dimension']}",
                    "action": "requires_migration",
                    "current_model": current_model_info,
                    "new_model": new_model_info
                }
            
            # Check model type compatibility
            if current_model_info["is_openai"] != new_model_info["is_openai"]:
                return {
                    "compatible": False,
                    "reason": "Model type mismatch (OpenAI vs Sentence Transformers)",
                    "action": "requires_migration",
                    "current_model": current_model_info,
                    "new_model": new_model_info
                }
            
            return {
                "compatible": True,
                "reason": "Models are compatible",
                "action": "safe_to_proceed",
                "current_model": current_model_info,
                "new_model": new_model_info
            }
            
        except Exception as e:
            logger.error(f"Error validating collection compatibility: {e}")
            return {
                "compatible": False,
                "reason": f"Validation error: {str(e)}",
                "action": "error"
            }
    
    async def migrate_collection(self, collection_name: str, new_model: str, 
                               batch_size: int = 100) -> Dict[str, Any]:
        """Migrate collection to new embedding model"""
        try:
            logger.info(f"Starting migration of collection {collection_name} to model {new_model}")
            
            # Validate compatibility
            validation = await self.validate_collection_compatibility(collection_name, new_model)
            if not validation["compatible"]:
                return {
                    "success": False,
                    "error": validation["reason"],
                    "action": validation["action"]
                }
            
            # Get all documents from collection
            documents = await self._get_all_documents(collection_name)
            
            if not documents:
                return {
                    "success": True,
                    "message": "No documents to migrate",
                    "migrated_documents": 0
                }
            
            # Create new collection with new model
            new_collection_name = f"{collection_name}_migrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Update embedding service to use new model
            old_model = settings.embedding_model
            settings.embedding_model = new_model
            
            try:
                # Re-embed all documents
                new_embeddings = []
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    batch_embeddings = await self.embedding_service.embed_documents_batch(batch)
                    new_embeddings.extend(batch_embeddings)
                    
                    logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                # Store in new collection
                success = await self.vector_store.add_documents(
                    new_collection_name, documents, new_embeddings
                )
                
                if success:
                    # Update collection metadata
                    await self._update_collection_metadata(new_collection_name, {
                        "original_collection": collection_name,
                        "migrated_from_model": old_model,
                        "migrated_to_model": new_model,
                        "migration_date": datetime.utcnow().isoformat(),
                        "document_count": len(documents)
                    })
                    
                    logger.info(f"Successfully migrated collection to {new_collection_name}")
                    
                    return {
                        "success": True,
                        "new_collection_name": new_collection_name,
                        "migrated_documents": len(documents),
                        "old_model": old_model,
                        "new_model": new_model
                    }
                else:
                    return {
                        "success": False,
                        "error": "Failed to store migrated documents"
                    }
                    
            finally:
                # Restore original model
                settings.embedding_model = old_model
            
        except Exception as e:
            logger.error(f"Collection migration failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_all_documents(self, collection_name: str) -> List[Any]:
        """Get all documents from a collection"""
        try:
            # This would implement getting all documents from the vector store
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting documents from collection: {e}")
            return []
    
    async def _update_collection_metadata(self, collection_name: str, metadata: Dict[str, Any]):
        """Update collection metadata"""
        try:
            # This would update collection metadata
            logger.info(f"Updated collection metadata for {collection_name}: {metadata}")
            
        except Exception as e:
            logger.error(f"Failed to update collection metadata: {e}")
    
    async def get_collection_model_info(self, collection_name: str) -> Dict[str, Any]:
        """Get model information for a collection"""
        try:
            collection_info = await self.vector_store.get_collection_info(collection_name)
            
            if not collection_info:
                return {
                    "error": "Collection not found",
                    "collection_name": collection_name
                }
            
            # Get current model info
            current_model = settings.embedding_model
            model_info = self.get_model_info(current_model)
            
            return {
                "collection_name": collection_name,
                "current_model": model_info,
                "dimension": collection_info.get("dimension", 0),
                "total_vectors": collection_info.get("total_vectors", 0),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting collection model info: {e}")
            return {
                "error": str(e),
                "collection_name": collection_name
            }
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available embedding models"""
        try:
            models = [
                {
                    "name": "text-embedding-3-small",
                    "dimension": 1536,
                    "provider": "openai",
                    "description": "OpenAI's smallest embedding model"
                },
                {
                    "name": "text-embedding-3-large",
                    "dimension": 3072,
                    "provider": "openai",
                    "description": "OpenAI's largest embedding model"
                },
                {
                    "name": "text-embedding-ada-002",
                    "dimension": 1536,
                    "provider": "openai",
                    "description": "OpenAI's legacy embedding model"
                },
                {
                    "name": "all-MiniLM-L6-v2",
                    "dimension": 384,
                    "provider": "sentence_transformers",
                    "description": "Popular sentence transformer model"
                },
                {
                    "name": "all-mpnet-base-v2",
                    "dimension": 768,
                    "provider": "sentence_transformers",
                    "description": "High-quality sentence transformer model"
                }
            ]
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing available models: {e}")
            return []
    
    async def create_migration_plan(self, collection_name: str, new_model: str) -> Dict[str, Any]:
        """Create a migration plan for a collection"""
        try:
            # Validate compatibility
            validation = await self.validate_collection_compatibility(collection_name, new_model)
            
            # Get collection info
            collection_info = await self.vector_store.get_collection_info(collection_name)
            
            # Estimate migration time and resources
            document_count = collection_info.get("total_vectors", 0)
            estimated_time = self._estimate_migration_time(document_count, new_model)
            
            return {
                "collection_name": collection_name,
                "new_model": new_model,
                "validation": validation,
                "document_count": document_count,
                "estimated_time_minutes": estimated_time,
                "estimated_cost": self._estimate_migration_cost(document_count, new_model),
                "recommendations": self._get_migration_recommendations(validation, document_count)
            }
            
        except Exception as e:
            logger.error(f"Error creating migration plan: {e}")
            return {
                "error": str(e),
                "collection_name": collection_name
            }
    
    def _estimate_migration_time(self, document_count: int, new_model: str) -> int:
        """Estimate migration time in minutes"""
        # Rough estimates based on model type
        if new_model.startswith("text-embedding"):
            # OpenAI models are faster
            return max(1, document_count // 1000)  # 1000 docs per minute
        else:
            # Sentence transformers are slower
            return max(1, document_count // 500)  # 500 docs per minute
    
    def _estimate_migration_cost(self, document_count: int, new_model: str) -> float:
        """Estimate migration cost in USD"""
        if new_model.startswith("text-embedding"):
            # OpenAI pricing (rough estimates)
            if "3-large" in new_model:
                return document_count * 0.00013  # $0.13 per 1K tokens
            else:
                return document_count * 0.00002  # $0.02 per 1K tokens
        else:
            # Sentence transformers are free
            return 0.0
    
    def _get_migration_recommendations(self, validation: Dict[str, Any], 
                                     document_count: int) -> List[str]:
        """Get migration recommendations"""
        recommendations = []
        
        if not validation["compatible"]:
            recommendations.append("Backup the original collection before migration")
            recommendations.append("Test the new model on a small subset first")
            recommendations.append("Plan for downtime during migration")
        
        if document_count > 10000:
            recommendations.append("Consider migrating during off-peak hours")
            recommendations.append("Monitor system resources during migration")
        
        if validation.get("action") == "requires_migration":
            recommendations.append("Update all queries to use the new collection name")
            recommendations.append("Verify embedding quality after migration")
        
        return recommendations
