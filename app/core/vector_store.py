import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
import pinecone
from langchain.vectorstores import Chroma, Pinecone
from langchain.schema import Document
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger()

class VectorStoreManager:
    def __init__(self):
        self.provider = settings.vector_db_provider
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the appropriate vector store based on configuration"""
        if self.provider == "pinecone":
            self._init_pinecone()
        else:
            self._init_chroma()
    
    def _init_chroma(self):
        """Initialize ChromaDB"""
        try:
            # Ensure directory exists
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB initialized at {settings.chroma_persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _init_pinecone(self):
        """Initialize Pinecone"""
        try:
            if not settings.pinecone_api_key or not settings.pinecone_environment:
                raise ValueError("Pinecone API key and environment must be set")
            
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Create index if it doesn't exist
            if settings.pinecone_index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=settings.pinecone_index_name,
                    dimension=settings.embedding_dimension,
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {settings.pinecone_index_name}")
            
            self.pinecone_index = pinecone.Index(settings.pinecone_index_name)
            logger.info(f"Pinecone initialized with index: {settings.pinecone_index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def add_documents(self, collection_name: str, documents: List[Document], embeddings: List[List[float]]) -> bool:
        """Add documents to vector store"""
        try:
            if self.provider == "pinecone":
                return await self._add_to_pinecone(collection_name, documents, embeddings)
            else:
                return await self._add_to_chroma(collection_name, documents, embeddings)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    async def _add_to_chroma(self, collection_name: str, documents: List[Document], embeddings: List[List[float]]) -> bool:
        """Add documents to ChromaDB"""
        try:
            # Get or create collection
            try:
                collection = self.chroma_client.get_collection(collection_name)
            except:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            # Prepare data
            ids = [f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_index', 0)}" for doc in documents]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
            raise
    
    async def _add_to_pinecone(self, collection_name: str, documents: List[Document], embeddings: List[List[float]]) -> bool:
        """Add documents to Pinecone"""
        try:
            # Prepare vectors
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                vector_id = f"{collection_name}_{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_index', 0)}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        **doc.metadata,
                        "collection": collection_name,
                        "text": doc.page_content
                    }
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1} to Pinecone")
            
            logger.info(f"Added {len(documents)} documents to Pinecone collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to Pinecone: {e}")
            raise
    
    async def similarity_search(self, collection_name: str, query_embedding: List[float], 
                              top_k: int = 5, filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        try:
            if self.provider == "pinecone":
                return await self._search_pinecone(collection_name, query_embedding, top_k, filter)
            else:
                return await self._search_chroma(collection_name, query_embedding, top_k, filter)
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    async def _search_chroma(self, collection_name: str, query_embedding: List[float], 
                           top_k: int, filter: Optional[Dict]) -> List[Tuple[Document, float]]:
        """Search ChromaDB"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter
            )
            
            documents = []
            for i in range(len(results['documents'][0])):
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                score = 1 - results['distances'][0][i]  # Convert distance to similarity
                documents.append((doc, score))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise
    
    async def _search_pinecone(self, collection_name: str, query_embedding: List[float], 
                             top_k: int, filter: Optional[Dict]) -> List[Tuple[Document, float]]:
        """Search Pinecone"""
        try:
            # Add collection filter to metadata
            search_filter = {"collection": collection_name}
            if filter:
                search_filter.update(filter)
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=search_filter
            )
            
            documents = []
            for match in results['matches']:
                doc = Document(
                    page_content=match['metadata']['text'],
                    metadata=match['metadata']
                )
                documents.append((doc, match['score']))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            raise
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            if self.provider == "pinecone":
                stats = self.pinecone_index.describe_index_stats()
                return {
                    "name": collection_name,
                    "total_vectors": stats.get('total_vector_count', 0),
                    "dimension": stats.get('dimension', 0)
                }
            else:
                collection = self.chroma_client.get_collection(collection_name)
                count = collection.count()
                return {
                    "name": collection_name,
                    "total_vectors": count,
                    "dimension": settings.embedding_dimension
                }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"name": collection_name, "total_vectors": 0, "dimension": 0}
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            if self.provider == "pinecone":
                # Note: Pinecone doesn't support deleting individual collections easily
                logger.warning("Pinecone collection deletion not implemented")
                return False
            else:
                self.chroma_client.delete_collection(collection_name)
                logger.info(f"Deleted ChromaDB collection: {collection_name}")
                return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
