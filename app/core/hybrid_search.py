import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

logger = structlog.get_logger()

class HybridSearchEngine:
    def __init__(self):
        self.elasticsearch_client = None
        self.index_name = "rag_documents"
        self._initialize_elasticsearch()
    
    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch client"""
        try:
            es_host = getattr(settings, 'elasticsearch_host', 'localhost')
            es_port = getattr(settings, 'elasticsearch_port', 9200)
            
            self.elasticsearch_client = AsyncElasticsearch(
                hosts=[{'host': es_host, 'port': es_port}],
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            logger.info(f"Elasticsearch initialized at {es_host}:{es_port}")
        except Exception as e:
            logger.warning(f"Elasticsearch not available: {e}")
            self.elasticsearch_client = None
    
    async def create_index(self, collection_name: str) -> bool:
        """Create Elasticsearch index for a collection"""
        if not self.elasticsearch_client:
            return False
        
        try:
            index_name = f"{self.index_name}_{collection_name}"
            
            # Check if index exists
            if await self.elasticsearch_client.indices.exists(index=index_name):
                logger.info(f"Index {index_name} already exists")
                return True
            
            # Create index with mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {
                                    "type": "keyword"
                                }
                            }
                        },
                        "source": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "file_name": {"type": "keyword"},
                        "file_type": {"type": "keyword"},
                        "collection": {"type": "keyword"},
                        "metadata": {"type": "object"},
                        "created_at": {"type": "date"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "custom_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"]
                            }
                        }
                    }
                }
            }
            
            await self.elasticsearch_client.indices.create(
                index=index_name,
                body=mapping
            )
            
            logger.info(f"Created Elasticsearch index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Elasticsearch index: {e}")
            return False
    
    async def index_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> bool:
        """Index documents in Elasticsearch"""
        if not self.elasticsearch_client:
            return False
        
        try:
            index_name = f"{self.index_name}_{collection_name}"
            
            # Prepare documents for bulk indexing
            actions = []
            for doc in documents:
                action = {
                    "_index": index_name,
                    "_id": doc.get("id"),
                    "_source": {
                        "text": doc.get("text", ""),
                        "source": doc.get("source", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "file_name": doc.get("file_name", ""),
                        "file_type": doc.get("file_type", ""),
                        "collection": collection_name,
                        "metadata": doc.get("metadata", {}),
                        "created_at": doc.get("created_at")
                    }
                }
                actions.append(action)
            
            # Bulk index
            success, failed = await async_bulk(
                self.elasticsearch_client,
                actions,
                chunk_size=1000,
                request_timeout=60
            )
            
            logger.info(f"Indexed {success} documents, {len(failed)} failed")
            return len(failed) == 0
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    async def bm25_search(self, collection_name: str, query: str, top_k: int = 10, 
                         filters: Optional[Dict] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Perform BM25 keyword search"""
        if not self.elasticsearch_client:
            return []
        
        try:
            index_name = f"{self.index_name}_{collection_name}"
            
            # Build query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^2", "file_name^1.5"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "_source": ["text", "source", "chunk_index", "file_name", "file_type", "metadata"]
            }
            
            # Add filters
            if filters:
                es_query["query"]["bool"]["filter"] = []
                for key, value in filters.items():
                    es_query["query"]["bool"]["filter"].append({
                        "term": {key: value}
                    })
            
            # Execute search
            response = await self.elasticsearch_client.search(
                index=index_name,
                body=es_query
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                score = hit['_score']
                results.append((doc, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    async def hybrid_search(self, collection_name: str, query: str, 
                          vector_results: List[Tuple[Any, float]], 
                          top_k: int = 10, 
                          semantic_weight: float = 0.7,
                          keyword_weight: float = 0.3) -> List[Tuple[Any, float]]:
        """Combine semantic and keyword search results"""
        try:
            # Get BM25 results
            bm25_results = await self.bm25_search(collection_name, query, top_k * 2)
            
            if not bm25_results:
                # Fallback to vector results only
                return vector_results[:top_k]
            
            # Normalize scores
            vector_scores = [score for _, score in vector_results]
            bm25_scores = [score for _, score in bm25_results]
            
            if vector_scores:
                max_vector = max(vector_scores)
                min_vector = min(vector_scores)
                vector_range = max_vector - min_vector if max_vector != min_vector else 1
                normalized_vector = [(score - min_vector) / vector_range for score in vector_scores]
            else:
                normalized_vector = []
            
            if bm25_scores:
                max_bm25 = max(bm25_scores)
                min_bm25 = min(bm25_scores)
                bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
                normalized_bm25 = [(score - min_bm25) / bm25_range for score in bm25_scores]
            else:
                normalized_bm25 = []
            
            # Create document ID to score mapping
            doc_scores = {}
            
            # Add vector scores
            for i, (doc, score) in enumerate(vector_results):
                doc_id = self._get_doc_id(doc)
                if i < len(normalized_vector):
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'vector_score': normalized_vector[i],
                        'bm25_score': 0.0
                    }
            
            # Add BM25 scores
            for i, (doc, score) in enumerate(bm25_results):
                doc_id = self._get_doc_id(doc)
                if doc_id in doc_scores:
                    doc_scores[doc_id]['bm25_score'] = normalized_bm25[i] if i < len(normalized_bm25) else 0.0
                else:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'vector_score': 0.0,
                        'bm25_score': normalized_bm25[i] if i < len(normalized_bm25) else 0.0
                    }
            
            # Calculate hybrid scores
            hybrid_results = []
            for doc_id, scores in doc_scores.items():
                hybrid_score = (semantic_weight * scores['vector_score'] + 
                              keyword_weight * scores['bm25_score'])
                hybrid_results.append((scores['doc'], hybrid_score))
            
            # Sort by hybrid score
            hybrid_results.sort(key=lambda x: x[1], reverse=True)
            
            return hybrid_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return vector_results[:top_k]
    
    def _get_doc_id(self, doc) -> str:
        """Generate document ID for deduplication"""
        if hasattr(doc, 'metadata'):
            source = doc.metadata.get('source', '')
            chunk_index = doc.metadata.get('chunk_index', 0)
            return f"{source}_{chunk_index}"
        elif isinstance(doc, dict):
            source = doc.get('source', '')
            chunk_index = doc.get('chunk_index', 0)
            return f"{source}_{chunk_index}"
        else:
            return str(hash(str(doc)))
    
    async def delete_index(self, collection_name: str) -> bool:
        """Delete Elasticsearch index"""
        if not self.elasticsearch_client:
            return False
        
        try:
            index_name = f"{self.index_name}_{collection_name}"
            await self.elasticsearch_client.indices.delete(index=index_name)
            logger.info(f"Deleted Elasticsearch index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    async def get_index_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.elasticsearch_client:
            return {"error": "Elasticsearch not available"}
        
        try:
            index_name = f"{self.index_name}_{collection_name}"
            stats = await self.elasticsearch_client.indices.stats(index=index_name)
            
            if index_name in stats['indices']:
                index_stats = stats['indices'][index_name]
                return {
                    "doc_count": index_stats['total']['docs']['count'],
                    "size_in_bytes": index_stats['total']['store']['size_in_bytes'],
                    "index_name": index_name
                }
            else:
                return {"error": "Index not found"}
                
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
