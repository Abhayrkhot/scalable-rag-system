from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps
import structlog

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'])
INGESTION_DOCUMENTS = Counter('rag_documents_ingested_total', 'Total documents ingested', ['collection'])
INGESTION_CHUNKS = Counter('rag_chunks_created_total', 'Total chunks created', ['collection'])
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Query duration in seconds', ['collection'])
EMBEDDING_DURATION = Histogram('rag_embedding_duration_seconds', 'Embedding generation duration in seconds')
VECTOR_STORE_OPERATIONS = Counter('rag_vector_store_operations_total', 'Vector store operations', ['operation', 'status'])
ACTIVE_CONNECTIONS = Gauge('rag_active_connections', 'Number of active connections')
COLLECTION_SIZE = Gauge('rag_collection_size', 'Collection size in documents', ['collection'])

def monitor_requests(func):
    """Decorator to monitor API requests"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = "unknown"
        endpoint = "unknown"
        status = "success"
        
        try:
            # Extract method and endpoint from FastAPI request if available
            if args and hasattr(args[0], 'method'):
                method = args[0].method
            if args and hasattr(args[0], 'url'):
                endpoint = str(args[0].url.path)
            
            result = await func(*args, **kwargs)
            return result
            
        except Exception as e:
            status = "error"
            logger.error(f"Request failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    return wrapper

def monitor_ingestion(collection_name: str):
    """Monitor ingestion metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                
                # Update metrics
                if isinstance(result, dict):
                    documents_processed = result.get('documents_processed', 0)
                    chunks_created = result.get('chunks_created', 0)
                    
                    INGESTION_DOCUMENTS.labels(collection=collection_name).inc(documents_processed)
                    INGESTION_CHUNKS.labels(collection=collection_name).inc(chunks_created)
                    COLLECTION_SIZE.labels(collection=collection_name).set(chunks_created)
                
                return result
            finally:
                duration = time.time() - start_time
                logger.info(f"Ingestion completed in {duration:.2f}s for collection {collection_name}")
        return wrapper
    return decorator

def monitor_query(collection_name: str):
    """Monitor query metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                QUERY_DURATION.labels(collection=collection_name).observe(duration)
                logger.info(f"Query completed in {duration:.2f}s for collection {collection_name}")
        return wrapper
    return decorator

def monitor_embeddings(func):
    """Monitor embedding generation"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            EMBEDDING_DURATION.observe(duration)
            logger.info(f"Embedding generation completed in {duration:.2f}s")
    return wrapper

def start_monitoring(port: int = 8001):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

def update_collection_metrics(collection_name: str, document_count: int, chunk_count: int):
    """Update collection-specific metrics"""
    COLLECTION_SIZE.labels(collection=collection_name).set(chunk_count)
    logger.info(f"Updated metrics for collection {collection_name}: {document_count} docs, {chunk_count} chunks")
