import time
import logging
from typing import Dict, Any, Optional
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
from functools import wraps

logger = structlog.get_logger()

# Create a custom registry
registry = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

REQUEST_DURATION = Histogram(
    'rag_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

# Ingestion metrics
INGESTION_DOCUMENTS = Counter(
    'rag_documents_ingested_total',
    'Total number of documents ingested',
    ['collection', 'file_type'],
    registry=registry
)

INGESTION_CHUNKS = Counter(
    'rag_chunks_created_total',
    'Total number of chunks created',
    ['collection'],
    registry=registry
)

INGESTION_DURATION = Histogram(
    'rag_ingestion_duration_seconds',
    'Ingestion duration in seconds',
    ['collection'],
    registry=registry
)

# Query metrics
QUERY_DURATION = Histogram(
    'rag_query_duration_seconds',
    'Query duration in seconds',
    ['collection', 'search_type'],
    registry=registry
)

QUERY_RESULTS = Histogram(
    'rag_query_results_count',
    'Number of results returned per query',
    ['collection'],
    registry=registry
)

# Embedding metrics
EMBEDDING_DURATION = Histogram(
    'rag_embedding_duration_seconds',
    'Embedding generation duration in seconds',
    ['model'],
    registry=registry
)

EMBEDDING_TOKENS = Counter(
    'rag_embedding_tokens_total',
    'Total tokens processed for embedding',
    ['model'],
    registry=registry
)

# Vector store metrics
VECTOR_STORE_OPERATIONS = Counter(
    'rag_vector_store_operations_total',
    'Total vector store operations',
    ['operation', 'collection', 'status'],
    registry=registry
)

VECTOR_STORE_DURATION = Histogram(
    'rag_vector_store_duration_seconds',
    'Vector store operation duration in seconds',
    ['operation', 'collection'],
    registry=registry
)

# Cache metrics
CACHE_HITS = Counter(
    'rag_cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

CACHE_MISSES = Counter(
    'rag_cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'rag_active_connections',
    'Number of active connections',
    registry=registry
)

COLLECTION_SIZE = Gauge(
    'rag_collection_size',
    'Number of documents in collection',
    ['collection'],
    registry=registry
)

# Error metrics
ERROR_COUNT = Counter(
    'rag_errors_total',
    'Total number of errors',
    ['error_type', 'component'],
    registry=registry
)

# Rate limiting metrics
RATE_LIMIT_HITS = Counter(
    'rag_rate_limit_hits_total',
    'Total rate limit hits',
    ['api_key', 'limit_type'],
    registry=registry
)

# Cost metrics
OPENAI_COST = Counter(
    'rag_openai_cost_usd_total',
    'Total OpenAI cost in USD',
    ['model', 'operation'],
    registry=registry
)

# Performance metrics
STAGE_DURATION = Histogram(
    'rag_stage_duration_seconds',
    'Duration of processing stages',
    ['stage', 'collection'],
    registry=registry
)

def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port, registry=registry)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        raise

def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Record request metrics"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

def record_ingestion_metrics(collection: str, file_type: str, doc_count: int, 
                           chunk_count: int, duration: float):
    """Record ingestion metrics"""
    INGESTION_DOCUMENTS.labels(collection=collection, file_type=file_type).inc(doc_count)
    INGESTION_CHUNKS.labels(collection=collection).inc(chunk_count)
    INGESTION_DURATION.labels(collection=collection).observe(duration)

def record_query_metrics(collection: str, search_type: str, result_count: int, duration: float):
    """Record query metrics"""
    QUERY_DURATION.labels(collection=collection, search_type=search_type).observe(duration)
    QUERY_RESULTS.labels(collection=collection).observe(result_count)

def record_embedding_metrics(model: str, token_count: int, duration: float):
    """Record embedding metrics"""
    EMBEDDING_DURATION.labels(model=model).observe(duration)
    EMBEDDING_TOKENS.labels(model=model).inc(token_count)

def record_vector_store_metrics(operation: str, collection: str, status: str, duration: float):
    """Record vector store metrics"""
    VECTOR_STORE_OPERATIONS.labels(operation=operation, collection=collection, status=status).inc()
    VECTOR_STORE_DURATION.labels(operation=operation, collection=collection).observe(duration)

def record_cache_metrics(cache_type: str, hit: bool):
    """Record cache metrics"""
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()

def record_error_metrics(error_type: str, component: str):
    """Record error metrics"""
    ERROR_COUNT.labels(error_type=error_type, component=component).inc()

def record_rate_limit_metrics(api_key: str, limit_type: str):
    """Record rate limit metrics"""
    RATE_LIMIT_HITS.labels(api_key=api_key[:8], limit_type=limit_type).inc()

def record_cost_metrics(model: str, operation: str, cost_usd: float):
    """Record cost metrics"""
    OPENAI_COST.labels(model=model, operation=operation).inc(cost_usd)

def record_stage_duration(stage: str, collection: str, duration: float):
    """Record stage duration"""
    STAGE_DURATION.labels(stage=stage, collection=collection).observe(duration)

def update_active_connections(count: int):
    """Update active connections gauge"""
    ACTIVE_CONNECTIONS.set(count)

def update_collection_size(collection: str, size: int):
    """Update collection size gauge"""
    COLLECTION_SIZE.labels(collection=collection).set(size)

def timing_decorator(metric_func, *labels):
    """Decorator for timing function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metric_func(*labels, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metric_func(*labels, duration)
                raise
        return async_wrapper
    return decorator

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.stage_times = {}
    
    def start_stage(self, stage: str, collection: str = "default"):
        """Start timing a stage"""
        key = f"{stage}:{collection}"
        self.stage_times[key] = time.time()
    
    def end_stage(self, stage: str, collection: str = "default"):
        """End timing a stage and record metric"""
        key = f"{stage}:{collection}"
        if key in self.stage_times:
            duration = time.time() - self.stage_times[key]
            record_stage_duration(stage, collection, duration)
            del self.stage_times[key]
            return duration
        return 0.0
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    def get_stage_metrics(self) -> Dict[str, Any]:
        """Get current stage metrics"""
        return {
            "uptime_seconds": self.get_uptime(),
            "active_stages": list(self.stage_times.keys()),
            "stage_count": len(self.stage_times)
        }

# Global metrics collector
metrics_collector = MetricsCollector()

def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of all metrics"""
    try:
        # This would collect all metrics from the registry
        # For now, return a placeholder
        return {
            "request_count": "N/A",
            "request_duration_avg": "N/A",
            "ingestion_documents": "N/A",
            "query_duration_avg": "N/A",
            "cache_hit_rate": "N/A",
            "error_rate": "N/A",
            "uptime_seconds": metrics_collector.get_uptime()
        }
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        return {"error": str(e)}
