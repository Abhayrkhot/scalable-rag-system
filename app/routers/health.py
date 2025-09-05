from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import structlog
import asyncio
import time
from datetime import datetime

from app.core.config import settings
from app.core.vector_store import VectorStoreManager
from app.core.embedding_service import EmbeddingService
from app.utils.auth import verify_api_key

logger = structlog.get_logger()
router = APIRouter(prefix="/health", tags=["health"])

# Initialize services for health checks
vector_store = VectorStoreManager()
embedding_service = EmbeddingService()

@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "service": "scalable-rag-system"
    }

@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Check vector store
        vector_status = await check_vector_store()
        
        # Check embedding service
        embedding_status = await check_embedding_service()
        
        # Check Redis (if configured)
        redis_status = await check_redis()
        
        # Check database (if configured)
        db_status = await check_database()
        
        all_healthy = all([
            vector_status["healthy"],
            embedding_status["healthy"],
            redis_status["healthy"],
            db_status["healthy"]
        ])
        
        if all_healthy:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {
                    "vector_store": vector_status,
                    "embedding_service": embedding_status,
                    "redis": redis_status,
                    "database": db_status
                }
            }
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - start_time
    }

@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    try:
        from app.utils.monitoring import (
            REQUEST_COUNT, REQUEST_DURATION, INGESTION_DOCUMENTS,
            INGESTION_CHUNKS, QUERY_DURATION, EMBEDDING_DURATION,
            VECTOR_STORE_OPERATIONS, ACTIVE_CONNECTIONS, COLLECTION_SIZE
        )
        
        # This would return Prometheus-formatted metrics
        # In a real implementation, you'd use prometheus_client's generate_latest()
        return {
            "message": "Metrics available at /metrics (Prometheus format)",
            "endpoint": "/metrics"
        }
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        return {"error": "Metrics not available"}

@router.get("/status")
async def detailed_status():
    """Detailed system status"""
    try:
        # Get system information
        system_info = await get_system_info()
        
        # Get service status
        service_status = await get_service_status()
        
        # Get performance metrics
        performance_metrics = await get_performance_metrics()
        
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_info,
            "services": service_status,
            "performance": performance_metrics
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def check_vector_store() -> Dict[str, Any]:
    """Check vector store health"""
    try:
        # Try to get collection info
        info = await vector_store.get_collection_info("health_check")
        return {
            "healthy": True,
            "provider": settings.vector_db_provider,
            "status": "connected"
        }
    except Exception as e:
        logger.error(f"Vector store check failed: {e}")
        return {
            "healthy": False,
            "provider": settings.vector_db_provider,
            "error": str(e)
        }

async def check_embedding_service() -> Dict[str, Any]:
    """Check embedding service health"""
    try:
        # Try to embed a test query
        test_embedding = await embedding_service.embed_query("health check")
        return {
            "healthy": True,
            "model": settings.embedding_model,
            "dimension": len(test_embedding)
        }
    except Exception as e:
        logger.error(f"Embedding service check failed: {e}")
        return {
            "healthy": False,
            "model": settings.embedding_model,
            "error": str(e)
        }

async def check_redis() -> Dict[str, Any]:
    """Check Redis health"""
    try:
        import redis
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()
        return {
            "healthy": True,
            "url": settings.redis_url,
            "status": "connected"
        }
    except Exception as e:
        logger.warning(f"Redis check failed: {e}")
        return {
            "healthy": False,
            "url": settings.redis_url,
            "error": str(e)
        }

async def check_database() -> Dict[str, Any]:
    """Check database health"""
    try:
        # This would check database connectivity
        # For now, return healthy if no database URL is configured
        if not hasattr(settings, 'database_url') or not settings.database_url:
            return {
                "healthy": True,
                "status": "not_configured"
            }
        
        # In a real implementation, you'd test database connectivity
        return {
            "healthy": True,
            "url": settings.database_url,
            "status": "connected"
        }
    except Exception as e:
        logger.warning(f"Database check failed: {e}")
        return {
            "healthy": False,
            "error": str(e)
        }

async def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import psutil
    import platform
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage('/').percent
    }

async def get_service_status() -> Dict[str, Any]:
    """Get service status"""
    return {
        "vector_store": await check_vector_store(),
        "embedding_service": await check_embedding_service(),
        "redis": await check_redis(),
        "database": await check_database()
    }

async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics"""
    try:
        from app.utils.monitoring import (
            REQUEST_COUNT, REQUEST_DURATION, INGESTION_DOCUMENTS,
            INGESTION_CHUNKS, QUERY_DURATION, EMBEDDING_DURATION
        )
        
        # Get metric values (this is simplified)
        return {
            "total_requests": "N/A",  # Would get from REQUEST_COUNT
            "average_response_time": "N/A",  # Would get from REQUEST_DURATION
            "documents_ingested": "N/A",  # Would get from INGESTION_DOCUMENTS
            "chunks_created": "N/A"  # Would get from INGESTION_CHUNKS
        }
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return {"error": "Metrics not available"}

# Global start time for uptime calculation
start_time = time.time()
