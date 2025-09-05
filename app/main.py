from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
import time
from datetime import datetime

from app.routers import ingest, query, massive_ingestion, advanced_query, health
from app.core.config import settings
from app.core.security import SecurityService, RequestValidator, AuditLogger
from app.core.rate_limiting import RateLimiter, BackpressureController, APIKeyQuota
from app.core.cache import CacheService, CacheMetrics
from app.utils.monitoring import start_monitoring

logger = structlog.get_logger()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="Scalable RAG System - Enterprise Edition",
    description="A high-performance Retrieval-Augmented Generation system with enterprise-grade security, monitoring, and evaluation capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Initialize services
security_service = SecurityService()
request_validator = RequestValidator()
audit_logger = AuditLogger()
rate_limiter = RateLimiter()
backpressure_controller = BackpressureController(rate_limiter)
cache_service = CacheService()
cache_metrics = CacheMetrics(cache_service)

# Initialize default API key quotas
default_quota = APIKeyQuota(
    api_key="default",
    requests_per_minute=100,
    requests_per_hour=1000,
    concurrent_requests=10,
    burst_limit=20,
    scopes=["query", "ingest"],
    is_active=True
)
rate_limiter.add_api_key_quota(default_quota)

# Request logging and security middleware
@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Get API key from request
    api_key = request.headers.get("x-api-key", "default")
    
    # Log request for audit
    audit_logger.log_request(request, api_key)
    
    # Check backpressure
    backpressure_check = await backpressure_controller.should_accept_request(api_key)
    if not backpressure_check["allowed"]:
        logger.warning(f"Request rejected due to backpressure: {backpressure_check['reason']}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": backpressure_check["reason"],
                "retry_after": backpressure_check.get("retry_after", 60)
            },
            headers={"Retry-After": str(backpressure_check.get("retry_after", 60))}
        )
    
    # Record request
    await rate_limiter.record_request(api_key)
    
    # Validate request size
    if not request_validator.validate_request_size(request):
        logger.warning("Request size validation failed", url=str(request.url))
        return JSONResponse(
            status_code=413,
            content={"error": "Request too large"}
        )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=round(process_time, 3),
            api_key=api_key[:8] + "..." if api_key != "default" else "default"
        )
        
        return response
        
    except Exception as e:
        # Log security events
        audit_logger.log_security_event(
            "request_error",
            {"error": str(e), "url": str(request.url)},
            api_key
        )
        
        logger.error(
            "Request processing error",
            exception=str(e),
            url=str(request.url),
            method=request.method,
            api_key=api_key[:8] + "..." if api_key != "default" else "default"
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    finally:
        # Release request
        await rate_limiter.release_request(api_key)

# Include routers
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(massive_ingestion.router)
app.include_router(advanced_query.router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Scalable RAG System - Enterprise Edition",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "/health/",
            "ready": "/health/ready",
            "live": "/health/live",
            "metrics": "/health/metrics",
            "status": "/health/status",
            "basic_ingestion": "/ingest/",
            "basic_query": "/query/",
            "streaming_query": "/query/stream",
            "batch_query": "/query/batch",
            "massive_ingestion": "/massive/",
            "advanced_query": "/advanced-query/"
        },
        "capabilities": [
            "Process 1+ million documents",
            "Hybrid search (semantic + keyword)",
            "Query expansion and reranking",
            "Real-time streaming responses",
            "Advanced precision querying",
            "Real-time monitoring",
            "Enterprise security",
            "RAGAS evaluation",
            "Deduplication and upserts",
            "Rate limiting and backpressure"
        ],
        "security_features": [
            "File type validation",
            "Content scanning",
            "Rate limiting",
            "Request size limits",
            "Audit logging",
            "Input sanitization",
            "API key quotas",
            "Backpressure control"
        ],
        "new_features": [
            "Query planning and optimization",
            "Section-aware chunking",
            "SSE streaming responses",
            "Redis caching",
            "Advanced rate limiting",
            "Comprehensive monitoring"
        ]
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        exception=str(exc),
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Scalable RAG System Enterprise Edition", 
                version="2.0.0",
                vector_db=settings.vector_db_provider,
                embedding_model=settings.embedding_model)
    
    # Start monitoring
    try:
        start_monitoring(settings.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {settings.prometheus_port}")
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")
    
    # Initialize cache
    try:
        cache_stats = await cache_service.get_cache_stats()
        logger.info(f"Cache initialized: {cache_stats}")
    except Exception as e:
        logger.warning(f"Cache initialization failed: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Scalable RAG System Enterprise Edition")
    
    # Cleanup security service
    security_service.cleanup_temp_directory()
    
    # Close cache connections
    if cache_service.redis_client:
        await cache_service.redis_client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
