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
from app.utils.monitoring import start_monitoring

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

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

# Initialize security services
security_service = SecurityService()
request_validator = RequestValidator()
audit_logger = AuditLogger()

# Request logging and security middleware
@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Log request for audit
    audit_logger.log_request(request)
    
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
            process_time=round(process_time, 3)
        )
        
        return response
        
    except Exception as e:
        # Log security events
        audit_logger.log_security_event(
            "request_error",
            {"error": str(e), "url": str(request.url)},
            user_id=None
        )
        
        logger.error(
            "Request processing error",
            exception=str(e),
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
            "massive_ingestion": "/massive/",
            "advanced_query": "/advanced-query/"
        },
        "capabilities": [
            "Process 1+ million documents",
            "Advanced precision querying",
            "Hybrid search (semantic + keyword)",
            "Query expansion and reranking",
            "Real-time monitoring",
            "Enterprise security",
            "RAGAS evaluation",
            "Deduplication and upserts",
            "Rate limiting and guardrails"
        ],
        "security_features": [
            "File type validation",
            "Content scanning",
            "Rate limiting",
            "Request size limits",
            "Audit logging",
            "Input sanitization"
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

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Scalable RAG System Enterprise Edition")
    
    # Cleanup security service
    security_service.cleanup_temp_directory()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
