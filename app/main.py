from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog
import time
from datetime import datetime

from app.routers import ingest, query, massive_ingestion, advanced_query
from app.core.config import settings
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

# Create FastAPI app
app = FastAPI(
    title="Scalable RAG System - Million Document Processing",
    description="A high-performance Retrieval-Augmented Generation system designed to handle millions of documents with enterprise-grade scalability, precision, and monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown"
    )
    
    # Process request
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

# Include routers
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(massive_ingestion.router)
app.include_router(advanced_query.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "vector_db_provider": settings.vector_db_provider,
        "embedding_model": settings.embedding_model,
        "features": [
            "million-document-processing",
            "advanced-query-precision",
            "hybrid-search",
            "query-expansion",
            "reranking",
            "prometheus-monitoring"
        ]
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Scalable RAG System - Million Document Processing",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
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
            "Cloud deployment ready"
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
    logger.info("Starting Scalable RAG System v2.0.0", 
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
    logger.info("Shutting down Scalable RAG System v2.0.0")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
