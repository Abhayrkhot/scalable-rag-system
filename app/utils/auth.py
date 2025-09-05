from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from app.core.config import settings

logger = structlog.get_logger()
security = HTTPBearer()

async def verify_api_key(request: Request):
    """Verify API key from request headers"""
    api_key = request.headers.get("x-api-key")
    
    if not api_key:
        logger.warning("Missing API key in request")
        raise HTTPException(status_code=401, detail="Missing API key")
    
    if api_key != settings.api_key:
        logger.warning(f"Invalid API key: {api_key[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True
