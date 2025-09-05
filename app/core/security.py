import logging
from typing import List, Dict, Any, Optional, Tuple
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
import os
import tempfile
import shutil
from pathlib import Path
import magic
import hashlib

logger = structlog.get_logger()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

class SecurityService:
    def __init__(self):
        self.allowed_file_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.markdown': 'text/markdown'
        }
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.temp_dir = None
        self._setup_temp_directory()
    
    def _setup_temp_directory(self):
        """Setup secure temporary directory"""
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
            logger.info(f"Created secure temp directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to create temp directory: {e}")
            self.temp_dir = None
    
    def cleanup_temp_directory(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temp directory")
            except Exception as e:
                logger.error(f"Failed to cleanup temp directory: {e}")
    
    def validate_file_type(self, file_path: str) -> bool:
        """Validate file type using magic numbers"""
        try:
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.allowed_file_types:
                logger.warning(f"Disallowed file extension: {file_ext}")
                return False
            
            # Check MIME type
            mime_type = magic.from_file(file_path, mime=True)
            expected_mime = self.allowed_file_types[file_ext]
            
            if mime_type != expected_mime:
                logger.warning(f"MIME type mismatch: {mime_type} != {expected_mime}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"File type validation failed: {e}")
            return False
    
    def validate_file_size(self, file_path: str) -> bool:
        """Validate file size"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.warning(f"File too large: {file_size} bytes")
                return False
            return True
        except Exception as e:
            logger.error(f"File size validation failed: {e}")
            return False
    
    def scan_for_malicious_content(self, file_path: str) -> bool:
        """Basic malicious content scanning"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024)  # Read first 1KB
                
                # Check for suspicious patterns
                suspicious_patterns = [
                    b'<script',
                    b'javascript:',
                    b'vbscript:',
                    b'<iframe',
                    b'<object',
                    b'<embed'
                ]
                
                for pattern in suspicious_patterns:
                    if pattern in content.lower():
                        logger.warning(f"Suspicious content detected: {pattern}")
                        return False
                
                return True
                
        except Exception as e:
            logger.error(f"Content scanning failed: {e}")
            return False
    
    def validate_upload(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive upload validation"""
        result = {
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result["errors"].append("File does not exist")
                return result
            
            # Validate file size
            if not self.validate_file_size(file_path):
                result["errors"].append(f"File too large (max: {self.max_file_size} bytes)")
            
            # Validate file type
            if not self.validate_file_type(file_path):
                result["errors"].append("Invalid file type")
            
            # Scan for malicious content
            if not self.scan_for_malicious_content(file_path):
                result["errors"].append("Suspicious content detected")
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                result["errors"].append("File not readable")
            
            result["valid"] = len(result["errors"]) == 0
            
            return result
            
        except Exception as e:
            logger.error(f"Upload validation failed: {e}")
            result["errors"].append(f"Validation error: {str(e)}")
            return result
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security"""
        import re
        
        # Remove path traversal attempts
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        filename = filename[:100]
        
        return filename
    
    def generate_secure_temp_path(self, original_filename: str) -> str:
        """Generate secure temporary file path"""
        if not self.temp_dir:
            raise Exception("Temp directory not initialized")
        
        sanitized_name = self.sanitize_filename(original_filename)
        temp_path = os.path.join(self.temp_dir, sanitized_name)
        
        return temp_path

class RequestValidator:
    def __init__(self):
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_query_length = 1000
        self.max_collection_name_length = 100
    
    def validate_request_size(self, request: Request) -> bool:
        """Validate request size"""
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    logger.warning(f"Request too large: {size} bytes")
                    return False
            except ValueError:
                logger.warning("Invalid content-length header")
                return False
        return True
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate query parameters"""
        result = {"valid": True, "errors": []}
        
        if len(query) > self.max_query_length:
            result["valid"] = False
            result["errors"].append(f"Query too long (max: {self.max_query_length} chars)")
        
        if not query.strip():
            result["valid"] = False
            result["errors"].append("Query cannot be empty")
        
        # Check for SQL injection patterns
        sql_patterns = ['union', 'select', 'insert', 'update', 'delete', 'drop', 'create']
        if any(pattern in query.lower() for pattern in sql_patterns):
            result["warnings"] = result.get("warnings", [])
            result["warnings"].append("Query contains potentially suspicious patterns")
        
        return result
    
    def validate_collection_name(self, name: str) -> Dict[str, Any]:
        """Validate collection name"""
        result = {"valid": True, "errors": []}
        
        if len(name) > self.max_collection_name_length:
            result["valid"] = False
            result["errors"].append(f"Collection name too long (max: {self.max_collection_name_length} chars)")
        
        if not name.replace('_', '').replace('-', '').isalnum():
            result["valid"] = False
            result["errors"].append("Collection name must be alphanumeric with underscores/hyphens only")
        
        return result

class AuditLogger:
    def __init__(self):
        self.audit_logger = structlog.get_logger("audit")
    
    def log_request(self, request: Request, user_id: str = None):
        """Log request for audit purposes"""
        self.audit_logger.info(
            "API Request",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            user_id=user_id,
            timestamp=structlog.processors.TimeStamper(fmt="iso")()
        )
    
    def log_file_upload(self, filename: str, file_size: int, user_id: str = None):
        """Log file upload for audit purposes"""
        self.audit_logger.info(
            "File Upload",
            filename=filename,
            file_size=file_size,
            user_id=user_id,
            timestamp=structlog.processors.TimeStamper(fmt="iso")()
        )
    
    def log_query(self, query: str, collection: str, user_id: str = None):
        """Log query for audit purposes"""
        self.audit_logger.info(
            "Query Executed",
            query_hash=hashlib.sha256(query.encode()).hexdigest()[:16],
            collection=collection,
            user_id=user_id,
            timestamp=structlog.processors.TimeStamper(fmt="iso")()
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], user_id: str = None):
        """Log security event"""
        self.audit_logger.warning(
            "Security Event",
            event_type=event_type,
            details=details,
            user_id=user_id,
            timestamp=structlog.processors.TimeStamper(fmt="iso")()
        )
