import asyncio
import time
import logging
from typing import Dict, Any, Optional
import structlog
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = structlog.get_logger()

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    concurrent_requests: int = 10
    burst_limit: int = 20
    window_size: int = 60  # seconds

@dataclass
class APIKeyQuota:
    api_key: str
    requests_per_minute: int
    requests_per_hour: int
    concurrent_requests: int
    burst_limit: int
    scopes: list  # ['ingest', 'query', 'admin']
    is_active: bool = True
    created_at: datetime = None
    last_used: datetime = None

class RateLimiter:
    def __init__(self, default_config: RateLimitConfig = None):
        self.default_config = default_config or RateLimitConfig()
        self.api_key_quotas: Dict[str, APIKeyQuota] = {}
        self.request_tracking: Dict[str, deque] = defaultdict(deque)
        self.concurrent_requests: Dict[str, int] = defaultdict(int)
        self.queue_depths: Dict[str, int] = defaultdict(int)
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def add_api_key_quota(self, quota: APIKeyQuota):
        """Add or update API key quota"""
        self.api_key_quotas[quota.api_key] = quota
        logger.info(f"Added quota for API key: {quota.api_key[:8]}...")
    
    def remove_api_key_quota(self, api_key: str):
        """Remove API key quota"""
        if api_key in self.api_key_quotas:
            del self.api_key_quotas[api_key]
            logger.info(f"Removed quota for API key: {api_key[:8]}...")
    
    async def check_rate_limit(self, api_key: str, request_type: str = "query") -> Dict[str, Any]:
        """Check if request is within rate limits"""
        try:
            # Cleanup old tracking data
            await self._cleanup_old_data()
            
            # Get quota for API key
            quota = self.api_key_quotas.get(api_key)
            if not quota or not quota.is_active:
                # Use default config
                config = self.default_config
            else:
                # Check scopes
                if request_type not in quota.scopes:
                    return {
                        "allowed": False,
                        "reason": "Insufficient scope",
                        "retry_after": None
                    }
                
                config = RateLimitConfig(
                    requests_per_minute=quota.requests_per_minute,
                    requests_per_hour=quota.requests_per_hour,
                    concurrent_requests=quota.concurrent_requests,
                    burst_limit=quota.burst_limit
                )
            
            current_time = time.time()
            
            # Check concurrent requests
            if self.concurrent_requests[api_key] >= config.concurrent_requests:
                return {
                    "allowed": False,
                    "reason": "Too many concurrent requests",
                    "retry_after": 1,
                    "current_concurrent": self.concurrent_requests[api_key],
                    "max_concurrent": config.concurrent_requests
                }
            
            # Check rate limits
            rate_check = self._check_rate_limits(api_key, config, current_time)
            if not rate_check["allowed"]:
                return rate_check
            
            # Check burst limit
            burst_check = self._check_burst_limit(api_key, config, current_time)
            if not burst_check["allowed"]:
                return burst_check
            
            # All checks passed
            return {
                "allowed": True,
                "reason": "OK",
                "retry_after": None,
                "current_concurrent": self.concurrent_requests[api_key],
                "max_concurrent": config.concurrent_requests
            }
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request
            return {
                "allowed": True,
                "reason": "Rate limit check failed",
                "retry_after": None
            }
    
    def _check_rate_limits(self, api_key: str, config: RateLimitConfig, current_time: float) -> Dict[str, Any]:
        """Check minute and hour rate limits"""
        requests = self.request_tracking[api_key]
        
        # Remove old requests outside the window
        cutoff_time = current_time - config.window_size
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        # Check minute limit
        minute_cutoff = current_time - 60
        minute_requests = [req_time for req_time in requests if req_time > minute_cutoff]
        
        if len(minute_requests) >= config.requests_per_minute:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded (per minute)",
                "retry_after": 60 - (current_time - minute_requests[0]),
                "current_requests": len(minute_requests),
                "max_requests": config.requests_per_minute
            }
        
        # Check hour limit
        hour_cutoff = current_time - 3600
        hour_requests = [req_time for req_time in requests if req_time > hour_cutoff]
        
        if len(hour_requests) >= config.requests_per_hour:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded (per hour)",
                "retry_after": 3600 - (current_time - hour_requests[0]),
                "current_requests": len(hour_requests),
                "max_requests": config.requests_per_hour
            }
        
        return {"allowed": True}
    
    def _check_burst_limit(self, api_key: str, config: RateLimitConfig, current_time: float) -> Dict[str, Any]:
        """Check burst limit (requests in short time window)"""
        requests = self.request_tracking[api_key]
        
        # Check last 10 seconds for burst
        burst_cutoff = current_time - 10
        burst_requests = [req_time for req_time in requests if req_time > burst_cutoff]
        
        if len(burst_requests) >= config.burst_limit:
            return {
                "allowed": False,
                "reason": "Burst limit exceeded",
                "retry_after": 10 - (current_time - burst_requests[0]),
                "current_burst": len(burst_requests),
                "max_burst": config.burst_limit
            }
        
        return {"allowed": True}
    
    async def record_request(self, api_key: str, request_type: str = "query"):
        """Record a request for rate limiting"""
        current_time = time.time()
        
        # Add to request tracking
        self.request_tracking[api_key].append(current_time)
        
        # Increment concurrent requests
        self.concurrent_requests[api_key] += 1
        
        # Update last used time
        if api_key in self.api_key_quotas:
            self.api_key_quotas[api_key].last_used = datetime.now()
        
        logger.debug(f"Recorded request for API key: {api_key[:8]}...")
    
    async def release_request(self, api_key: str):
        """Release a request (decrement concurrent count)"""
        if self.concurrent_requests[api_key] > 0:
            self.concurrent_requests[api_key] -= 1
            logger.debug(f"Released request for API key: {api_key[:8]}...")
    
    async def _cleanup_old_data(self):
        """Cleanup old tracking data"""
        current_time = time.time()
        
        # Only cleanup every 5 minutes
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        
        # Remove old requests (older than 1 hour)
        cutoff_time = current_time - 3600
        for api_key in list(self.request_tracking.keys()):
            requests = self.request_tracking[api_key]
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Remove empty tracking
            if not requests:
                del self.request_tracking[api_key]
        
        logger.debug("Cleaned up old rate limit data")
    
    def get_queue_depth(self, api_key: str) -> int:
        """Get current queue depth for API key"""
        return self.queue_depths.get(api_key, 0)
    
    def set_queue_depth(self, api_key: str, depth: int):
        """Set queue depth for API key"""
        self.queue_depths[api_key] = depth
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics"""
        total_concurrent = sum(self.concurrent_requests.values())
        total_queue_depth = sum(self.queue_depths.values())
        
        api_key_stats = {}
        for api_key, quota in self.api_key_quotas.items():
            current_concurrent = self.concurrent_requests.get(api_key, 0)
            queue_depth = self.queue_depths.get(api_key, 0)
            
            api_key_stats[api_key[:8] + "..."] = {
                "concurrent_requests": current_concurrent,
                "max_concurrent": quota.concurrent_requests,
                "queue_depth": queue_depth,
                "is_active": quota.is_active,
                "scopes": quota.scopes
            }
        
        return {
            "total_concurrent_requests": total_concurrent,
            "total_queue_depth": total_queue_depth,
            "active_api_keys": len([q for q in self.api_key_quotas.values() if q.is_active]),
            "api_key_stats": api_key_stats
        }

class BackpressureController:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.max_queue_depth = 100
        self.overload_threshold = 0.8
    
    async def should_accept_request(self, api_key: str) -> Dict[str, Any]:
        """Check if request should be accepted based on backpressure"""
        try:
            # Check rate limits first
            rate_check = await self.rate_limiter.check_rate_limit(api_key)
            if not rate_check["allowed"]:
                return rate_check
            
            # Check queue depth
            queue_depth = self.rate_limiter.get_queue_depth(api_key)
            if queue_depth >= self.max_queue_depth:
                return {
                    "allowed": False,
                    "reason": "Queue depth limit exceeded",
                    "retry_after": 5,
                    "current_queue_depth": queue_depth,
                    "max_queue_depth": self.max_queue_depth
                }
            
            # Check system overload
            total_concurrent = sum(self.rate_limiter.concurrent_requests.values())
            total_capacity = sum(q.concurrent_requests for q in self.rate_limiter.api_key_quotas.values())
            
            if total_capacity > 0:
                load_ratio = total_concurrent / total_capacity
                if load_ratio >= self.overload_threshold:
                    return {
                        "allowed": False,
                        "reason": "System overload",
                        "retry_after": 10,
                        "load_ratio": load_ratio,
                        "overload_threshold": self.overload_threshold
                    }
            
            return {
                "allowed": True,
                "reason": "OK",
                "retry_after": None,
                "queue_depth": queue_depth,
                "load_ratio": total_concurrent / total_capacity if total_capacity > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Backpressure check error: {e}")
            return {
                "allowed": True,
                "reason": "Backpressure check failed",
                "retry_after": None
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        metrics = self.rate_limiter.get_metrics()
        
        total_concurrent = metrics["total_concurrent_requests"]
        total_queue_depth = metrics["total_queue_depth"]
        
        # Calculate load ratio
        total_capacity = sum(q.concurrent_requests for q in self.rate_limiter.api_key_quotas.values())
        load_ratio = total_concurrent / total_capacity if total_capacity > 0 else 0
        
        health_status = "healthy"
        if load_ratio >= self.overload_threshold:
            health_status = "overloaded"
        elif total_queue_depth >= self.max_queue_depth * 0.8:
            health_status = "high_queue"
        
        return {
            "status": health_status,
            "load_ratio": load_ratio,
            "total_concurrent_requests": total_concurrent,
            "total_queue_depth": total_queue_depth,
            "max_queue_depth": self.max_queue_depth,
            "overload_threshold": self.overload_threshold,
            "metrics": metrics
        }
