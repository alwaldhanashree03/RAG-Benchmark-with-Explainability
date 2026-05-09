"""Rate limiting middleware for API endpoints.

Industry best practice: Prevent API abuse with token bucket algorithm.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple
from collections import defaultdict
import time

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """Token bucket rate limiter.
    
    Industry standard: Token bucket algorithm allows burst traffic
    while maintaining average rate limit.
    """
    
    def __init__(self, rate_limit: int = 60, window_seconds: int = 60):
        """Initialize rate limiter.
        
        Args:
            rate_limit: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """Check if request is allowed for client.
        
        Args:
            client_id: Unique client identifier (IP or API key)
            
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove old requests outside window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        current_requests = len(self.requests[client_id])
        
        if current_requests < self.rate_limit:
            self.requests[client_id].append(now)
            return True, self.rate_limit - current_requests - 1
        
        return False, 0
    
    def cleanup_old_entries(self, max_age_hours: int = 24):
        """Cleanup old client entries to prevent memory bloat.
        
        Args:
            max_age_hours: Remove clients inactive for this many hours
        """
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = [
            client_id for client_id, requests in self.requests.items()
            if not requests or max(requests) < cutoff
        ]
        for client_id in to_remove:
            del self.requests[client_id]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting.
    
    Usage:
        app.add_middleware(RateLimitMiddleware, rate_limit=100, window_seconds=60)
    """
    
    def __init__(self, app, rate_limit: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(rate_limit, window_seconds)
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response or 429 Too Many Requests
        """
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        allowed, remaining = self.limiter.is_allowed(client_ip)
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.limiter.rate_limit} requests per {self.limiter.window_seconds} seconds",
                    "retry_after": self.limiter.window_seconds
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.limiter.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(self.limiter.window_seconds)
        
        return response
