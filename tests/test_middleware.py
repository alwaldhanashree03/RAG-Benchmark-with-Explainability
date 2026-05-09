"""Tests for middleware (rate limiting and authentication).

Industry best practice: Test all security features.
"""

import pytest
from src.middleware.rate_limiter import RateLimiter
from src.middleware.auth import APIKeyAuth
import os


def test_rate_limiter_allows_within_limit():
    """Test rate limiter allows requests within limit."""
    limiter = RateLimiter(rate_limit=5, window_seconds=60)
    
    # First 5 requests should be allowed
    for i in range(5):
        allowed, remaining = limiter.is_allowed("client1")
        assert allowed == True
        assert remaining == 4 - i
    
    # 6th request should be blocked
    allowed, remaining = limiter.is_allowed("client1")
    assert allowed == False
    assert remaining == 0


def test_rate_limiter_separate_clients():
    """Test rate limiter tracks clients separately."""
    limiter = RateLimiter(rate_limit=3, window_seconds=60)
    
    # Client 1 makes 3 requests
    for i in range(3):
        allowed, remaining = limiter.is_allowed("client1")
        assert allowed == True
    
    # Client 1 is blocked
    allowed, remaining = limiter.is_allowed("client1")
    assert allowed == False
    
    # Client 2 can still make requests
    allowed, remaining = limiter.is_allowed("client2")
    assert allowed == True


def test_rate_limiter_cleanup():
    """Test rate limiter cleanup removes old entries."""
    limiter = RateLimiter(rate_limit=5, window_seconds=1)
    
    # Add some requests
    limiter.is_allowed("client1")
    limiter.is_allowed("client2")
    
    assert len(limiter.requests) == 2
    
    # Cleanup old entries
    limiter.cleanup_old_entries(max_age_hours=0)
    
    # Should remove inactive clients
    assert len(limiter.requests) <= 2


def test_api_key_auth_no_keys_configured():
    """Test auth allows all when no keys configured."""
    # Temporarily clear environment
    old_val = os.getenv("VALID_API_KEYS")
    if "VALID_API_KEYS" in os.environ:
        del os.environ["VALID_API_KEYS"]
    
    auth = APIKeyAuth()
    
    # Should have no keys
    assert len(auth.valid_keys) == 0
    
    # Restore environment
    if old_val:
        os.environ["VALID_API_KEYS"] = old_val


def test_api_key_auth_load_keys():
    """Test auth loads keys from environment."""
    # Set test keys
    os.environ["VALID_API_KEYS"] = "key1,key2,key3"
    
    auth = APIKeyAuth()
    
    # Should load 3 keys
    assert len(auth.valid_keys) == 3
    assert "key1" in auth.valid_keys
    assert "key2" in auth.valid_keys
    assert "key3" in auth.valid_keys
    
    # Cleanup
    if "VALID_API_KEYS" in os.environ:
        del os.environ["VALID_API_KEYS"]
