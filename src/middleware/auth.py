"""API authentication middleware.

Industry best practice: API key authentication for production endpoints.
"""

import os
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from loguru import logger


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth:
    """API key authentication handler.
    
    Industry standard: Simple API key authentication for microservices.
    For production, consider OAuth2 or JWT tokens.
    """
    
    def __init__(self):
        """Initialize with API keys from environment."""
        # In production, load from secure secrets manager (AWS Secrets Manager, HashiCorp Vault)
        self.valid_keys = self._load_valid_keys()
        
        if not self.valid_keys:
            logger.warning("No API keys configured - authentication disabled")
    
    def _load_valid_keys(self) -> set:
        """Load valid API keys from environment.
        
        Returns:
            Set of valid API keys
        """
        # Support multiple API keys separated by comma
        keys_str = os.getenv("VALID_API_KEYS", "")
        
        if not keys_str:
            return set()
        
        keys = {key.strip() for key in keys_str.split(",") if key.strip()}
        logger.info(f"Loaded {len(keys)} API keys from environment")
        return keys
    
    def verify_key(self, api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
        """Verify API key from request header.
        
        Args:
            api_key: API key from X-API-Key header
            
        Returns:
            Validated API key
            
        Raises:
            HTTPException: 401 if key is invalid or missing
        """
        # If no keys configured, allow all requests (dev mode)
        if not self.valid_keys:
            logger.debug("Auth disabled - no API keys configured")
            return "dev-mode"
        
        # Check if key provided
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "Missing API key",
                    "message": "Please provide X-API-Key header"
                }
            )
        
        # Verify key is valid
        if api_key not in self.valid_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid"
                }
            )
        
        logger.debug(f"API key validated: {api_key[:8]}...")
        return api_key


# Global auth instance
auth_handler = APIKeyAuth()


def get_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Dependency for route authentication.
    
    Usage:
        @app.get("/protected")
        def protected_route(api_key: str = Depends(get_api_key)):
            return {"message": "Access granted"}
    """
    return auth_handler.verify_key(api_key)
