"""Input validation using Pydantic.

Industry best practice: Validate all inputs to prevent injection and crashes.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from src.utils.exceptions import ValidationError


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    model: str = Field(default="baseline", description="RAG model to use")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")
    apply_guardrails: bool = Field(default=True, description="Apply hallucination guardrails")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty or malicious."""
        if not v or not v.strip():
            raise ValidationError("Query cannot be empty")
        
        # Basic SQL injection prevention
        forbidden = ['DROP', 'DELETE', 'INSERT', 'UPDATE', '--', ';']
        query_upper = v.upper()
        for word in forbidden:
            if word in query_upper:
                raise ValidationError(f"Query contains forbidden keyword: {word}")
        
        return v.strip()
    
    @validator('model')
    def validate_model(cls, v):
        """Validate model name."""
        allowed_models = ['baseline', 'hybrid', 'reranker', 'query_decomposition']
        if v not in allowed_models:
            raise ValidationError(f"Invalid model: {v}. Allowed: {allowed_models}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "model": "baseline",
                "top_k": 3,
                "apply_guardrails": True
            }
        }


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    
    query: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    guardrail_triggered: bool
    guardrail_reason: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is a subset of AI...",
                "confidence_score": 0.85,
                "guardrail_triggered": False,
                "guardrail_reason": None,
                "retrieved_chunks": [],
                "metadata": {"latency_ms": 1234, "cost_usd": 0.0012}
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    vector_store_count: int = Field(ge=0)
    models_available: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "vector_store_count": 10000,
                "models_available": ["baseline", "hybrid", "reranker", "query_decomposition"]
            }
        }


def validate_file_upload(filename: str, max_size_mb: int = 10) -> None:
    """Validate uploaded file.
    
    Args:
        filename: Name of uploaded file
        max_size_mb: Maximum file size in MB
        
    Raises:
        ValidationError: If file is invalid
    """
    allowed_extensions = ['.txt', '.json', '.jsonl', '.csv', '.pdf', '.docx']
    
    # Check extension
    ext = '.' + filename.split('.')[-1].lower()
    if ext not in allowed_extensions:
        raise ValidationError(
            f"Invalid file extension: {ext}. Allowed: {allowed_extensions}"
        )
    
    # Check for path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise ValidationError("Invalid filename: contains path traversal characters")
