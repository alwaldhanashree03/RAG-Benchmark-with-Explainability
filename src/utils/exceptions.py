"""Custom exceptions for RAG system.

Industry best practice: Define custom exceptions for better error handling.
"""


class RAGException(Exception):
    """Base exception for RAG system."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid or missing."""
    pass


class DataLoadError(RAGException):
    """Raised when data loading fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class LLMError(RAGException):
    """Raised when LLM API calls fail."""
    pass


class GuardrailError(RAGException):
    """Raised when guardrail checks fail."""
    pass


class ValidationError(RAGException):
    """Raised when input validation fails."""
    pass


class RateLimitError(RAGException):
    """Raised when API rate limits are exceeded."""
    pass
