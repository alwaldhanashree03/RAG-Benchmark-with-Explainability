"""Base RAG implementation with common functionality.

Industry best practice: Abstract base class for different RAG configurations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger


@dataclass
class RetrievedChunk:
    """Container for retrieved chunk with metadata."""
    
    chunk_id: str
    text: str
    score: float
    metadata: dict
    rank: int


@dataclass
class RAGResponse:
    """Container for RAG response with explainability."""
    
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    confidence_score: float
    guardrail_triggered: bool
    guardrail_reason: Optional[str]
    metadata: dict


class BaseRAG(ABC):
    """Abstract base class for RAG implementations.
    
    Best practice: Define common interface for all RAG configurations.
    This enables easy benchmarking and comparison.
    """

    def __init__(self, config_name: str):
        """Initialize base RAG.
        
        Args:
            config_name: Name of RAG configuration
        """
        self.config_name = config_name
        logger.info(f"Initialized {self.__class__.__name__}: {config_name}")

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with scores
        """
        pass

    @abstractmethod
    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Generate answer from query and retrieved chunks.
        
        Args:
            query: Query text
            chunks: Retrieved chunks
            
        Returns:
            Generated answer
        """
        pass

    def answer(
        self,
        query: str,
        top_k: int = 3,
        apply_guardrails: bool = True,
    ) -> RAGResponse:
        """Complete RAG pipeline: retrieve + generate.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            apply_guardrails: Whether to apply hallucination guardrails
            
        Returns:
            RAG response with explainability
            
        Best practice: Return full response with retrieved context for explainability
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(query, top_k)
        
        # Check guardrails
        guardrail_triggered = False
        guardrail_reason = None
        
        if apply_guardrails:
            guardrail_triggered, guardrail_reason = self._check_guardrails(query, chunks)
        
        # Generate answer if guardrails pass
        if not guardrail_triggered:
            answer = self.generate(query, chunks)
        else:
            answer = self._get_guardrail_response(guardrail_reason)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(chunks)
        
        # Create response
        response = RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=chunks,
            confidence_score=confidence_score,
            guardrail_triggered=guardrail_triggered,
            guardrail_reason=guardrail_reason,
            metadata={
                "config_name": self.config_name,
                "num_chunks_retrieved": len(chunks),
            },
        )
        
        return response

    def _check_guardrails(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> tuple[bool, Optional[str]]:
        """Check if guardrails should be triggered.
        
        Args:
            query: Query text
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (triggered, reason)
            
        This is a placeholder - will be implemented in guardrails module
        """
        # Will be implemented with actual guardrail logic
        return False, None

    def _calculate_confidence(self, chunks: List[RetrievedChunk]) -> float:
        """Calculate confidence score based on retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Confidence score (0-1)
            
        Best practice: Use retrieval scores as proxy for confidence
        """
        if not chunks:
            return 0.0
        
        # Use max score as confidence
        max_score = max(chunk.score for chunk in chunks)
        return max_score

    def _get_guardrail_response(self, reason: Optional[str]) -> str:
        """Get response when guardrails are triggered.
        
        Args:
            reason: Reason for triggering guardrails
            
        Returns:
            Guardrail response message
        """
        return (
            f"I don't have enough confident information to answer this question. "
            f"Reason: {reason or 'Low confidence in retrieved information'}. "
            f"Please rephrase your question or provide more context."
        )

    @property
    def name(self) -> str:
        """Get configuration name."""
        return self.config_name
