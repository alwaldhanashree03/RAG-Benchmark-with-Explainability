"""Hallucination guardrails using retrieval scores and NLI.

Industry best practice: Multi-layer guardrails for hallucination detection.
Reference: 
- Retrieval score thresholding
- NLI-based entailment checking (DeBERTa-NLI)
"""

from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

from src.models.base_rag import RetrievedChunk
from src.utils.config_loader import get_config


class GuardrailChecker:
    """Check for hallucination risks using multiple guardrails.
    
    Best practices:
    1. Retrieval score threshold - refuse if retrieved context has low similarity
    2. NLI-based entailment - check if answer is entailed by context
    
    Reference: DeBERTa-v3-large for NLI (SuperGLUE benchmark leader)
    """

    def __init__(self):
        """Initialize guardrail checker."""
        self.config = get_config()
        
        # Retrieval threshold configuration
        self.retrieval_threshold = self.config.get("guardrails.retrieval_threshold", 0.6)
        
        # NLI configuration
        self.nli_enabled = self.config.get("guardrails.nli_enabled", True)
        self.nli_threshold = self.config.get("guardrails.nli_threshold", 0.5)
        
        # Confidence levels
        self.confidence_levels = self.config.get("guardrails.confidence_levels", {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
        })
        
        # Initialize NLI model if enabled
        self.nli_model = None
        self.nli_tokenizer = None
        
        if self.nli_enabled:
            self._load_nli_model()
        
        logger.info(
            f"GuardrailChecker initialized: retrieval_threshold={self.retrieval_threshold}, "
            f"nli_enabled={self.nli_enabled}"
        )

    def _load_nli_model(self) -> None:
        """Load NLI model for entailment checking.
        
        Uses facebook/bart-large-mnli - proven, widely used, stable NLI model.
        """
        # Use proven working model - facebook/bart-large-mnli
        # This is the standard, reliable NLI model used in production
        model_name = self.config.get("guardrails.nli_model", "facebook/bart-large-mnli")
        
        try:
            logger.info(f"Loading NLI model: {model_name}")
            
            self.nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.nli_model.to(device)
            self.nli_model.eval()
            
            logger.info(f"NLI model '{model_name}' loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load NLI model '{model_name}': {e}")
            logger.warning("NLI guardrail will be disabled - system will use retrieval threshold only")
            self.nli_enabled = False

    def check_guardrails(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        answer: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], dict]:
        """Check all guardrails.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            answer: Generated answer (optional, for NLI check)
            
        Returns:
            Tuple of (triggered, reason, details)
            - triggered: Whether guardrails were triggered
            - reason: Human-readable reason if triggered
            - details: Dictionary with detailed scores
        """
        details = {}
        triggered = False
        reason = None
        
        # 1. Retrieval score check
        retrieval_triggered, retrieval_reason, retrieval_score = self._check_retrieval_scores(chunks)
        details["retrieval_score"] = retrieval_score
        details["retrieval_threshold"] = self.retrieval_threshold
        
        if retrieval_triggered:
            triggered = True
            reason = retrieval_reason
        
        # 2. NLI entailment check (if answer provided)
        nli_score = None
        if self.nli_enabled and answer:
            nli_triggered, nli_reason, nli_score = self._check_nli_entailment(chunks, answer)
            details["nli_score"] = nli_score
            details["nli_threshold"] = self.nli_threshold
            
            if nli_triggered and not triggered:
                triggered = True
                reason = nli_reason
        
        # 3. ALWAYS calculate overall confidence (even if guardrail triggered)
        confidence_level = self._calculate_confidence_level(retrieval_score, nli_score)
        details["confidence_level"] = confidence_level
        
        return triggered, reason, details

    def _check_retrieval_scores(
        self,
        chunks: List[RetrievedChunk],
    ) -> Tuple[bool, Optional[str], float]:
        """Check if retrieval scores are above threshold.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (triggered, reason, max_score)
        """
        if not chunks:
            return True, "No relevant context found", 0.0
        
        # Get max score
        max_score = max(chunk.score for chunk in chunks)
        
        if max_score < self.retrieval_threshold:
            return (
                True,
                f"Retrieved context confidence too low ({max_score:.2f} < {self.retrieval_threshold})",
                max_score,
            )
        
        return False, None, max_score

    def _check_nli_entailment(
        self,
        chunks: List[RetrievedChunk],
        answer: str,
    ) -> Tuple[bool, Optional[str], float]:
        """Check if answer is entailed by retrieved chunks using NLI.
        
        Args:
            chunks: Retrieved chunks
            answer: Generated answer
            
        Returns:
            Tuple of (triggered, reason, entailment_score)
            
        Best practice: Use NLI to verify answer is supported by context
        """
        if not self.nli_enabled or not self.nli_model:
            return False, None, 1.0
        
        try:
            # Combine chunks as premise
            premise = " ".join([chunk.text for chunk in chunks])
            
            # Check entailment
            entailment_score = self._compute_entailment(premise, answer)
            
            if entailment_score < self.nli_threshold:
                return (
                    True,
                    f"Answer not sufficiently supported by context (entailment: {entailment_score:.2f})",
                    entailment_score,
                )
            
            return False, None, entailment_score
            
        except Exception as e:
            logger.error(f"NLI check failed: {e}")
            # Don't trigger on error
            return False, None, 1.0

    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """Compute entailment probability using NLI model.
        
        Args:
            premise: Context (retrieved chunks)
            hypothesis: Generated answer
            
        Returns:
            Entailment probability (0-1)
        """
        # Tokenize
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Move to same device as model
        device = next(self.nli_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # DeBERTa-NLI label mapping: 0=contradiction, 1=neutral, 2=entailment
        entailment_prob = probs[0][2].item()
        
        return entailment_prob

    def _calculate_confidence_level(
        self,
        retrieval_score: float,
        nli_score: Optional[float] = None,
    ) -> str:
        """Calculate overall confidence level.
        
        Args:
            retrieval_score: Retrieval similarity score
            nli_score: NLI entailment score (optional)
            
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        # Use average of available scores
        scores = [retrieval_score]
        if nli_score is not None:
            scores.append(nli_score)
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= self.confidence_levels["high"]:
            return "high"
        elif avg_score >= self.confidence_levels["medium"]:
            return "medium"
        else:
            return "low"

    def get_confidence_message(self, confidence_level: str) -> str:
        """Get user-friendly message for confidence level.
        
        Args:
            confidence_level: Confidence level
            
        Returns:
            User-friendly message
        """
        messages = {
            "high": "High confidence - answer is well-supported by retrieved context",
            "medium": "Medium confidence - answer has some support from context",
            "low": "Low confidence - limited support from retrieved context",
        }
        return messages.get(confidence_level, "Unknown confidence level")
