"""RAGAS metrics integration for comprehensive RAG evaluation.

Industry best practice: Use RAGAS framework for context-aware evaluation.
Reference: https://docs.ragas.io/
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        answer_relevancy,
        faithfulness,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available. Install with: pip install ragas")

from src.models.base_rag import RAGResponse, RetrievedChunk
from src.models.llm_client import LLMClient


class RAGASEvaluator:
    """RAGAS-based evaluation for RAG systems.
    
    Implements four key RAGAS metrics:
    1. Context Precision: How relevant are the retrieved chunks?
    2. Context Recall: How much of the answer is supported by context?
    3. Answer Relevancy: How relevant is the answer to the query?
    4. Faithfulness: Is the answer faithful to the retrieved context?
    
    Reference: RAGAS - Evaluation framework for RAG systems
    https://arxiv.org/abs/2309.15217
    """

    def __init__(self):
        """Initialize RAGAS evaluator."""
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS not installed. Install with: pip install ragas"
            )
        
        self.llm_client = LLMClient()
        logger.info("RAGASEvaluator initialized")

    def evaluate_response(
        self,
        query: str,
        response: RAGResponse,
        ground_truth_answer: Optional[str] = None,
        ground_truth_contexts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate a RAG response using RAGAS metrics.
        
        Args:
            query: User query
            response: RAG response object
            ground_truth_answer: Reference answer (optional)
            ground_truth_contexts: Reference contexts (optional)
            
        Returns:
            Dictionary of RAGAS metric scores
            
        Example:
            >>> evaluator = RAGASEvaluator()
            >>> scores = evaluator.evaluate_response(
            ...     query="What is AI?",
            ...     response=rag_response,
            ...     ground_truth_answer="AI is artificial intelligence..."
            ... )
            >>> print(scores['faithfulness'])
            0.92
        """
        # Extract retrieved contexts
        retrieved_contexts = [chunk.text for chunk in response.retrieved_chunks]
        
        # Prepare data for RAGAS
        data = {
            "question": [query],
            "answer": [response.answer],
            "contexts": [retrieved_contexts],
        }
        
        # Add ground truth if available
        if ground_truth_answer:
            data["ground_truth"] = [ground_truth_answer]
        if ground_truth_contexts:
            data["ground_truths"] = [ground_truth_contexts]
        
        # Convert to dataset
        dataset = Dataset.from_dict(data)
        
        # Select metrics based on available data
        metrics = [faithfulness, answer_relevancy]
        
        if ground_truth_answer:
            metrics.append(context_recall)
        if ground_truth_contexts:
            metrics.append(context_precision)
        
        try:
            # Run RAGAS evaluation
            results = evaluate(dataset, metrics=metrics)
            
            # Extract scores
            scores = {
                "ragas_faithfulness": results.get("faithfulness", 0.0),
                "ragas_answer_relevancy": results.get("answer_relevancy", 0.0),
            }
            
            if ground_truth_answer:
                scores["ragas_context_recall"] = results.get("context_recall", 0.0)
            if ground_truth_contexts:
                scores["ragas_context_precision"] = results.get("context_precision", 0.0)
            
            logger.debug(f"RAGAS scores: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {
                "ragas_faithfulness": 0.0,
                "ragas_answer_relevancy": 0.0,
                "ragas_context_recall": 0.0,
                "ragas_context_precision": 0.0,
            }

    def evaluate_batch(
        self,
        queries: List[str],
        responses: List[RAGResponse],
        ground_truth_answers: Optional[List[str]] = None,
        ground_truth_contexts: Optional[List[List[str]]] = None,
    ) -> Dict[str, List[float]]:
        """Evaluate multiple RAG responses in batch.
        
        Args:
            queries: List of queries
            responses: List of RAG responses
            ground_truth_answers: List of reference answers (optional)
            ground_truth_contexts: List of reference contexts (optional)
            
        Returns:
            Dictionary mapping metric names to lists of scores
        """
        # Prepare batch data
        questions = queries
        answers = [r.answer for r in responses]
        contexts = [[chunk.text for chunk in r.retrieved_chunks] for r in responses]
        
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truth_answers:
            data["ground_truth"] = ground_truth_answers
        if ground_truth_contexts:
            data["ground_truths"] = ground_truth_contexts
        
        # Convert to dataset
        dataset = Dataset.from_dict(data)
        
        # Select metrics
        metrics = [faithfulness, answer_relevancy]
        if ground_truth_answers:
            metrics.append(context_recall)
        if ground_truth_contexts:
            metrics.append(context_precision)
        
        try:
            # Run batch evaluation
            results = evaluate(dataset, metrics=metrics)
            
            # Convert to dict of lists
            scores = {
                "ragas_faithfulness": results["faithfulness"],
                "ragas_answer_relevancy": results["answer_relevancy"],
            }
            
            if ground_truth_answers:
                scores["ragas_context_recall"] = results["context_recall"]
            if ground_truth_contexts:
                scores["ragas_context_precision"] = results["context_precision"]
            
            logger.info(f"RAGAS batch evaluation complete: {len(queries)} queries")
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS batch evaluation failed: {e}")
            # Return empty lists
            return {
                "ragas_faithfulness": [0.0] * len(queries),
                "ragas_answer_relevancy": [0.0] * len(queries),
                "ragas_context_recall": [0.0] * len(queries),
                "ragas_context_precision": [0.0] * len(queries),
            }

    def compute_context_precision_manual(
        self,
        retrieved_chunks: List[RetrievedChunk],
        ground_truth_contexts: List[str],
    ) -> float:
        """Manually compute context precision.
        
        Context Precision measures what fraction of retrieved contexts
        are actually relevant to answering the question.
        
        Args:
            retrieved_chunks: Retrieved chunks
            ground_truth_contexts: Ground truth relevant contexts
            
        Returns:
            Context precision score (0-1)
        """
        if not retrieved_chunks or not ground_truth_contexts:
            return 0.0
        
        retrieved_texts = [chunk.text for chunk in retrieved_chunks]
        
        # Count how many retrieved contexts match ground truth
        relevant_count = 0
        for retrieved in retrieved_texts:
            for gt in ground_truth_contexts:
                # Simple overlap check (can be improved with semantic similarity)
                if retrieved in gt or gt in retrieved:
                    relevant_count += 1
                    break
        
        precision = relevant_count / len(retrieved_texts)
        return precision

    def compute_context_recall_manual(
        self,
        retrieved_chunks: List[RetrievedChunk],
        ground_truth_answer: str,
    ) -> float:
        """Manually compute context recall.
        
        Context Recall measures what fraction of the ground truth answer
        can be attributed to the retrieved contexts.
        
        Args:
            retrieved_chunks: Retrieved chunks
            ground_truth_answer: Ground truth answer
            
        Returns:
            Context recall score (0-1)
        """
        if not retrieved_chunks or not ground_truth_answer:
            return 0.0
        
        retrieved_texts = [chunk.text for chunk in retrieved_chunks]
        combined_context = " ".join(retrieved_texts)
        
        # Simple word overlap (can be improved with NLI)
        answer_words = set(ground_truth_answer.lower().split())
        context_words = set(combined_context.lower().split())
        
        overlap = answer_words.intersection(context_words)
        recall = len(overlap) / len(answer_words) if answer_words else 0.0
        
        return recall


def get_ragas_evaluator() -> Optional[RAGASEvaluator]:
    """Get RAGAS evaluator instance.
    
    Returns:
        RAGASEvaluator instance or None if RAGAS not available
    """
    if not RAGAS_AVAILABLE:
        logger.warning("RAGAS not available")
        return None
    
    try:
        return RAGASEvaluator()
    except Exception as e:
        logger.error(f"Failed to initialize RAGAS evaluator: {e}")
        return None
