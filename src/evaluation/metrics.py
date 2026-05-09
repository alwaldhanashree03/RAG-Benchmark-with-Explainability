"""Evaluation metrics for RAG systems.

Industry best practice: Use standard IR and NLG metrics.
Reference: 
- Retrieval: Precision@k, MRR (Manning et al., 2008)
- Generation: ROUGE, RAGAS framework
"""

from typing import List, Dict, Any, Optional
import time

import numpy as np
from rouge_score import rouge_scorer
from loguru import logger

from src.models.base_rag import RAGResponse, RetrievedChunk


class RAGMetrics:
    """Comprehensive metrics for RAG evaluation.
    
    Best practices:
    - Retrieval metrics: Precision@k, MRR, Recall@k
    - Generation metrics: ROUGE-L, Faithfulness
    - Operational metrics: Latency, Cost
    
    Reference: RAGAS evaluation framework
    """

    def __init__(self):
        """Initialize metrics calculator."""
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        logger.info("RAGMetrics initialized")

    def evaluate_retrieval(
        self,
        retrieved_chunks: List[RetrievedChunk],
        relevant_ids: List[str],
        k_values: List[int] = [3, 5, 10],
    ) -> Dict[str, float]:
        """Evaluate retrieval quality.
        
        Args:
            retrieved_chunks: Retrieved chunks
            relevant_ids: Ground truth relevant chunk IDs
            k_values: K values for Precision@k and Recall@k
            
        Returns:
            Dictionary of retrieval metrics
        """
        retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        
        metrics = {}
        
        # Precision@k and Recall@k
        for k in k_values:
            precision_k, recall_k = self._precision_recall_at_k(
                retrieved_ids, relevant_ids, k
            )
            metrics[f"precision@{k}"] = precision_k
            metrics[f"recall@{k}"] = recall_k
        
        # MRR (Mean Reciprocal Rank)
        mrr = self._mean_reciprocal_rank(retrieved_ids, relevant_ids)
        metrics["mrr"] = mrr
        
        return metrics

    def _precision_recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
    ) -> tuple[float, float]:
        """Calculate Precision@k and Recall@k.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            Tuple of (precision@k, recall@k)
            
        Reference: Introduction to Information Retrieval (Manning et al., 2008)
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0, 0.0
        
        # Get top-k retrieved
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        # Calculate intersection
        relevant_retrieved = retrieved_at_k.intersection(relevant_set)
        
        # Precision@k = |relevant ∩ retrieved@k| / k
        precision = len(relevant_retrieved) / k if k > 0 else 0.0
        
        # Recall@k = |relevant ∩ retrieved@k| / |relevant|
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
        
        return precision, recall

    def _mean_reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            
        Returns:
            MRR score
            
        Reference: MRR is 1/rank of first relevant document
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0

    def evaluate_generation(
        self,
        generated_answer: str,
        reference_answer: str,
    ) -> Dict[str, float]:
        """Evaluate generation quality.
        
        Args:
            generated_answer: Generated answer
            reference_answer: Reference (ground truth) answer
            
        Returns:
            Dictionary of generation metrics
        """
        metrics = {}
        
        # ROUGE-L
        rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
        metrics["rouge_l_precision"] = rouge_scores['rougeL'].precision
        metrics["rouge_l_recall"] = rouge_scores['rougeL'].recall
        metrics["rouge_l_f1"] = rouge_scores['rougeL'].fmeasure
        
        return metrics

    def evaluate_faithfulness(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> float:
        """Evaluate answer faithfulness to context.
        
        Args:
            answer: Generated answer
            context_chunks: Retrieved context chunks
            
        Returns:
            Faithfulness score (0-1)
            
        Note: Simplified version - ideally use NLI model (implemented in guardrails)
        For full RAGAS faithfulness, we would use NLI to check each claim
        """
        if not context_chunks or not answer:
            return 0.0
        
        # Simplified: Check if key words from answer appear in context
        # In production, use NLI-based claim verification (RAGAS approach)
        
        context_text = " ".join(context_chunks).lower()
        answer_words = set(answer.lower().split())
        
        # Remove stop words (simplified)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
        answer_words = answer_words - stop_words
        
        if not answer_words:
            return 1.0
        
        # Count how many answer words appear in context
        words_in_context = sum(1 for word in answer_words if word in context_text)
        
        faithfulness = words_in_context / len(answer_words)
        
        return faithfulness

    def evaluate_complete(
        self,
        response: RAGResponse,
        relevant_ids: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Complete evaluation of a RAG response.
        
        Args:
            response: RAG response object
            relevant_ids: Ground truth relevant chunk IDs
            reference_answer: Ground truth answer
            start_time: Query start time
            end_time: Query end time
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            "query": response.query,
            "config": response.metadata.get("config_name", "unknown"),
        }
        
        # Retrieval metrics (if ground truth available)
        if relevant_ids:
            retrieval_metrics = self.evaluate_retrieval(
                response.retrieved_chunks,
                relevant_ids,
            )
            metrics.update(retrieval_metrics)
        
        # Generation metrics (if reference available)
        if reference_answer and not response.guardrail_triggered:
            generation_metrics = self.evaluate_generation(
                response.answer,
                reference_answer,
            )
            metrics.update(generation_metrics)
            
            # Faithfulness
            chunk_texts = [c.text for c in response.retrieved_chunks]
            faithfulness = self.evaluate_faithfulness(response.answer, chunk_texts)
            metrics["faithfulness"] = faithfulness
        
        # Operational metrics
        if start_time and end_time:
            latency_ms = (end_time - start_time) * 1000
            metrics["latency_ms"] = latency_ms
        
        # Guardrail info
        metrics["guardrail_triggered"] = response.guardrail_triggered
        metrics["confidence_score"] = response.confidence_score
        
        return metrics


def aggregate_metrics(
    metric_dicts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics across multiple queries.
    
    Args:
        metric_dicts: List of metric dictionaries
        
    Returns:
        Aggregated metrics (mean, std, etc.)
    """
    if not metric_dicts:
        return {}
    
    # Collect numeric metrics
    numeric_metrics = {}
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in numeric_metrics:
                    numeric_metrics[key] = []
                numeric_metrics[key].append(value)
    
    # Calculate statistics
    aggregated = {}
    for key, values in numeric_metrics.items():
        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)
        aggregated[f"{key}_min"] = np.min(values)
        aggregated[f"{key}_max"] = np.max(values)
    
    # Count guardrail triggers
    guardrail_triggers = sum(
        1 for m in metric_dicts if m.get("guardrail_triggered", False)
    )
    aggregated["guardrail_trigger_rate"] = guardrail_triggers / len(metric_dicts)
    
    return aggregated
